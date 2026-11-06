"""Local repository ingestion into Docker; never bind mount the user's checkout."""

import io
import subprocess
import tarfile
from dataclasses import replace
from pathlib import Path

from docker.errors import DockerException

from codepilot.benchmark.base import Task
from codepilot.sandbox._repository_operations import file_entry, repository_path
from codepilot.sandbox.base import SandboxError
from codepilot.sandbox.docker import DockerSandbox, DockerSandboxConfig
from codepilot.sandbox.repository import RepositorySnapshot


def local_snapshot(repo: Path) -> RepositorySnapshot:
    repo = repo.resolve()
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    )
    if Path(result.stdout.strip()).resolve() != repo:
        raise ValueError("--repo must be the repository root")
    names = (
        subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
            ],
            capture_output=True,
            check=True,
        )
        .stdout.decode()
        .split("\0")
    )
    entries = {}
    for name in sorted(set(names) - {""}):
        target = repository_path(repo, name)
        if target.is_dir() and not target.is_symlink():
            raise ValueError("Submodules are not supported: " + name)
        entry = file_entry(target)
        if entry is not None:
            entries[name] = entry
    return RepositorySnapshot(entries)


class LocalDockerSandbox(DockerSandbox):
    def __init__(self, repo: Path, image: str = "codepilot-cli:local", *, client=None):
        self.repo = Path(repo).resolve()
        self.local_baseline = local_snapshot(self.repo)
        self.local_image = image
        # Reuse Docker execution, snapshots, patch collection and enforced forks.
        super().__init__(
            Task("cli", str(self.repo), "local", ""),
            DockerSandboxConfig(remove_on_stop=False),
            client=client,
        )
        self.workdir = "/testbed"
        self.container_user = "0:0"

    @property
    def image_name(self):
        return self.local_image

    def start(self):
        if self.is_running:
            return
        if self.container is not None:
            self.container.start()
            return
        try:
            self.client.ping()
            self.client.images.get(self.image_name)
            self.container = self.client.containers.create(
                image=self.image_name,
                name=self.container_name,
                detach=True,
                entrypoint=["/bin/sh", "-c"],
                command=["exec tail -f /dev/null"],
                network_disabled=True,
            )
            self.container.start()
            self.container.exec_run(["/bin/sh", "-c", "rm -rf /testbed && mkdir -p /testbed"])
            from tempfile import TemporaryDirectory

            with TemporaryDirectory(prefix="codepilot-import-") as directory:
                root = Path(directory)
                self.local_baseline.materialize(root)
                data = io.BytesIO()
                with tarfile.open(fileobj=data, mode="w") as archive:
                    for name in self.local_baseline.entries:
                        archive.add(root / name, arcname=name, recursive=False)
                if not self.container.put_archive(self.workdir, data.getvalue()):
                    raise SandboxError("Could not populate Docker working copy")
            self.execute(
                "git init -q && git add -f -A && "
                "git -c user.name=CodePilot -c user.email=local@codepilot.invalid "
                "commit --allow-empty -qm 'CLI baseline'",
                check=True,
            )
            self._initial_head = self.head_commit()
            self.execute(
                "printf %s " + self.container_name + " > /codepilot-workspace-id", check=True
            )
        except BaseException:
            if self.container is not None:
                self.container.remove(force=True)
                self.container = None
            raise

    def fork(self, baseline, allowed_files):
        child = super().fork(baseline, allowed_files)
        # Only the main session copy is persistent; child FUSE mounts are temporary.
        child.config = replace(child.config, remove_on_stop=True)
        return child

    def stop(self):
        # Session state owns lifetime. Stopping preserves the container filesystem.
        if self.container is not None:
            try:
                self.container.stop(timeout=3)
            except DockerException as exc:
                raise SandboxError(f"Could not stop working copy: {exc}") from exc

    def interrupt(self):
        # Stop container processes, then retain and restart the same filesystem.
        self.stop()
        self.start()

    def describe(self):
        return {
            "container_id": self.container.id,
            "container_name": self.container_name,
            "image": self.local_image,
            "initial_head": self.initial_head,
            "repo": str(self.repo),
            "baseline": self.local_baseline.entries,
        }

    @classmethod
    def restore(cls, descriptor):
        import shlex

        obj = cls.__new__(cls)
        obj.repo = Path(descriptor["repo"])
        obj.local_image = descriptor["image"]
        obj.local_baseline = RepositorySnapshot(descriptor["baseline"])
        DockerSandbox.__init__(
            obj, Task("cli", str(obj.repo), "local", ""), DockerSandboxConfig(remove_on_stop=False)
        )
        obj.workdir = "/testbed"
        obj.container_user = "0:0"
        obj.container_name = descriptor["container_name"]
        obj._initial_head = descriptor["initial_head"]
        try:
            obj.container = obj.client.containers.get(descriptor["container_id"])
            obj.container.start()
            marker = obj.execute("cat /codepilot-workspace-id", check=True).stdout
            if marker != obj.container_name:
                raise SandboxError("Working copy identity changed")
            obj.execute(
                "git cat-file -e " + shlex.quote(obj.initial_head + "^{commit}"), check=True
            )
            return obj
        except Exception as exc:
            obj.stop()
            raise SandboxError(
                f"Cannot fully restore session: working copy missing or invalid: {exc}"
            ) from exc
