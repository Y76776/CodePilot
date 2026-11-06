"""Readable snapshot deltas, including changes with no text hunks."""

import json

from codepilot.agent.delegation import describe_patch


def describe_changes(changes):
    lines = []
    for change in changes:
        name = change["path"]
        before, after = change["before"], change["after"]
        old_mode, new_mode = mode(before), mode(after)
        lines.append("diff --git " + json.dumps("a/" + name) + " " + json.dumps("b/" + name) + "\n")
        if before is None:
            lines.append("new file mode " + new_mode + "\n")
        elif after is None:
            lines.append("deleted file mode " + old_mode + "\n")
        elif old_mode != new_mode:
            lines.append(f"old mode {old_mode}\nnew mode {new_mode}\n")
        lines.append(describe_patch([change]))
    return "".join(lines)


def mode(entry):
    if entry is None:
        return ""
    return "120000" if entry["kind"] == "symlink" else format(0o100000 | entry["mode"], "o")
