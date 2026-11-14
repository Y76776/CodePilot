"""Configurable live chat-completions provider."""

from dataclasses import dataclass
from urllib.parse import urlsplit


@dataclass(frozen=True)
class ModelConfig:
    name: str = "deepseek-flash"
    base_url: str = "https://api.deepseek.com"
    window_tokens: int = 1_000_000
    max_output_tokens: int = 32_000
    effort: str = "default"
    thinking_parameter: str = "auto"

    def __post_init__(self):
        if self.effort not in {"default", "on", "off"}:
            raise ValueError("Effort must be default, on or off")
        if self.thinking_parameter not in {
            "auto",
            "thinking",
            "enable_thinking",
            "reasoning_effort",
            "chat_template_kwargs",
        }:
            raise ValueError("Unknown thinking_parameter")


def thinking_parameter(config):
    parameter = config.thinking_parameter
    if parameter == "auto":
        host = urlsplit(config.base_url).hostname or ""
        if host == "api.openai.com":
            parameter = "reasoning_effort"
        elif "qwen" in config.name.lower() or "dashscope" in host:
            parameter = "enable_thinking"
        else:
            parameter = "thinking"
    return parameter


def effort_options(config):
    if config.effort == "default":
        return {}
    parameter = thinking_parameter(config)
    enabled = config.effort == "on"
    if parameter == "reasoning_effort":
        return {"reasoning_effort": "medium" if enabled else "none"}
    if parameter == "thinking":
        body = {"thinking": {"type": "enabled" if enabled else "disabled"}}
    elif parameter == "chat_template_kwargs":
        body = {"chat_template_kwargs": {"enable_thinking": enabled}}
    else:
        body = {"enable_thinking": enabled}
    return {"extra_body": body}


class ProviderError(ValueError):
    """A provider failure safe to display and store in trajectories."""

    code: str | None = None


class LiveModel:
    """Late-bound provider shared by sessions, tools and summary calls."""

    def __init__(self, provider=None, *, owner=None, options=None, tools=None):
        self.owner = owner or self
        self.provider = provider
        self.options = options or {}
        self.tools = tools

    def bind(self, **kwargs):
        return LiveModel(owner=self.owner, options={**self.options, **kwargs}, tools=self.tools)

    def bind_tools(self, tools, **kwargs):
        return LiveModel(owner=self.owner, options={**self.options, **kwargs}, tools=tools)

    def invoke(self, messages):
        provider = self.owner.provider
        if provider is None:
            raise ProviderError("Use \\login (or /login) to configure the model first.")
        try:
            bound = (
                provider.bind_tools(self.tools, **self.options)
                if self.tools is not None
                else provider.bind(**self.options)
            )
            return bound.invoke(messages)
        except Exception as exc:  # noqa: BLE001 - Prevent secrets entering trajectory errors.
            # Provider bodies can echo credentials; never archive or print them.
            status = getattr(exc, "status_code", None)
            error = ProviderError(
                f"Model request failed{f' (HTTP {status})' if isinstance(status, int) else ''}. "
                "Check API Key, Base URL, model name and connection; use /login to reconfigure."
            )
            error.code = getattr(exc, "code", None)
            raise error from None


def create_provider(config, api_key):
    from pydantic import SecretStr

    from codepilot.session.provider import CompatibleChatOpenAI

    options = effort_options(config)
    return CompatibleChatOpenAI(
        model=config.name,
        base_url=config.base_url,
        api_key=SecretStr(api_key),
        max_completion_tokens=config.max_output_tokens,
        timeout=120,
        max_retries=1,
        use_responses_api=False,
        replay_reasoning=config.effort != "off" and thinking_parameter(config) == "thinking",
        **options,
    )
