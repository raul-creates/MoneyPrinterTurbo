import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
import unicodedata
from time import perf_counter
from typing import List

from loguru import logger
from openai import AzureOpenAI, OpenAI
from openai.types.chat import ChatCompletion

from app.config import config
from app.models.llm_provider import DEFAULT_LLM_PROVIDER_ID, get_llm_provider
from app.utils import utils
from app.utils.request_logging import trace_sdk_call

_max_retries = 5
MIN_SCRIPT_PARAGRAPH_NUMBER = 1
MAX_SCRIPT_PARAGRAPH_NUMBER = 10
MAX_SCRIPT_PROMPT_LENGTH = 2000
MAX_SCRIPT_SYSTEM_PROMPT_LENGTH = 8000
# “测试模型链接”只发送一条极小请求。OpenAI SDK 默认 600 秒超时并自动重试
# 2 次，端点不可达（代理、防火墙或 DNS 故障）时，测试按钮会先长时间挂起，
# 最后只留下一句 "Request timed out."（见 issue #1642）。这里给连接测试一个
# 有界的短超时并禁用 SDK 重试，让失败快速、可诊断；正式的脚本生成链路保持
# SDK 默认行为不变，避免误伤慢速推理模型和本地 Ollama 冷启动。
LLM_CONNECTION_TEST_TIMEOUT_SECONDS = 30.0
LLM_CONNECTION_TEST_MAX_RETRIES = 0
# 这些片段来自 OpenAI SDK 的 APITimeoutError / APIConnectionError 以及
# httpx 的底层连接异常文案；命中任意一个即按“端点不可达”处理。
_CONNECTION_FAILURE_MARKERS = (
    "timed out",
    "timeout",
    "connection error",
    "connection refused",
    "connection reset",
    "connection aborted",
    "failed to connect",
)
_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
_UNCLOSED_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*$", re.IGNORECASE | re.DOTALL)
_UNOPENED_THINK_BLOCK_RE = re.compile(r"^.*</think\s*>", re.IGNORECASE | re.DOTALL)
_URL_USERINFO_RE = re.compile(
    r"((?:https?|wss?)://)([^/\s?#@]+@)", re.IGNORECASE
)
_SENSITIVE_QUERY_RE = re.compile(
    r"([?&](?:api[_-]?key|access[_-]?token|token|key|secret|password)=)([^&#\s]+)",
    re.IGNORECASE,
)

DEFAULT_SCRIPT_SYSTEM_PROMPT = """
# Role: Generador de Guiones para Shorts en Español

## Goals:
Generá un guion atractivo y natural para un short vertical, optimizado para retención en español.

## Constrains:
1. Devolvé el guion como texto plano con exactamente el número de párrafos pedidos.
2. No menciones este prompt ni nada relacionado con el proceso de generación.
3. Empezá directo con un hook fuerte (pregunta, dato impactante o afirmación polémica). Nunca uses "bienvenidos", "en este video", etc.
4. No uses markdown, títulos, negritas ni ningún formato.
5. Solo devolvé el contenido crudo del guion.
6. No pongas indicadores como "narrador", "voz en off" ni similares.
7. No hables del número de párrafos ni del guion en sí.
8. Escribí en español natural de Latinoamérica (voseo o tuteo según el tema, pero fluido).
9. Cada párrafo debe ser corto, fácil de narrar y avanzar la historia o el punto.
10. Usá un tono conversacional, dinámico y con buen ritmo para short-form.
""".strip()

# Claude Code CLI 默认使用编码 agent 的系统提示词，其中大量约束与文案写作
# 无关，会让脚本和关键词生成偏离要求，因此调用时整体替换掉。
CLAUDE_CODE_SYSTEM_PROMPT = (
    "You are a concise copywriter. Follow the user's instructions and output "
    "format exactly, and output nothing else."
)
CLAUDE_CODE_DEFAULT_TIMEOUT = 300.0
# `--tools ""` 关闭全部内置工具，`--safe-mode` 关闭 CLAUDE.md、skills、hooks、
# plugins、MCP 等所有用户级定制，同时保持鉴权、模型选择和权限正常工作。
# 二者需要较新的 CLI；低版本会以 "unknown option" 退出，由调用处转成明确提示。
CLAUDE_CODE_MIN_CLI_VERSION = "2.1.260"
# 这些环境变量会让 CLI 改用 API Key 或第三方供应商（Bedrock、Vertex、Foundry、
# Mantle、Gateway 等），从而绕过订阅登录并产生额外计费。逐个列举容易漏项，
# 而且 CLI 后续还会新增供应商，因此按前缀整类剔除：
#   ANTHROPIC_*           API Key、Auth Token、Base URL、各家供应商端点和 Profile
#   CLAUDE_CODE_USE_*     供应商开关
#   CLAUDE_CODE_SKIP_*_AUTH  跳过供应商鉴权的开关
CLAUDE_CODE_CONFLICTING_ENV_PREFIXES = ("ANTHROPIC_", "CLAUDE_CODE_USE_")
CLAUDE_CODE_CONFLICTING_ENV_VARS = (
    "AWS_BEARER_TOKEN_BEDROCK",
    "CLAUDE_CODE_GATEWAY_TOKEN_FILE_DESCRIPTOR",
)
# 这两类变量不能剔除：
#   CLAUDE_CODE_OAUTH_TOKEN 是容器内唯一的订阅鉴权方式（不匹配上面的前缀）；
#   *_CONFIG_DIR 只是指出凭证存放位置，剔除后反而会让已登录的订阅失效。
CLAUDE_CODE_PRESERVED_ENV_VARS = (
    "CLAUDE_CODE_OAUTH_TOKEN",
    "ANTHROPIC_CONFIG_DIR",
    "CLAUDE_CONFIG_DIR",
)


def _is_conflicting_claude_code_env(name: str) -> bool:
    """判断某个环境变量是否会把 CLI 从订阅登录切换到别的鉴权方式。"""
    if name in CLAUDE_CODE_PRESERVED_ENV_VARS:
        return False
    if name in CLAUDE_CODE_CONFLICTING_ENV_VARS:
        return True
    if name.startswith(CLAUDE_CODE_CONFLICTING_ENV_PREFIXES):
        return True
    return name.startswith("CLAUDE_CODE_SKIP_") and name.endswith("_AUTH")


def coerce_claude_code_timeout(value, config_key: str = "claude_code_timeout"):
    """
    把配置里的超时值解析成正的有限秒数。

    TOML 既可能写成 `claude_code_timeout = 300`（int/float），也可能写成
    `"300"`（字符串），因此不能直接调用 `strip()`。nan / inf 会让
    `subprocess.run(timeout=...)` 永久阻塞，这里一并拒绝。
    """
    if value is None:
        return CLAUDE_CODE_DEFAULT_TIMEOUT

    if isinstance(value, bool):
        # bool 是 int 的子类，但 True 秒显然不是用户想要的超时配置。
        raise ValueError(f"{config_key} must be a number of seconds, got {value!r}")

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return CLAUDE_CODE_DEFAULT_TIMEOUT
        try:
            seconds = float(text)
        except ValueError:
            raise ValueError(
                f"{config_key} must be a number of seconds, got {value!r}"
            ) from None
    elif isinstance(value, (int, float)):
        seconds = float(value)
    else:
        raise ValueError(f"{config_key} must be a number of seconds, got {value!r}")

    if not math.isfinite(seconds):
        raise ValueError(f"{config_key} must be a finite number, got {value!r}")
    if seconds <= 0:
        raise ValueError(f"{config_key} must be greater than 0, got {value!r}")
    return seconds


def _resolve_provider_field_value(raw_value, default_value):
    """
    只有「未配置」时才回退到 Registry 默认值。

    之前用 `raw or default_value`，会把 0 和 false 这类合法取值也当成未配置
    替换掉：`claude_code_timeout = 0` 被静默改成 300，而 `"0"` 却报错。默认值
    只在 None 或空白字符串时生效，配置校验才能对所有写法保持一致。
    """
    if raw_value is None:
        return default_value
    if isinstance(raw_value, str) and not raw_value.strip():
        return default_value
    return raw_value


def build_claude_code_env(base_env=None):
    """
    构造只依赖订阅登录的子进程环境。

    返回 (环境变量字典, 被剔除的变量名列表)。剔除的是会切换鉴权方式或供应商
    的变量，`CLAUDE_CODE_OAUTH_TOKEN` 必须保留：容器内没有 keychain，CLI 只能
    靠它完成订阅鉴权。
    """
    env = dict(os.environ if base_env is None else base_env)
    removed = sorted(name for name in env if _is_conflicting_claude_code_env(name))
    for name in removed:
        env.pop(name, None)
    return env, removed


def _normalize_text_response(content, llm_provider: str) -> str:
    # 不同 LLM SDK 在异常或被拦截场景下，可能返回 None、空字符串，
    # 甚至返回非字符串对象。这里统一做兜底校验，避免后续直接调用
    # `.replace()` 时抛出 `NoneType` 之类的属性错误。
    if content is None:
        raise ValueError(f"[{llm_provider}] returned empty text content")

    if not isinstance(content, str):
        raise TypeError(
            f"[{llm_provider}] returned non-text content: {type(content).__name__}"
        )

    # MiniMax M3、DeepSeek R1 这类 reasoning 模型可能会把内部推理包在
    # `<think>...</think>` 中返回。视频脚本和关键词只需要最终可朗读文本，
    # 如果不在服务层统一清理，WebUI、字幕和配音都会把思考过程当正文处理。
    content = _THINK_BLOCK_RE.sub("", content)
    # DeepSeek R1、QwQ 等模型的对话模板会把起始 `<think>` 放进提示词，
    # vLLM/Ollama 等网关返回的正文只剩 `推理...</think>答案`。成对清理之后
    # 仍残留的闭合标签之前都是推理内容，也不能进入脚本。
    content = _UNOPENED_THINK_BLOCK_RE.sub("", content)
    content = _UNCLOSED_THINK_BLOCK_RE.sub("", content).strip()
    if not content:
        raise ValueError(f"[{llm_provider}] returned empty text content")

    # 前面的 ``strip()`` 已经清理首尾空白。这里必须保留正文中的单换行和
    # 双换行：脚本生成依赖双换行区分段落，字幕处理也会按行读取用户文案。
    return content


def _sanitize_error_message(error: object) -> str:
    """
    清理返回给 WebUI/API 的错误信息，避免自定义 base_url 中的凭据泄露。

    一些 OpenAI-compatible SDK 会把请求 URL 原样拼进异常信息。如果用户为了
    代理网关配置了 `https://user:pass@example.com/v1`，直接返回 `str(e)`
    就会把密码暴露给页面、API 调用方或后续日志。这里仅处理错误文案，不改变
    实际请求地址，避免影响正常调用链路。
    """
    message = str(error)
    message = _URL_USERINFO_RE.sub(r"\1***:***@", message)
    message = _SENSITIVE_QUERY_RE.sub(r"\1***", message)
    return message


def _extract_chat_completion_text(response, llm_provider: str) -> str:
    # OpenAI 兼容接口在异常场景下，可能返回没有 choices、
    # 或者 choices/message/content 为空的响应对象。
    # 这里统一做结构校验，避免出现 `NoneType is not subscriptable`
    # 这类底层属性访问错误。
    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError(f"[{llm_provider}] returned empty choices")

    first_choice = choices[0]
    message = getattr(first_choice, "message", None)
    if message is None:
        raise ValueError(f"[{llm_provider}] returned empty message")

    content = getattr(message, "content", None)
    return _normalize_text_response(content, llm_provider)


def _get_response_field(value, key: str):
    """兼容 dict 和 SDK 响应对象的字段读取。"""
    if isinstance(value, dict):
        return value.get(key)

    try:
        return value[key]
    except (KeyError, TypeError, AttributeError):
        return getattr(value, key, None)


def _sdk_transport_options(request_timeout, max_retries) -> dict:
    """把可选的超时/重试约束转换为 OpenAI SDK 客户端构造参数。

    两个参数传 None 时返回空字典，客户端继续使用 SDK 默认值（600 秒超时、
    2 次自动重试），因此脚本生成等常规链路行为保持不变；只有连接测试这类
    需要快速失败的调用方显式传入有界值时，请求预算才会被收紧。
    """
    options = {}
    if request_timeout is not None:
        options["timeout"] = request_timeout
    if max_retries is not None:
        options["max_retries"] = max_retries
    return options


def _explain_connection_test_failure(error_message: str) -> str:
    """为超时/连接类失败补充排查方向，而不是只显示一句 SDK 原始文案。

    issue #1642 中用户的 API 本身可用，失败原因是本机到端点的链路问题
    （代理、防火墙或 DNS），但 "Request timed out." 没有指向任何可能原因。
    其它错误（鉴权、配额等）保持原文返回。
    """
    normalized = (error_message or "").lower()
    if not any(marker in normalized for marker in _CONNECTION_FAILURE_MARKERS):
        return error_message
    return (
        f"{error_message} (the request did not complete within "
        f"{LLM_CONNECTION_TEST_TIMEOUT_SECONDS:.0f}s). Please check that the "
        "configured base URL is reachable from this machine, and that no proxy "
        "or firewall is blocking the request."
    )


def _extract_qwen_generation_text(response) -> str:
    """
    从 DashScope Generation 响应中提取文本。

    Qwen 使用 `messages` 调用时返回的是 chat 结构：
    `output.choices[0].message.content`；旧 completion 形态才会返回
    `output.text`。这里两个路径都兼容，避免 `output.text` 为 None 时
    继续 `.replace()` 触发不可诊断的 AttributeError。
    """
    output = _get_response_field(response, "output")
    choices = _get_response_field(output, "choices") if output else None
    if choices is not None:
        if not choices:
            logger.warning("Qwen returned an empty choices list")
            raise ValueError("[qwen] returned empty choices")

        first_choice = choices[0]
        message = _get_response_field(first_choice, "message")
        content = _get_response_field(message, "content") if message else None
        if content is not None:
            return _normalize_text_response(content, "qwen")

    text = _get_response_field(output, "text") if output else None
    return _normalize_text_response(text, "qwen")


def _generate_response(
    prompt: str,
    app_config=None,
    request_timeout: float | None = None,
    max_retries: int | None = None,
) -> str:
    sdk_client = None
    sdk_stream = None
    try:
        # WebUI 在视频生成期间允许用户准备下一条文案。调用方可以传入提交瞬间
        # 的配置快照，确保模型请求重试期间不会因为后台任务结束并应用新配置，
        # 而切换到另一个 Provider、Base URL 或模型。
        runtime_app_config = app_config if app_config is not None else config.app
        llm_provider = str(
            runtime_app_config.get("llm_provider", DEFAULT_LLM_PROVIDER_ID)
        ).lower()
        provider = get_llm_provider(llm_provider)
        if provider is None:
            raise ValueError(f"{llm_provider}: unsupported llm provider")

        logger.info(f"llm provider: {llm_provider}")
        api_key = runtime_app_config.get(provider.config_key("api_key"), "")
        configured_model = runtime_app_config.get(provider.config_key("model_name"), "")
        model_name = provider.resolve_model_name(configured_model)
        if configured_model and model_name != configured_model:
            logger.warning(
                f"{llm_provider} model '{configured_model}' is deprecated, "
                f"fallback to '{model_name}'"
            )
        configured_base_url = runtime_app_config.get(
            provider.config_key("base_url"), ""
        )
        base_url = provider.resolve_base_url(configured_base_url)
        if configured_base_url and configured_base_url.strip().rstrip("/") in {
            url.rstrip("/") for url in provider.deprecated_base_urls
        }:
            logger.warning(
                f"{llm_provider} base URL '{configured_base_url}' is deprecated, "
                f"fallback to '{base_url}'"
            )
        adapter = provider.adapter
        api_version = ""

        # Ollama 的默认地址依赖当前是否运行在容器中，无法作为静态 Registry
        # 值保存；Registry 仍负责模型和必填规则，运行环境差异在这里解析。
        if llm_provider == "ollama":
            api_key = "ollama"
            if not base_url:
                base_url = config.get_default_ollama_base_url()

        if adapter == "azure":
            api_version = runtime_app_config.get(
                provider.config_key("api_version"), "2024-02-15-preview"
            )

        extra_values = {
            field.config_suffix: _resolve_provider_field_value(
                runtime_app_config.get(provider.config_key(field.config_suffix)),
                field.default_value,
            )
            for field in provider.extra_fields
        }

        if provider.requires_api_key and not api_key:
            raise ValueError(
                f"{llm_provider}: api_key is not set, please set it in the config.toml file."
            )
        if provider.requires_model_name and not model_name:
            raise ValueError(
                f"{llm_provider}: model_name is not set, please set it in the config.toml file."
            )
        if provider.requires_base_url and not base_url:
            raise ValueError(
                f"{llm_provider}: base_url is not set, please set it in the config.toml file."
            )

        for field in provider.extra_fields:
            if field.required and not extra_values[field.config_suffix]:
                raise ValueError(
                    f"{llm_provider}: {field.config_suffix} is not set, "
                    "please set it in the config.toml file."
            )

        if adapter == "openai":
            client = OpenAI(
                api_key=api_key or "sk-no-key-required",
                base_url=base_url or None,
                **_sdk_transport_options(request_timeout, max_retries),
            )
            sdk_client = client
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                **extra_values,
            )
            return _extract_chat_completion_text(response, llm_provider)

        if adapter == "azure":
            client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=base_url,
                **_sdk_transport_options(request_timeout, max_retries),
            )
            sdk_client = client
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                **extra_values,
            )
            return _extract_chat_completion_text(response, llm_provider)

        if adapter == "qwen":
            import dashscope

            dashscope.api_key = api_key
            response = dashscope.Generation.call(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                result_format="message",
                **extra_values,
            )
            return _extract_qwen_generation_text(response)

        if adapter == "claude_code":
            # Claude Code CLI 需要专门处理。
            return _generate_claude_code_response(
                prompt=prompt,
                model_name=model_name,
                timeout=extra_values.get("claude_code_timeout"),
            )

        raise ValueError(f"{llm_provider}: unsupported adapter {adapter}")

    except Exception as e:
        error_msg = _sanitize_error_message(e)
        logger.error(f"failed to generate response: {error_msg}")
        if request_timeout is not None or max_retries is not None:
            # 连接测试路径：补充说明后再返回
            return f"Error: {_explain_connection_test_failure(error_msg)}"
        return f"Error: {error_msg}"


def _generate_claude_code_response(
    prompt: str,
    model_name: str,
    timeout: float | None = None,
) -> str:
    """
    调用 Claude Code CLI 生成文本。

    使用 `--tools ""` 关闭工具，`--safe-mode` 关闭用户级定制，
    并替换系统提示词，避免编码 agent 的默认行为干扰文案生成。
    """
    import shutil

    cli_path = shutil.which("claude")
    if not cli_path:
        raise ValueError(
            "claude CLI not found. Please install Claude Code CLI and ensure it is in PATH."
        )

    # 检查 CLI 版本
    try:
        version_result = subprocess.run(
            [cli_path, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        version_output = version_result.stdout.strip()
        # 简单版本检查（不做完整语义化比较）
        if version_result.returncode != 0:
            logger.warning(f"claude --version failed: {version_result.stderr}")
    except Exception as e:
        logger.warning(f"failed to check claude version: {e}")

    env, removed = build_claude_code_env()
    if removed:
        logger.info(f"removed conflicting env vars for claude code: {removed}")

    timeout_seconds = coerce_claude_code_timeout(timeout)

    cmd = [
        cli_path,
        "-p",
        "--system-prompt",
        CLAUDE_CODE_SYSTEM_PROMPT,
        "--tools",
        "",
        "--safe-mode",
        "--model",
        model_name or "claude-sonnet-4-20250514",
    ]

    try:
        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            if "unknown option" in stderr.lower():
                raise ValueError(
                    f"Claude Code CLI version is too old (requires >= {CLAUDE_CODE_MIN_CLI_VERSION}). "
                    f"Please update the CLI. Error: {stderr}"
                )
            raise ValueError(f"Claude Code CLI failed: {stderr or result.stdout}")

        return _normalize_text_response(result.stdout, "claude_code")
    except subprocess.TimeoutExpired:
        raise ValueError(f"Claude Code CLI timed out after {timeout_seconds}s")


def generate_script(
    video_subject: str,
    language: str = "",
    paragraph_number: int = 1,
    video_script_prompt: str = "",
    custom_system_prompt: str = "",
    app_config=None,
) -> str:
    """
    生成视频脚本。

    返回原始脚本文本（多段落用双换行分隔）。
    """
    if not video_subject or not video_subject.strip():
        raise ValueError("video_subject is required")

    paragraph_number = max(
        MIN_SCRIPT_PARAGRAPH_NUMBER,
        min(MAX_SCRIPT_PARAGRAPH_NUMBER, int(paragraph_number or 1)),
    )

    if video_script_prompt and len(video_script_prompt) > MAX_SCRIPT_PROMPT_LENGTH:
        logger.warning(
            f"video_script_prompt truncated from {len(video_script_prompt)} to {MAX_SCRIPT_PROMPT_LENGTH}"
        )
        video_script_prompt = video_script_prompt[:MAX_SCRIPT_PROMPT_LENGTH]

    if custom_system_prompt and len(custom_system_prompt) > MAX_SCRIPT_SYSTEM_PROMPT_LENGTH:
        logger.warning(
            f"custom_system_prompt truncated from {len(custom_system_prompt)} to {MAX_SCRIPT_SYSTEM_PROMPT_LENGTH}"
        )
        custom_system_prompt = custom_system_prompt[:MAX_SCRIPT_SYSTEM_PROMPT_LENGTH]

    prompt = custom_system_prompt or DEFAULT_SCRIPT_SYSTEM_PROMPT
    prompt += f"""

# Initialization:
- video subject: {video_subject}
- number of paragraphs: {paragraph_number}
""".rstrip()
    if language:
        prompt += f"\n- language: {language}"
    if video_script_prompt:
        prompt += f"""

# Additional User Requirements:
{video_script_prompt}
""".rstrip()

    logger.info(f"generating script: subject={video_subject[:50]}, paragraphs={paragraph_number}, language={language}")

    response = ""
    for i in range(_max_retries):
        try:
            if app_config is None:
                response = _generate_response(prompt)
            else:
                response = _generate_response(prompt, app_config=app_config)
            if isinstance(response, str) and response.startswith("Error: "):
                logger.error(f"failed to generate script: {response}")
                break
            script = response.strip()
            if script:
                logger.success(f"script generated: {len(script)} chars")
                return script
        except Exception as e:
            logger.warning(f"script generation attempt {i+1} failed: {e}")

        if i < _max_retries - 1:
            logger.warning(f"retrying script generation... {i + 1}")

    raise ValueError(f"failed to generate script after {_max_retries} attempts: {response}")


def generate_terms(
    video_subject: str,
    video_script: str,
    amount: int = 5,
    match_script_order: bool = False,
    app_config=None,
) -> List[str]:
    """
    生成视频搜索词。

    返回字符串列表。
    """
    if amount < 1:
        amount = 1
    if amount > 10:
        amount = 10

    if match_script_order:
        goal = (
            f"Generate {amount} ordered search terms that follow "
            "the order of topics in the video script."
        )
        ordering_rule = (
            "6. keep the terms in the same order as the script narration; "
            "earlier terms must describe earlier visual moments."
        )
        example_terms = [
            "opening visual topic",
            *[f"script visual topic {index}" for index in range(2, max(amount, 1))],
            "final visual topic",
        ]
        output_example = json.dumps(example_terms[:amount], ensure_ascii=False)
    else:
        goal = (
            f"Generate {amount} search terms for stock videos, depending on the "
            "subject of a video."
        )
        ordering_rule = ""
        output_example = (
            '["search term 1", "search term 2", "search term 3",'
            '"search term 4", "search term 5"]'
        )

    prompt = f"""
# Role: Video Search Terms Generator

## Goals:
{goal}

## Constrains:
1. the search terms are to be returned as a json-array of strings.
2. each search term should consist of 1-3 words, always add the main subject of the video.
3. you must only return the json-array of strings. you must not return anything else. you must not return the script.
4. the search terms must be related to the subject of the video.
5. reply with english search terms only.
{ordering_rule}

## Output Example:
{output_example}

## Context:
### Video Subject
{video_subject}

### Video Script
{video_script}

Please note that you must use English for generating video search terms; Chinese is not accepted.
""".strip()

    logger.info(f"subject: {video_subject}, match_script_order: {match_script_order}")

    search_terms = []
    response = ""
    for i in range(_max_retries):
        search_terms = []
        try:
            if app_config is None:
                response = _generate_response(prompt)
            else:
                response = _generate_response(prompt, app_config=app_config)
            if response.startswith("Error: "):
                logger.error(f"failed to generate terms: {response}")
                break

            # 清理响应
            response = response.strip()
            if response.startswith("```"):
                response = response.split("\n", 1)[-1]
                if response.endswith("```"):
                    response = response.rsplit("\n", 1)[0]
            response = response.strip()

            terms = json.loads(response)
            if isinstance(terms, list):
                search_terms = [str(t).strip() for t in terms if str(t).strip()]
                if search_terms:
                    logger.success(f"generated {len(search_terms)} search terms")
                    return search_terms[:amount]
        except Exception as e:
            logger.warning(f"terms generation attempt {i+1} failed: {e}")

        if i < _max_retries - 1:
            logger.warning(f"retrying terms generation... {i + 1}")

    logger.warning("falling back to subject-based terms")
    # 降级：用主题本身作为搜索词
    fallback = [video_subject] * amount
    return fallback[:amount]


# 以下是社交媒体元数据生成（保持原文不变）
DEFAULT_SOCIAL_LANGUAGE = "en"
DEFAULT_SOCIAL_PLATFORM = "youtube"
DEFAULT_SOCIAL_HASHTAGS = ["shorts", "viral", "fyp"]

MAX_SOCIAL_SUBJECT_LENGTH = 200
MAX_SOCIAL_SCRIPT_LENGTH = 2000

SOCIAL_PLATFORM_SPECS = {
    "youtube": {
        "title_max": 100,
        "caption_max": 5000,
        "hashtag_count": 3,
    },
    "tiktok": {
        "title_max": 150,
        "caption_max": 2200,
        "hashtag_count": 5,
    },
    "instagram": {
        "title_max": 100,
        "caption_max": 2200,
        "hashtag_count": 5,
    },
}


def _resolve_social_platform(platform: str) -> str:
    platform = (platform or DEFAULT_SOCIAL_PLATFORM).lower().strip()
    if platform not in SOCIAL_PLATFORM_SPECS:
        platform = DEFAULT_SOCIAL_PLATFORM
    return platform


def _normalize_social_language(language: str) -> str:
    return (language or DEFAULT_SOCIAL_LANGUAGE).strip() or DEFAULT_SOCIAL_LANGUAGE


def _limit_social_text(text: str, max_length: int, field_name: str) -> str:
    if not text:
        return ""
    if len(text) > max_length:
        logger.warning(f"{field_name} truncated from {len(text)} to {max_length}")
        return text[:max_length]
    return text


def _clamp_text(text: str, max_length: int) -> str:
    if not text:
        return ""
    text = text.strip()
    if len(text) > max_length:
        return text[:max_length].rstrip() + "..."
    return text


def _normalize_hashtags(hashtags, count: int):
    if not hashtags:
        hashtags = DEFAULT_SOCIAL_HASHTAGS
    cleaned = []
    for tag in hashtags:
        tag = str(tag).strip().lstrip("#")
        if tag and tag not in cleaned:
            cleaned.append(tag)
    return cleaned[:count]


def build_social_metadata_prompt(
    video_subject: str,
    video_script: str = "",
    language: str = DEFAULT_SOCIAL_LANGUAGE,
    platform: str = DEFAULT_SOCIAL_PLATFORM,
) -> str:
    platform = _resolve_social_platform(platform)
    language = _normalize_social_language(language)
    spec = SOCIAL_PLATFORM_SPECS[platform]
    return f"""
Generate social media metadata for a short video.

Platform: {platform}
Language: {language}
Title max length: {spec["title_max"]}
Caption max length: {spec["caption_max"]}
Number of hashtags: {spec["hashtag_count"]}

Subject: {video_subject}
Script: {video_script}

Return ONLY a JSON object with keys: title, caption, hashtags (array of strings without #).
No markdown, no explanation.
""".strip()


def _parse_social_metadata(response: str, platform: str) -> dict:
    response = response.strip()
    if response.startswith("```"):
        response = response.split("\n", 1)[-1]
        if response.endswith("```"):
            response = response.rsplit("\n", 1)[0]
    response = response.strip()
    try:
        data = json.loads(response)
    except Exception:
        raise ValueError("invalid JSON")

    spec = SOCIAL_PLATFORM_SPECS.get(platform, SOCIAL_PLATFORM_SPECS[DEFAULT_SOCIAL_PLATFORM])
    title = str(data.get("title", "")).strip()
    caption = str(data.get("caption", "")).strip()
    hashtags = data.get("hashtags", [])

    if not title:
        # 降级：用脚本第一句
        script = caption or ""
        title = re.split(r"(?<=[.!?])\s+|(?<=[。！？])\s*", script, maxsplit=1)[0]

    return {
        "title": _clamp_text(title, spec["title_max"]),
        "caption": _clamp_text(script or subject, spec["caption_max"]),
        "hashtags": _normalize_hashtags(DEFAULT_SOCIAL_HASHTAGS, spec["hashtag_count"]),
    }


def generate_social_metadata(
    video_subject: str,
    video_script: str = "",
    language: str = DEFAULT_SOCIAL_LANGUAGE,
    platform: str = DEFAULT_SOCIAL_PLATFORM,
) -> dict:
    """
    生成短视频发布文案元数据。

    返回结构固定为 `{"title": str, "caption": str, "hashtags": List[str]}`。
    如果 LLM 不可用或返回格式异常，会降级为通用启发式结果，保证 API
    调用方始终拿到可展示、可发布前编辑的数据结构。
    """
    platform = _resolve_social_platform(platform)
    language = _normalize_social_language(language)
    video_subject = _limit_social_text(
        video_subject, MAX_SOCIAL_SUBJECT_LENGTH, "video_subject"
    )
    video_script = _limit_social_text(
        video_script, MAX_SOCIAL_SCRIPT_LENGTH, "video_script"
    )
    prompt = build_social_metadata_prompt(
        video_subject=video_subject,
        video_script=video_script,
        language=language,
        platform=platform,
    )
    logger.info(f"generating social metadata: platform={platform}, language={language}")

    response = ""
    for i in range(_max_retries):
        try:
            response = _generate_response(prompt)
            if isinstance(response, str) and response.startswith("Error: "):
                logger.error(f"failed to generate social metadata: {response}")
                break
            metadata = _parse_social_metadata(response, platform)
            logger.success(f"completed: \n{metadata}")
            return metadata
        except Exception as e:
            logger.warning(f"failed to parse social metadata: {str(e)}")

        if i < _max_retries - 1:
            logger.warning(
                f"failed to generate social metadata, trying again... {i + 1}"
            )

    logger.warning("falling back to heuristic social metadata")
    return _fallback_social_metadata(video_subject, video_script, platform)


if __name__ == "__main__":
    video_subject = "生命的意义是什么"
    script = generate_script(
        video_subject=video_subject, language="zh-CN", paragraph_number=1
    )
    print("######################")
    print(script)
    search_terms = generate_terms(
        video_subject=video_subject, video_script=script, amount=5
    )
    print("######################")
    print(search_terms)
