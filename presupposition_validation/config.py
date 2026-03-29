"""Pydantic 类型安全配置模型

默认值从 _conf_schema.json 动态加载，保证与 AstrBot WebUI 的配置保持一致。
AstrBot 通过 _conf_schema.json 驱动 WebUI 渲染，本模型为 main.py 提供类型安全访问。
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Literal

from astrbot.api import logger
from pydantic import BaseModel, ConfigDict, Field

_SCHEMA_PATH = Path(__file__).parent / "_conf_schema.json"
_schema_cache: dict | None = None
_schema_lock = threading.Lock()


def _load_schema() -> dict:
    global _schema_cache
    if _schema_cache is not None:
        return _schema_cache
    with _schema_lock:
        if _schema_cache is not None:
            return _schema_cache
        try:
            if _SCHEMA_PATH.exists():
                _schema_cache = json.loads(_SCHEMA_PATH.read_text("utf-8"))
            else:
                _schema_cache = {}
        except Exception:
            logger.error(
                "[presupposition_validation] 加载 _conf_schema.json 失败",
                exc_info=True,
            )
            _schema_cache = {}
    return _schema_cache


def invalidate_schema_cache() -> None:
    global _schema_cache
    _schema_cache = None


def _schema_default(key: str, fallback: str = "") -> str:
    schema = _load_schema()
    return str(schema.get(key, {}).get("default", fallback))


class PluginConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # ── ⚙️ 核心基础设置 ─────────────────────────────────────

    enabled: bool = Field(
        default=True,
        title="启用前提核查",
        description="总开关。关闭后插件将完全不起作用，所有消息直接放行。",
    )
    api_request_timeout: int = Field(
        default=15,
        title="请求超时时间（秒）",
        description="所有 LLM 和网络请求的超时秒数。超时后自动放行，不阻塞正常对话。",
    )
    enable_async_mode: bool = Field(
        default=False,
        title="异步并行核查（实验性）",
        description="开启后核查任务在后台异步执行，不阻塞主回复流程。降低闲聊延迟但可能增加复杂度。",
    )
    enable_recall: bool = Field(
        default=False,
        title="开启自动撤回",
        description="异步模式下，发现事实错误且主回复已发出时，尝试撤回错误消息（仅 QQ/aiocqhttp，需管理员权限）。",
    )
    enable_meme_detect: bool = Field(
        default=True,
        title="开启跟风检测",
        description="开启后，将拦截群聊中高度相似的复读造句。仅群聊生效。",
    )
    history_window_size: int = Field(
        default=5,
        title="上下文记忆长度",
        description="记录最近 N 条群聊消息用于相似度比对，建议 3 ~ 10。",
    )
    similarity_threshold: float = Field(
        default=0.75,
        title="跟风相似度阈值",
        description="0 ~ 1 之间，越接近 1 要求越严格。建议 0.65 ~ 0.85。",
    )
    enable_argumentative_mode: bool = Field(
        default=False,
        title="对线模式（实验性）",
        description="【实验性】开启最极致的逻辑审查，不仅核查事实，还会对因果关系、排他性逻辑进行深度「对线」。注入逻辑学专家人格，识别 XOR、蕴含等复杂逻辑谬误。",
    )

    # ── 🧠 模式与策略选择 ─────────────────────────────────────

    fact_check_method: Literal["llm", "web_search"] = Field(
        default="llm",
        title="事实核查模式",
        description="llm：仅依靠大模型常识判断，单次调用完成全部核查；web_search：大模型无法判定时按需触发联网搜索。",
    )
    action_mode: Literal["intercept", "warn_and_answer"] = Field(
        default="warn_and_answer",
        title="核查结果处理模式",
        description="intercept：直接拦截并指出错误，不回答原问题；warn_and_answer：指出错误但仍尝试回答。",
    )
    meme_detect_method: Literal["algorithm", "llm"] = Field(
        default="algorithm",
        title="跟风判定算法",
        description="algorithm：纯文本相似度算法（快）；llm：算法初筛 + 大模型二次确认（准但多一次 API 调用）。",
    )
    meme_response_mode: Literal["fixed", "dynamic_llm"] = Field(
        default="dynamic_llm",
        title="跟风回复策略",
        description="fixed：使用固定文案吐槽；dynamic_llm：调用 LLM 基于人设动态生成吐槽。",
    )
    meme_action_mode: Literal["intercept", "check_anyway"] = Field(
        default="intercept",
        title="拦截后放行策略",
        description="intercept：吐槽后直接拦截；check_anyway：吐槽后仍然继续核查事实，将吐槽与正式回答拼接。",
    )

    # ── 📝 Prompt 模板 ──────────────────────────────────────────

    unified_check_prompt: str = Field(
        default=_schema_default("unified_check_prompt"),
        title="[Prompt] 全能预审指令",
        description="一次性完成意图判定、原子化前提提取、逐前提事实核查与逻辑关系识别。输出 JSON（is_factual_question, premises数组, premise_truths数组, premise_relation, corrections数组, needs_search）。",
    )
    argumentative_prompt_appendix: str = Field(
        default=_schema_default("argumentative_prompt_appendix"),
        title="[Prompt] 对线模式附加指令",
        description="对线模式开启时追加到全能预审 Prompt 末尾的逻辑学专家指令。仅在 enable_argumentative_mode 为 True 时生效。",
    )
    meme_llm_check_prompt: str = Field(
        default=_schema_default("meme_llm_check_prompt"),
        title="[Prompt] 跟风判定系统指令",
        description="用于 LLM 二次确认是否为跟风造句的 System Prompt。要求仅回复 True 或 False。",
    )
    meme_dynamic_reply_prompt: str = Field(
        default=_schema_default("meme_dynamic_reply_prompt"),
        title="[Prompt] 动态吐槽系统指令",
        description="动态生成跟风吐槽回复的 System Prompt。"
                    "可用占位符：{bot_persona}、{original_message}、{copy_message}",
    )
    search_verify_prompt: str = Field(
        default=_schema_default("search_verify_prompt"),
        title="[Prompt] 搜索结果验证指令",
        description="联网搜索模式下根据搜索结果验证前提的 System Prompt。",
    )

    # ── 💬 固定文案 ────────────────────────────────────────────

    meme_fixed_reply_text: str = Field(
        default=_schema_default("meme_fixed_reply_text", "检测到高相似度句式，拒绝跟风造句！"),
        title="固定吐槽文案",
        description="跟风命中且 meme_response_mode 为 fixed 时的回复内容。若留空则不发送此提示，实现静默拦截。",
    )
    error_fallback_text: str = Field(
        default=_schema_default("error_fallback_text", "抱歉，我的大脑暂时短路了，没法核查这句话的真伪……"),
        title="错误兜底文案",
        description="遇到 LLM API 崩溃或网络超时时的兜底回复。若留空则保持沉默，直接放行原始消息。",
    )
    recall_success_prefix: str = Field(
        default=_schema_default("recall_success_prefix", "❌ 已撤回刚才的错误回复，纠正如下："),
        title="撤回成功前缀",
        description="撤回错误消息成功后，纠错消息的开头文案。若留空则直接发送纠错正文，不加前缀。",
    )
    recall_fail_prefix: str = Field(
        default=_schema_default("recall_fail_prefix", "⚠️ 补充更正："),
        title="撤回失败前缀",
        description="撤回失败（或不支持撤回）时，追击纠错消息的开头文案。若留空则直接发送纠错正文，不加前缀。",
    )
    meme_async_roast_prefix: str = Field(
        default=_schema_default("meme_async_roast_prefix", "⚠️ 补充吐槽：{roast}"),
        title="异步追击吐槽前缀",
        description="异步模式下跟风吐槽晚于主回复发出时的追击开头文案。可用占位符：{roast}。若留空则不发送。",
    )
    intercept_message_text: str = Field(
        default=_schema_default("intercept_message_text"),
        title="拦截纠错消息模板",
        description="拦截模式下发送的完整纠错消息。可用占位符：{premise}、{correction}。若留空则静默拦截，不发送任何消息。",
    )
    warning_prefix_text: str = Field(
        default=_schema_default("warning_prefix_text"),
        title="提示注入模板",
        description="warn_and_answer 模式下注入 system_prompt 的提示文案。可用占位符：{premise}、{correction}。",
    )
    correction_followup_text: str = Field(
        default=_schema_default("correction_followup_text"),
        title="追击纠错正文模板",
        description="异步追击纠错时附加在前缀后的正文。可用占位符：{premise}、{correction}。",
    )
    meme_followup_text: str = Field(
        default=_schema_default("meme_followup_text"),
        title="追击跟风拦截模板",
        description="异步模式下跟风拦截追击消息。可用占位符：{original}、{copy}。",
    )
