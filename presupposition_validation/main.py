"""presupposition_validation - AstrBot 预设前提核查插件

在机器人正式生成回复之前，拦截用户输入，通过单次 LLM 调用完成
意图判定、前提提取与事实核查，避免基于虚假前提生成误导性回答。

使用方式：
1. 将本插件目录放置到 AstrBot 的 data/plugins/ 目录下
2. 在 AstrBot WebUI 的插件管理中启用本插件
3. 在插件配置中选择核查方式（llm / web_search）和处理模式（intercept / warn_and_answer）
"""

import asyncio
import importlib.util
import json
import re
import string as _string
import time as _time
from collections import OrderedDict, deque
from difflib import SequenceMatcher
from typing import Optional

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, register

from .config import PluginConfig


def _parse_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return default


@register(
    "presupposition_validation",
    "presupposition_validation_dev",
    "核查用户提问中隐藏的预设前提，防止基于虚假前提的回答",
    "1.0.9",
)
class PresuppositionValidation(Star):

    _MAX_GROUP_CACHE = 200
    _GROUP_GC_INTERVAL = 300
    _API_CALL_TIMEOUT = 5.0
    _MAX_CONCURRENT_TASKS = 10

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.cfg = PluginConfig(**{k: v for k, v in config.items()})
        self._group_msg_cache: OrderedDict[str, deque[tuple[str, list[str]]]] = OrderedDict()
        self._group_last_active: dict[str, float] = {}
        self._pending_tasks: set[asyncio.Task] = set()
        self._session_sent_events: dict[str, asyncio.Event] = {}
        self._sent_bot_msg_ids: dict[str, int] = {}
        self._task_semaphore = asyncio.Semaphore(self._MAX_CONCURRENT_TASKS)
        self._search_driver: Optional[str] = self._detect_search_driver()

    @staticmethod
    def _get_session_id(event: AstrMessageEvent) -> str:
        if hasattr(event, "get_session_id"):
            sid = event.get_session_id()
            if sid:
                return str(sid)
        return event.unified_msg_origin

    _SEARCH_TOOL_KEYWORDS = (
        "search", "web_search", "tavily", "serp", "bing",
        "google_search", "duckduckgo", "brave_search",
    )

    def _detect_search_driver(self) -> Optional[str]:
        framework_tool = self._probe_framework_search_tool()
        if framework_tool:
            logger.info(
                f"[presupposition_validation] 检测到框架已注册搜索工具 "
                f"「{framework_tool}」，将优先使用框架原生搜索能力"
            )
            return "framework"
        if importlib.util.find_spec("duckduckgo_search") is not None:
            logger.info(
                "[presupposition_validation] 检测到本地 duckduckgo-search，"
                "已启用独立搜索"
            )
            return "duckduckgo"
        logger.info(
            "[presupposition_validation] 未检测到搜索依赖，将降级至纯 LLM 常识校验"
        )
        return None

    def _probe_framework_search_tool(self) -> Optional[str]:
        try:
            tool_mgr = self.context.get_llm_tool_manager()
            for kw in self._SEARCH_TOOL_KEYWORDS:
                if tool_mgr.get_func(kw):
                    return kw
        except Exception:
            pass
        return None

    # ==================================================================
    # 关系评估工具
    # ==================================================================

    @staticmethod
    def _evaluate_relation(
        truths: list[bool],
        relation: str,
        false_indices: list[int],
        premises_len: int,
    ) -> bool:
        if not false_indices:
            return False
        if relation == "or":
            return len(false_indices) == premises_len
        elif relation == "xor":
            return (premises_len - len(false_indices)) != 1
        elif relation == "implication":
            if premises_len >= 2 and truths[0]:
                return any(not t for t in truths[1:])
            return bool(false_indices)
        elif relation == "biconditional":
            if premises_len >= 2:
                return truths[0] != truths[1]
            return bool(false_indices)
        return True

    # ==================================================================
    # 群组缓存 GC
    # ==================================================================

    def _gc_group_cache(self):
        if len(self._group_last_active) <= 100:
            return
        now = _time.monotonic()
        stale = [
            gid for gid, ts in self._group_last_active.items()
            if now - ts > self._GROUP_GC_INTERVAL
        ]
        if not stale:
            return
        for gid in stale:
            self._group_last_active.pop(gid, None)
            self._group_msg_cache.pop(gid, None)
        stale_prefixes = {f"{gid}:" for gid in stale}
        for key in list(self._sent_bot_msg_ids):
            if any(key.startswith(p) for p in stale_prefixes):
                del self._sent_bot_msg_ids[key]
        logger.debug(
            f"[presupposition_validation] GC 清理 {len(stale)} 个过期群组缓存"
        )

    # ==================================================================
    # 核心 Hook
    # ==================================================================

    @filter.on_llm_request()
    async def on_llm_request(
        self, event: AstrMessageEvent, req: ProviderRequest
    ):
        if not self.cfg.enabled:
            return

        self._gc_group_cache()

        user_message = ""
        if hasattr(event, "message_str"):
            try:
                user_message = (event.message_str or "").strip()
            except AttributeError:
                return

        if not user_message or len(user_message) < 3:
            return

        if self.cfg.enable_async_mode:
            task_id = f"{self._get_session_id(event)}:{_time.time_ns()}"
            sent_evt = asyncio.Event()
            self._session_sent_events[task_id] = sent_evt

            async def _guarded_pipeline(t_id=task_id):
                async with self._task_semaphore:
                    await self._cleanup_pipeline(event, req, user_message, sent_evt, t_id)

            task = asyncio.create_task(
                _guarded_pipeline(),
                name="presupposition_validation_check",
            )
            self._pending_tasks.add(task)
            task.add_done_callback(self._pending_tasks.discard)
            await asyncio.sleep(0.05)
            return

        await self._run_pipeline(event, req, user_message, sent_event=None)

    # ==================================================================
    # 主回复发出回调（用于异步模式感知竞态状态）
    # ==================================================================

    @filter.after_message_sent()
    async def on_message_sent(self, event: AstrMessageEvent):
        session = self._get_session_id(event)
        prefix = f"{session}:"
        for key in list(self._session_sent_events):
            if key.startswith(prefix):
                self._session_sent_events[key].set()

    # ==================================================================
    # 核查管线（串行/异步共用）
    # ==================================================================

    async def _cleanup_pipeline(
        self,
        event: AstrMessageEvent,
        req: ProviderRequest,
        user_message: str,
        sent_evt: asyncio.Event,
        evt_key: str,
    ):
        try:
            await self._run_pipeline(event, req, user_message, sent_event=sent_evt)
        except Exception as e:
            logger.error(f"[presupposition_validation] 核查管线异常: {e}")
        finally:
            self._session_sent_events.pop(evt_key, None)

    async def _run_pipeline(
        self,
        event: AstrMessageEvent,
        req: ProviderRequest,
        user_message: str,
        sent_event: Optional[asyncio.Event],
    ):
        logger.debug(
            f"[presupposition_validation] 开始预审消息: {user_message[:60]}..."
        )

        result = await self._unified_llm_check(event, user_message)
        if result is None:
            logger.debug("[presupposition_validation] 预审失败或解析异常，放行")
            return

        is_factual = result["is_factual"]
        premises = result["premises"]
        truths = result["truths"]
        relation = result["relation"]
        corrections = result["corrections"]
        logic_flaw = result.get("logic_flaw", "")
        needs_search = result["needs_search"]

        if not is_factual:
            logger.debug("[presupposition_validation] 非事实性问题，放行")
            return

        logger.debug(
            f"[presupposition_validation] 预审结果: premises={premises}, "
            f"truths={truths}, relation={relation}, needs_search={needs_search}"
        )

        meme_hit = False
        meme_matched_msg: Optional[str] = None

        if self.cfg.enable_meme_detect and event.is_private_chat() is False:
            group_id = event.get_group_id()
            if group_id:
                meme_matched_msg = await self._check_meme_pattern(
                    group_id, user_message, premises
                )

                if meme_matched_msg is not None:
                    confirmed = True
                    if self.cfg.meme_detect_method == "llm":
                        confirmed = await self._llm_verify_meme(
                            event, group_id, user_message, premises
                        )
                        if confirmed is None:
                            confirmed = True

                    if confirmed:
                        meme_hit = True
                        logger.info(
                            f"[presupposition_validation] 跟风造句命中, "
                            f"action={self.cfg.meme_action_mode}"
                        )

        response_sent = sent_event is not None and sent_event.is_set()

        if meme_hit:
            if self.cfg.meme_action_mode == "intercept":
                if response_sent:
                    followup = self._build_meme_followup(meme_matched_msg, user_message)
                    await self._send_withdraw_or_followup(event, "跟风拦截", followup)
                else:
                    event.stop_event()
                    try:
                        await self._send_meme_roast(
                            event, meme_matched_msg, user_message, req.system_prompt
                        )
                    except Exception as e:
                        logger.error(
                            f"[presupposition_validation] 发送跟风拦截回复失败: {e}"
                        )
                return

            if self.cfg.meme_action_mode == "check_anyway":
                if response_sent:
                    try:
                        roast_text = await self._resolve_meme_roast_text(
                            event, meme_matched_msg, user_message, req.system_prompt
                        )
                        if not self._quiet(roast_text):
                            try:
                                msg = self.cfg.meme_async_roast_prefix.format(
                                    roast=roast_text
                                )
                            except KeyError:
                                msg = roast_text
                            await self._safe_send(event, msg)
                    except Exception as e:
                        logger.error(
                            f"[presupposition_validation] 异步跟风吐槽发送失败: {e}"
                        )
                else:
                    try:
                        await self._send_meme_roast(
                            event, meme_matched_msg, user_message, req.system_prompt
                        )
                    except Exception as e:
                        logger.error(
                            f"[presupposition_validation] 即时吐槽发送失败，"
                            f"继续核查流程: {e}"
                        )

        llm_false_indices = [i for i, t in enumerate(truths) if not t]

        should_correct = self._evaluate_relation(
            truths, relation, llm_false_indices, len(premises)
        )

        if should_correct:
            logger.info(
                f"[presupposition_validation] 发现虚假预设前提(LLM), "
                f"count={len(llm_false_indices)}, relation={relation}, "
                f"action={self.cfg.action_mode}, response_sent={response_sent}"
            )
            premise_text, correction_text = self._aggregate_corrections(
                llm_false_indices, premises, corrections
            )
            if not correction_text:
                correction_text = "经核查，该提问的前提存在事实性偏差。"
            if premise_text or correction_text:
                await self._handle_correction(
                    event, req, premise_text, correction_text,
                    response_sent, "前提纠错", logic_flaw=logic_flaw,
                )
            return

        if needs_search and self.cfg.fact_check_method == "web_search" and premises:
            uncertain_indices = [
                i for i in range(len(premises))
                if i not in llm_false_indices and premises[i].strip()
            ]
            search_false: dict[int, str] = {}
            for idx in uncertain_indices:
                search_result = await self._verify_with_web_search(
                    event, premises[idx]
                )
                if search_result is not None:
                    search_is_true, search_correction = search_result
                    if not search_is_true and search_correction:
                        search_false[idx] = search_correction

            if search_false:
                all_false_indices = llm_false_indices + list(search_false.keys())
                merged_corrections = list(corrections)
                for idx, corr in search_false.items():
                    while len(merged_corrections) <= idx:
                        merged_corrections.append("")
                    merged_corrections[idx] = corr

                merged_truths = list(truths)
                for idx in search_false:
                    while len(merged_truths) <= idx:
                        merged_truths.append(True)
                    merged_truths[idx] = False

                should_correct = self._evaluate_relation(
                    merged_truths, relation, all_false_indices, len(premises)
                )

                if should_correct:
                    logger.info(
                        f"[presupposition_validation] 搜索确认虚假前提, "
                        f"count={len(all_false_indices)}, action={self.cfg.action_mode}"
                    )
                    premise_text, correction_text = self._aggregate_corrections(
                        all_false_indices, premises, merged_corrections
                    )
                    if not correction_text:
                        correction_text = "经核查，该提问的前提存在事实性偏差。"
                    search_sent = sent_event is not None and sent_event.is_set()
                    if premise_text or correction_text:
                        await self._handle_correction(
                            event, req, premise_text, correction_text,
                            search_sent, "搜索纠错", logic_flaw=logic_flaw,
                        )
                    return

        if logic_flaw:
            logger.info(
                f"[presupposition_validation] 检测到逻辑谬误: {logic_flaw[:60]}"
            )
            logic_sent = sent_event is not None and sent_event.is_set()
            await self._handle_correction(
                event, req, "", "", logic_sent, "逻辑批驳",
                logic_flaw=logic_flaw,
            )

    # ==================================================================
    # 工具方法
    # ==================================================================

    @staticmethod
    def _quiet(text: Optional[str]) -> bool:
        if text is None:
            return True
        return not text.strip()

    async def _safe_send(self, event: AstrMessageEvent, text: Optional[str]):
        if self._quiet(text):
            return
        try:
            await event.send(text)
        except Exception as e:
            logger.error(f"[presupposition_validation] 发送消息失败: {e}")

    async def _safe_send_with_fallback(
        self, event: AstrMessageEvent, text: str
    ):
        if self._quiet(text):
            return
        try:
            await event.send(text)
        except Exception as e:
            logger.error(f"[presupposition_validation] 发送消息失败: {e}")
            if not self._quiet(self.cfg.error_fallback_text):
                try:
                    await event.send(self.cfg.error_fallback_text)
                except Exception:
                    pass

    # ==================================================================
    # 撤回 / 追击纠错（异步模式专用）
    # ==================================================================

    async def _send_withdraw_or_followup(
        self,
        event: AstrMessageEvent,
        reason: str,
        followup_text: str,
    ):
        withdrawn = await self._try_withdraw_message(event)
        if withdrawn:
            prefix = self.cfg.recall_success_prefix
            logger.info(f"[presupposition_validation] {reason}: 撤回成功")
        else:
            prefix = self.cfg.recall_fail_prefix
            logger.info(f"[presupposition_validation] {reason}: 撤回失败，发送追击更正")

        if self._quiet(prefix):
            full_msg = followup_text
        else:
            full_msg = f"{prefix}\n\n{followup_text}"

        await self._safe_send_with_fallback(event, full_msg)

    async def _try_withdraw_message(self, event: AstrMessageEvent) -> bool:
        if not self.cfg.enable_recall:
            return False

        group_id = event.get_group_id()
        if not group_id:
            return False

        session = self._get_session_id(event)
        tracked_id = None
        for key in list(self._sent_bot_msg_ids):
            if key.startswith(f"{session}:"):
                tracked_id = self._sent_bot_msg_ids.pop(key)
                break

        try:
            platform = event.get_platform_name()
            if platform != "aiocqhttp":
                logger.debug(
                    f"[presupposition_validation] 平台 {platform} 不支持撤回，跳过"
                )
                return False

            from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import (
                AiocqhttpMessageEvent,
            )
            if not isinstance(event, AiocqhttpMessageEvent):
                logger.debug("[presupposition_validation] 非 aiocqhttp 平台，跳过撤回")
                return False
            client = event.bot
            if not hasattr(client, "api"):
                return False

            target_id = tracked_id
            if not target_id:
                ret = await asyncio.wait_for(
                    client.api.call_action(
                        "get_group_msg_log",
                        group_id=group_id,
                        count=10,
                    ),
                    timeout=self._API_CALL_TIMEOUT,
                )
                if not ret:
                    return False
                messages = ret.get("data", ret) if isinstance(ret, dict) else ret
                if not isinstance(messages, list):
                    return False
                bot_id = event.get_self_id()
                for msg in messages:
                    sender = msg.get("sender", {})
                    if str(sender.get("user_id", "")) == str(bot_id):
                        target_id = msg.get("message_id")
                        if target_id:
                            break

            if not target_id:
                return False

            await asyncio.wait_for(
                client.api.call_action(
                    "delete_msg", message_id=int(target_id)
                ),
                timeout=self._API_CALL_TIMEOUT,
            )
            return True
        except asyncio.TimeoutError:
            logger.warning("[presupposition_validation] 撤回 API 调用超时")
            return False
        except Exception as e:
            logger.debug(
                f"[presupposition_validation] 撤回尝试失败（降级为追击更正）: {e}"
            )
            return False

    def _aggregate_corrections(
        self,
        false_indices: list[int],
        premises: list[str],
        corrections: list[str],
    ) -> tuple[str, str]:
        error_premises = [premises[i] for i in false_indices if i < len(premises)]
        error_corrections = [
            corrections[i]
            for i in false_indices
            if i < len(corrections) and corrections[i]
        ]
        if not error_premises:
            return "", ""
        if len(error_premises) == 1:
            return (
                error_premises[0],
                error_corrections[0] if error_corrections else "",
            )
        premise_text = "\n".join(
            f"{j + 1}. {p}" for j, p in enumerate(error_premises)
        )
        correction_text = "\n".join(
            f"{j + 1}. {c}" for j, c in enumerate(error_corrections)
        )
        return premise_text, correction_text

    async def _handle_correction(
        self,
        event: AstrMessageEvent,
        req: ProviderRequest,
        premise_text: str,
        correction_text: str,
        response_sent: bool,
        reason: str,
        logic_flaw: str = "",
    ):
        if logic_flaw and not premise_text:
            msg = self._build_logic_flaw_message(logic_flaw)
            if self.cfg.action_mode == "intercept":
                if response_sent:
                    await self._send_withdraw_or_followup(event, reason, msg)
                else:
                    event.stop_event()
                    await self._safe_send_with_fallback(event, msg)
            else:
                if response_sent:
                    await self._send_withdraw_or_followup(event, reason, msg)
                else:
                    req.system_prompt = (
                        f"[系统提示：用户论证存在逻辑漏洞]\n"
                        f"{logic_flaw}\n\n"
                        f"请在回复中指出该逻辑问题，然后基于正确逻辑回答。\n\n"
                        f"---\n\n{req.system_prompt}"
                    )
                    if self._search_driver is None and self.cfg.no_search_disclaimer:
                        req.system_prompt = (
                            f"{self.cfg.no_search_disclaimer}\n\n"
                            f"{req.system_prompt}"
                        )
            return

        if logic_flaw and correction_text:
            correction_text = f"{correction_text}\n\n⚠️ 逻辑漏洞：{logic_flaw}"
            if self._search_driver is None and self.cfg.no_search_disclaimer:
                correction_text = f"{correction_text}\n\n{self.cfg.no_search_disclaimer}"

        if self.cfg.action_mode == "intercept":
            if response_sent:
                followup = self._format_correction_followup(
                    premise_text, correction_text
                )
                await self._send_withdraw_or_followup(event, reason, followup)
            else:
                msg = self._build_intercept_message(premise_text, correction_text)
                event.stop_event()
                await self._safe_send_with_fallback(event, msg)
        else:
            if response_sent:
                followup = self._format_correction_followup(
                    premise_text, correction_text
                )
                await self._send_withdraw_or_followup(event, reason, followup)
            else:
                warning = self._build_warning_prefix(premise_text, correction_text)
                req.system_prompt = f"{warning}\n\n---\n\n{req.system_prompt}"

    def _build_logic_flaw_message(self, logic_flaw: str) -> str:
        parts = [
            "⚠️ 您的论证存在逻辑漏洞：\n\n"
            f"{logic_flaw}\n\n"
            "请修正您的论证逻辑后再提问。"
        ]
        if self._search_driver is None and self.cfg.no_search_disclaimer:
            parts.append(self.cfg.no_search_disclaimer)
        return "\n\n".join(parts)

    # ==================================================================
    # 追击文案构建
    # ==================================================================

    def _format_correction_followup(self, premise: str, correction: str) -> str:
        try:
            return self.cfg.correction_followup_text.format(
                premise=premise, correction=correction
            )
        except KeyError:
            return f"「{premise}」— 事实并非如此\n✅ 修正：{correction}"

    def _build_meme_followup(
        self, original_msg: Optional[str], copy_msg: str
    ) -> str:
        if not original_msg:
            return f"检测到跟风造句，拒绝回答。「{copy_msg}」"
        try:
            return self.cfg.meme_followup_text.format(
                original=original_msg, copy=copy_msg
            )
        except KeyError:
            return (
                f"检测到跟风造句！\n"
                f"原问题：「{original_msg}」\n"
                f"跟风提问：「{copy_msg}」\n"
                f"拒绝回答跟风恶搞。"
            )

    async def _resolve_meme_roast_text(
        self,
        event: AstrMessageEvent,
        original_msg: Optional[str],
        copy_msg: str,
        bot_system_prompt: str = "",
    ) -> Optional[str]:
        if self.cfg.meme_response_mode == "dynamic_llm":
            roast = await self._generate_dynamic_roast(
                event, original_msg or "", copy_msg, bot_system_prompt
            )
            return roast if roast else self.cfg.meme_fixed_reply_text
        return self.cfg.meme_fixed_reply_text

    # ==================================================================
    # 单次 LLM 全能预审
    # ==================================================================

    async def _unified_llm_check(
        self, event: AstrMessageEvent, user_message: str
    ) -> Optional[dict]:
        provider = self.context.get_using_provider(umo=event.unified_msg_origin)
        if provider is None:
            logger.warning("[presupposition_validation] 未找到可用的 LLM 提供商")
            return None

        system_prompt = self.cfg.unified_check_prompt
        if not system_prompt:
            return None

        if self.cfg.enable_argumentative_mode and self.cfg.argumentative_prompt_appendix:
            system_prompt = system_prompt + self.cfg.argumentative_prompt_appendix

        try:
            llm_resp = await asyncio.wait_for(
                provider.text_chat(
                    prompt=f"用户发了一条消息：{user_message}",
                    system_prompt=system_prompt,
                ),
                timeout=self.cfg.api_request_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("[presupposition_validation] 全能预审 LLM 调用超时")
            return None
        except Exception as e:
            logger.error(f"[presupposition_validation] 全能预审 LLM 调用出错: {e}")
            return None

        if llm_resp is None:
            return None

        completion = llm_resp.completion_text
        if not completion:
            return None

        return self._parse_unified_response(completion)

    def _parse_unified_response(self, text: str) -> Optional[dict]:
        if not text:
            return None
        try:
            raw = self._extract_json(text)
            if raw is None:
                return None
            data = json.loads(raw)

            is_factual = _parse_bool(data.get("is_factual_question"), False)

            premises = data.get("premises")
            if isinstance(premises, list) and len(premises) > 0:
                raw_truths = data.get("premise_truths", [])
                raw_corrections = data.get("corrections", [])
                truths = [
                    _parse_bool(raw_truths[i], True) if i < len(raw_truths) else True
                    for i in range(len(premises))
                ]
                corrections = [
                    str(raw_corrections[i]).strip() if i < len(raw_corrections) else ""
                    for i in range(len(premises))
                ]
                relation = str(data.get("premise_relation", "and")).strip().lower()
                if relation not in ("and", "or", "xor", "implication", "biconditional"):
                    relation = "and"
                normalized = [self._normalize_premise(str(p)) for p in premises]
                logic_flaw = str(data.get("logic_flaw", "")).strip()
                return {
                    "is_factual": is_factual,
                    "premises": normalized,
                    "truths": truths,
                    "relation": relation,
                    "corrections": corrections,
                    "logic_flaw": logic_flaw,
                    "needs_search": _parse_bool(data.get("needs_search"), False),
                }

            premise = self._normalize_premise(str(data.get("extracted_premise", "")))
            has_false = _parse_bool(data.get("has_false_premise"), False)
            correction = str(data.get("correction_info", "")).strip()
            if not premise:
                return {
                    "is_factual": is_factual,
                    "premises": [],
                    "truths": [],
                    "relation": "and",
                    "corrections": [],
                    "logic_flaw": "",
                    "needs_search": _parse_bool(data.get("needs_search"), False),
                }
            return {
                "is_factual": is_factual,
                "premises": [premise],
                "truths": [False] if has_false else [True],
                "relation": "and",
                "corrections": [correction] if has_false else [""],
                "logic_flaw": "",
                "needs_search": _parse_bool(data.get("needs_search"), False),
            }
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"[presupposition_validation] 解析全能预审 JSON 失败: {e}")
            return None

    @staticmethod
    def _normalize_premise(text: str) -> str:
        text = text.strip()
        if not text:
            return ""
        text = text.lower()
        text = text.translate(
            str.maketrans(
                "", "",
                _string.punctuation
                + "，。！？、；：""''（）【】《》…—·「」『』"
                + "\u201c\u201d\u2018\u2019"
            )
        )
        text = " ".join(text.split())
        return text

    # ==================================================================
    # 联网搜索验证（仅当 LLM 标记 needs_search 时触发）
    # ==================================================================

    async def _verify_with_web_search(
        self, event: AstrMessageEvent, premise: str
    ) -> Optional[tuple]:
        if self._search_driver is None:
            return None
        provider = self.context.get_using_provider(umo=event.unified_msg_origin)
        if provider is None:
            return None
        try:
            return await asyncio.wait_for(
                self._verify_single_with_search(provider, premise),
                timeout=self.cfg.api_request_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("[presupposition_validation] 联网搜索验证超时")
            return None
        except Exception as e:
            logger.error(f"[presupposition_validation] 联网搜索验证出错: {e}")
            return None

    async def _verify_single_with_search(
        self, provider, presupposition: str
    ) -> Optional[tuple]:
        search_summary = await self._execute_search(presupposition)
        if not search_summary:
            return None

        verify_prompt = (
            f"需要验证的命题：{presupposition}\n\n"
            f"搜索结果：\n{search_summary}\n\n"
            f"请根据搜索结果判断该命题是否为真。"
        )

        verify_system_prompt = self.cfg.search_verify_prompt
        if not verify_system_prompt:
            return None

        verify_resp = await provider.text_chat(
            prompt=verify_prompt,
            system_prompt=verify_system_prompt,
        )
        if verify_resp is None:
            return None

        return self._parse_verify_response(verify_resp.completion_text)

    async def _execute_search(self, query: str) -> Optional[str]:
        if self._search_driver == "framework":
            return await self._search_via_framework(query)
        if self._search_driver == "duckduckgo":
            return await self._search_via_ddg(query)
        return None

    async def _search_via_framework(self, query: str) -> Optional[str]:
        try:
            from astrbot.core.agent.tool import ToolSet

            tool_mgr = self.context.get_llm_tool_manager()
            search_tool = None
            for kw in self._SEARCH_TOOL_KEYWORDS:
                candidate = tool_mgr.get_func(kw)
                if candidate:
                    search_tool = candidate
                    break
            if search_tool is None:
                return None

            tool_set = ToolSet()
            tool_set.add_tool(search_tool)

            prov_id = await self.context.get_current_chat_provider_id(
                self._get_session_id(self.context)
            )
        except Exception as e:
            logger.debug(
                f"[presupposition_validation] 框架搜索工具探测失败: {e}"
            )
            return None

        try:
            llm_resp = await self.context.tool_loop_agent(
                event=None,
                chat_provider_id=prov_id,
                prompt=f"请搜索以下内容并返回摘要：{query}",
                tools=tool_set,
                max_steps=3,
                tool_call_timeout=10,
            )
            if llm_resp and hasattr(llm_resp, "completion_text"):
                text = llm_resp.completion_text.strip()
                if text:
                    return text
        except Exception as e:
            logger.debug(f"[presupposition_validation] 框架搜索调用失败: {e}")

        return None

    async def _search_via_ddg(self, query: str) -> Optional[str]:
        try:
            from duckduckgo_search import AsyncDDGS

            async with AsyncDDGS() as ddgs:
                results = await ddgs.text(query, max_results=3)
        except ImportError:
            logger.error(
                "[presupposition_validation] duckduckgo-search 运行时导入失败，"
                "请执行: pip install duckduckgo-search"
            )
            return None
        except Exception as e:
            logger.error(f"[presupposition_validation] DuckDuckGo 搜索出错: {e}")
            return None

        if not results:
            return None

        return "\n".join(
            f"- {r.get('title', '')}: {r.get('body', '')}" for r in results
        )

    def _parse_verify_response(self, text: str) -> Optional[tuple]:
        if not text:
            return None
        try:
            raw = self._extract_json(text)
            if raw is None:
                return None
            data = json.loads(raw)
            return (_parse_bool(data.get("is_true"), False), data.get("correction", ""))
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"[presupposition_validation] 解析搜索验证结果失败: {e}")
            return None

    # ==================================================================
    # 群聊跟风造句检测
    # ==================================================================

    async def _check_meme_pattern(
        self,
        group_id: str,
        message: str,
        premises: Optional[list[str]] = None,
    ) -> Optional[str]:
        threshold = max(0.0, min(1.0, self.cfg.similarity_threshold))

        if len(self._group_msg_cache) >= self._MAX_GROUP_CACHE:
            self._group_msg_cache.popitem(last=False)

        queue = self._group_msg_cache.get(group_id)
        if queue is None:
            queue = deque(maxlen=self.cfg.history_window_size)
            self._group_msg_cache[group_id] = queue

        self._group_last_active[group_id] = _time.monotonic()
        self._group_msg_cache.move_to_end(group_id)

        matched = None
        for cached_entry in queue:
            if not cached_entry:
                continue
            cached_msg = cached_entry[0]
            cached_prs = cached_entry[1] if len(cached_entry) > 1 else []

            ratio = SequenceMatcher(None, cached_msg, message).ratio()
            if ratio >= threshold and ratio < 1.0:
                matched = cached_msg
                break

            if premises and cached_prs:
                for cp in cached_prs:
                    if not cp:
                        continue
                    for np_item in premises:
                        if not np_item:
                            continue
                        pratio = SequenceMatcher(None, cp, np_item).ratio()
                        if pratio >= threshold and pratio < 1.0:
                            matched = cached_msg
                            break
                    if matched:
                        break
                if matched:
                    break

        queue.append((message, premises or []))
        return matched

    async def _llm_verify_meme(
        self,
        event: AstrMessageEvent,
        group_id: str,
        new_message: str,
        premises: Optional[list[str]] = None,
    ) -> Optional[bool]:
        provider = self.context.get_using_provider(umo=event.unified_msg_origin)
        if provider is None:
            return None

        system_prompt = self.cfg.meme_llm_check_prompt
        if not system_prompt:
            return None

        queue = self._group_msg_cache.get(group_id)
        if not queue:
            return None

        entries = list(queue)

        history_lines = []
        for entry in entries:
            if not entry:
                continue
            msg = entry[0]
            prs = entry[1] if len(entry) > 1 else []
            if prs:
                history_lines.append(f"- {msg} (前提: {'; '.join(prs)})")
            else:
                history_lines.append(f"- {msg}")

        history_text = "\n".join(history_lines)

        premise_info = ""
        if premises:
            premise_info = f"\n当前消息提取的前提：{'; '.join(premises)}"

        prompt = (
            f"以下是群里的历史提问：\n{history_text}\n\n"
            f"现在有用户提问：{new_message}{premise_info}\n"
            f"请判断这个新问题仅仅是无意义的改词跟风造句，"
            f"还是一个有其实际求知意义的独立问题？"
            f"请仅回复 True 或 False。"
        )

        try:
            llm_resp = await asyncio.wait_for(
                provider.text_chat(
                    prompt=prompt,
                    system_prompt=system_prompt,
                ),
                timeout=self.cfg.api_request_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("[presupposition_validation] LLM 跟风二次确认超时")
            return None
        except Exception as e:
            logger.error(f"[presupposition_validation] LLM 跟风二次确认出错: {e}")
            return None

        if llm_resp is None:
            return None

        text = llm_resp.completion_text.strip()
        is_meme = text.upper().strip() == "TRUE"
        logger.debug(
            f"[presupposition_validation] LLM 跟风判定: "
            f"raw={text[:30]}, result={is_meme}"
        )
        return is_meme

    async def _send_meme_roast(
        self,
        event: AstrMessageEvent,
        original_msg: Optional[str],
        copy_msg: str,
        bot_system_prompt: str = "",
    ):
        roast_text = await self._resolve_meme_roast_text(
            event, original_msg, copy_msg, bot_system_prompt
        )
        await self._safe_send(event, roast_text or "")

    async def _generate_dynamic_roast(
        self, event: AstrMessageEvent, original_msg: str, copy_msg: str,
        bot_system_prompt: str = "",
    ) -> Optional[str]:
        provider = self.context.get_using_provider(umo=event.unified_msg_origin)
        if provider is None:
            return None

        prompt_template = self.cfg.meme_dynamic_reply_prompt
        if not prompt_template:
            return None

        bot_persona = ""
        if bot_system_prompt:
            bot_persona = (
                "以下是你的原始人设提示词，请据此保持你的性格和说话风格：\n"
                f"---\n{bot_system_prompt}\n---\n\n"
            )

        try:
            system_prompt = prompt_template.format(
                original_message=original_msg,
                copy_message=copy_msg,
                bot_persona=bot_persona,
            )
        except KeyError as e:
            logger.warning(f"[presupposition_validation] meme_dynamic_reply_prompt 占位符缺失: {e}")
            return None

        try:
            llm_resp = await asyncio.wait_for(
                provider.text_chat(
                    prompt=f"群友跟风提问：{copy_msg}（原问题：{original_msg}），请吐槽。",
                    system_prompt=system_prompt,
                ),
                timeout=self.cfg.api_request_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("[presupposition_validation] 动态吐槽生成超时")
            return None
        except Exception as e:
            logger.error(f"[presupposition_validation] 动态吐槽生成出错: {e}")
            return None

        if llm_resp is None:
            return None

        roast_text = llm_resp.completion_text.strip()
        if not roast_text:
            return None

        logger.debug(f"[presupposition_validation] 动态吐槽生成成功: {roast_text[:60]}...")
        return roast_text

    # ==================================================================
    # JSON 解析工具
    # ==================================================================

    @staticmethod
    def _extract_json(text: str) -> Optional[str]:
        text = text.strip()

        if "```" in text:
            match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
            if match:
                block = match.group(1).strip()
                if block.startswith("{"):
                    return block

        depth = 0
        start = -1
        for i, ch in enumerate(text):
            if ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and start >= 0:
                    return text[start:i + 1]
        return None

    # ==================================================================
    # 消息构建
    # ==================================================================

    def _build_intercept_message(self, premise: str, correction: str) -> str:
        try:
            return self.cfg.intercept_message_text.format(
                premise=premise, correction=correction
            )
        except KeyError:
            return (
                "⚠️ 您的提问中包含不成立的预设前提：\n\n"
                f"❌ {premise}\n\n"
                f"✅ 修正：\n{correction}\n\n"
                "请您基于正确的事实重新提问。"
            )

    def _build_warning_prefix(self, premise: str, correction: str) -> str:
        try:
            return self.cfg.warning_prefix_text.format(
                premise=premise, correction=correction
            )
        except KeyError:
            return (
                "[系统提示：用户提问中包含不成立的预设前提，请先指出错误再回答]\n\n"
                f"❌ {premise}\n\n"
                f"✅ 修正：\n{correction}\n\n"
                "请在回复中先指出上述前提错误，然后基于修正后的事实回答用户的原始问题。"
            )

    # ==================================================================
    # 生命周期
    # ==================================================================

    async def terminate(self):
        tasks = list(self._pending_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._pending_tasks.clear()
        self._session_sent_events.clear()
        self._sent_bot_msg_ids.clear()
        self._group_msg_cache.clear()
        self._group_last_active.clear()
        from .config import invalidate_schema_cache
        invalidate_schema_cache()
