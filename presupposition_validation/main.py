"""presupposition_validation - AstrBot 预设前提核查插件

在机器人正式生成回复之前，拦截用户输入，通过单次 LLM 调用完成
意图判定、前提提取与事实核查，避免基于虚假前提生成误导性回答。

使用方式：
1. 将本插件目录放置到 AstrBot 的 data/plugins/ 目录下
2. 在 AstrBot WebUI 的插件管理中启用本插件
3. 在插件配置中选择核查方式（llm / web_search）和处理模式（intercept / warn_and_answer）
"""

import asyncio
import json
import re
import string as _string
import time as _time
from collections import deque, OrderedDict
from difflib import SequenceMatcher
from typing import Optional

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, register

from .config import PluginConfig


@register(
    "presupposition_validation",
    "presupposition_validation_dev",
    "核查用户提问中隐藏的预设前提，防止基于虚假前提的回答",
    "1.0.1",
)
class PresuppositionValidation(Star):

    _MAX_GROUP_CACHE = 200
    _GROUP_GC_INTERVAL = 300

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.cfg = PluginConfig(**{k: v for k, v in config.items()})
        self._group_msg_cache: OrderedDict[str, deque] = OrderedDict()
        self._group_last_active: dict[str, float] = {}
        self._pending_tasks: set[asyncio.Task] = set()
        self._session_sent_events: dict[str, asyncio.Event] = {}
        self._pipeline_lock = asyncio.Lock()

    # ==================================================================
    # 核心 Hook
    # ==================================================================

    @filter.on_llm_request()
    async def on_llm_request(
        self, event: AstrMessageEvent, req: ProviderRequest
    ):
        if not self.cfg.enabled:
            return

        user_message = ""
        try:
            user_message = (event.message_str or "").strip()
        except Exception:
            return

        if not user_message or len(user_message) < 3:
            return

        if self.cfg.enable_async_mode:
            umo = event.unified_msg_origin
            sent_evt = asyncio.Event()
            self._session_sent_events[umo] = sent_evt

            task = asyncio.create_task(
                self._cleanup_pipeline(event, req, user_message, sent_evt),
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
        umo = event.unified_msg_origin
        sent_evt = self._session_sent_events.get(umo)
        if sent_evt is not None:
            sent_evt.set()

    # ==================================================================
    # 核查管线（串行/异步共用）
    # ==================================================================

    async def _cleanup_pipeline(
        self,
        event: AstrMessageEvent,
        req: ProviderRequest,
        user_message: str,
        sent_evt: asyncio.Event,
    ):
        try:
            await self._run_pipeline(event, req, user_message, sent_event=sent_evt)
        except Exception as e:
            logger.error(f"[presupposition_validation] 核查管线异常: {e}")
        finally:
            umo = event.unified_msg_origin
            self._session_sent_events.pop(umo, None)

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

        # ==================================================================
        # 步骤一：单次 LLM 全能预审
        # ==================================================================
        result = await self._unified_llm_check(event, user_message)
        if result is None:
            logger.debug("[presupposition_validation] 预审失败或解析异常，放行")
            return

        is_factual, premise, has_false, correction, needs_search = result

        # ==================================================================
        # 步骤二：非事实性问题 → 直接放行
        # ==================================================================
        if not is_factual:
            logger.debug("[presupposition_validation] 非事实性问题，放行")
            return

        logger.debug(
            f"[presupposition_validation] 预审结果: premise={premise[:40] if premise else '无'}, "
            f"has_false={has_false}, needs_search={needs_search}"
        )

        # ==================================================================
        # 步骤三：跟风造句检测（基于用户原始消息）
        # ==================================================================
        meme_hit = False
        meme_matched_msg: Optional[str] = None

        if self.cfg.enable_meme_detect and event.is_private_chat() is False:
            group_id = event.get_group_id()
            if group_id:
                meme_matched_msg = self._check_meme_pattern(group_id, user_message)

                if meme_matched_msg is not None:
                    confirmed = True
                    if self.cfg.meme_detect_method == "llm":
                        confirmed = await self._llm_verify_meme(
                            event, group_id, user_message
                        )
                        if confirmed is None:
                            confirmed = True

                    if confirmed:
                        meme_hit = True
                        logger.info(
                            f"[presupposition_validation] 跟风造句命中, "
                            f"action={self.cfg.meme_action_mode}"
                        )

        async with self._pipeline_lock:
            response_sent = sent_event is not None and sent_event.is_set()

            # ==================================================================
            # 步骤四：综合判定 —— 跟风 + 前提错误
            # ==================================================================
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

            # ==================================================================
            # 步骤四续：处理前提错误（LLM 常识已判定为假）
            # ==================================================================
            if has_false and premise and correction:
                logger.info(
                    f"[presupposition_validation] 发现虚假预设前提, "
                    f"action={self.cfg.action_mode}, response_sent={response_sent}"
                )
                if self.cfg.action_mode == "intercept":
                    if response_sent:
                        followup = self._format_correction_followup(premise, correction)
                        await self._send_withdraw_or_followup(event, "前提纠错", followup)
                    else:
                        msg = self._build_intercept_message(premise, correction)
                        event.stop_event()
                        await self._safe_send_with_fallback(event, msg)
                else:
                    if response_sent:
                        followup = self._format_correction_followup(premise, correction)
                        await self._send_withdraw_or_followup(event, "前提纠错", followup)
                    else:
                        warning = self._build_warning_prefix(premise, correction)
                        req.system_prompt = f"{warning}\n\n---\n\n{req.system_prompt}"
                return

            # ==================================================================
            # 步骤五：联网搜索兜底（LLM 标记 needs_search 且模式允许）
            # ==================================================================
            if needs_search and self.cfg.fact_check_method == "web_search" and premise:
                logger.info(
                    f"[presupposition_validation] LLM 无法常识判定，触发联网搜索验证"
                )
                if not premise.strip():
                    return
                search_result = await self._verify_with_web_search(event, premise)
                if search_result is not None:
                    search_has_false, search_correction = search_result
                    if search_has_false and search_correction:
                        logger.info(
                            f"[presupposition_validation] 搜索确认虚假前提, "
                            f"action={self.cfg.action_mode}"
                        )
                        search_sent = sent_event is not None and sent_event.is_set()
                        if self.cfg.action_mode == "intercept":
                            if search_sent:
                                followup = self._format_correction_followup(premise, search_correction)
                                await self._send_withdraw_or_followup(event, "搜索纠错", followup)
                            else:
                                msg = self._build_intercept_message(premise, search_correction)
                                event.stop_event()
                                await self._safe_send_with_fallback(event, msg)
                        else:
                            if search_sent:
                                followup = self._format_correction_followup(premise, search_correction)
                                await self._send_withdraw_or_followup(event, "搜索纠错", followup)
                            else:
                                warning = self._build_warning_prefix(premise, search_correction)
                                req.system_prompt = f"{warning}\n\n---\n\n{req.system_prompt}"
                        return

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

            ret = await client.api.call_action(
                "get_group_msg_log",
                group_id=event.get_group_id(),
                count=10,
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
                    msg_id = msg.get("message_id")
                    if msg_id:
                        await client.api.call_action(
                            "delete_msg", message_id=int(msg_id)
                        )
                        return True
            return False
        except Exception as e:
            logger.debug(
                f"[presupposition_validation] 撤回尝试失败（降级为追击更正）: {e}"
            )
            return False

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
    ) -> Optional[tuple]:
        provider = self.context.get_using_provider(umo=event.unified_msg_origin)
        if provider is None:
            logger.warning("[presupposition_validation] 未找到可用的 LLM 提供商")
            return None

        system_prompt = self.cfg.unified_check_prompt
        if not system_prompt:
            return None

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

    def _parse_unified_response(self, text: str) -> Optional[tuple]:
        if not text:
            return None
        try:
            raw = self._extract_json(text)
            if raw is None:
                return None
            data = json.loads(raw)
            premise = self._normalize_premise(
                str(data.get("extracted_premise", ""))
            )
            return (
                bool(data.get("is_factual_question", False)),
                premise,
                bool(data.get("has_false_premise", False)),
                str(data.get("correction_info", "")).strip(),
                bool(data.get("needs_search", False)),
            )
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
        try:
            from duckduckgo_search import DDGS

            search_results = await asyncio.to_thread(
                lambda: list(
                    DDGS().text(presupposition, max_results=3)
                )
            )
        except ImportError:
            logger.error(
                "[presupposition_validation] web_search 模式需要安装 "
                "duckduckgo-search 库，请执行: pip install duckduckgo-search"
            )
            return None
        except Exception as e:
            logger.error(f"[presupposition_validation] 搜索出错: {e}")
            return None

        if not search_results:
            return None

        search_summary = "\n".join(
            f"- {r.get('title', '')}: {r.get('body', '')}" for r in search_results
        )

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

    def _parse_verify_response(self, text: str) -> Optional[tuple]:
        if not text:
            return None
        try:
            raw = self._extract_json(text)
            if raw is None:
                return None
            data = json.loads(raw)
            return (data.get("is_true", True), data.get("correction", ""))
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"[presupposition_validation] 解析搜索验证结果失败: {e}")
            return None

    # ==================================================================
    # 群聊跟风造句检测
    # ==================================================================

    def _check_meme_pattern(self, group_id: str, message: str) -> Optional[str]:
        if len(self._group_msg_cache) >= self._MAX_GROUP_CACHE:
            now = _time.monotonic()
            stale = [
                gid for gid, ts in self._group_last_active.items()
                if now - ts > self._GROUP_GC_INTERVAL
            ]
            for gid in stale:
                self._group_msg_cache.pop(gid, None)
                self._group_last_active.pop(gid, None)

        self._group_last_active[group_id] = _time.monotonic()
        self._group_msg_cache.move_to_end(group_id)

        queue = self._group_msg_cache.setdefault(
            group_id, deque(maxlen=self.cfg.history_window_size)
        )
        threshold = max(0.0, min(1.0, self.cfg.similarity_threshold))
        matched = None

        for cached_msg in queue:
            if not cached_msg:
                continue
            ratio = SequenceMatcher(None, cached_msg, message).ratio()
            if ratio >= threshold and ratio < 1.0:
                matched = cached_msg
                break

        queue.append(message)
        return matched

    async def _llm_verify_meme(
        self, event: AstrMessageEvent, group_id: str, new_message: str
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

        history_lines = [f"{i + 1}. {msg}" for i, msg in enumerate(queue) if msg]
        history_text = "\n".join(history_lines)

        prompt = (
            f"以下是群里的历史提问：\n{history_text}\n\n"
            f"现在有用户提问：{new_message}\n"
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

        text = llm_resp.completion_text.strip().upper()
        is_meme = "TRUE" in text
        logger.debug(
            f"[presupposition_validation] LLM 跟风判定: "
            f"raw={llm_resp.completion_text.strip()[:30]}, result={is_meme}"
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

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return match.group(0).strip()
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
                f"❌ 「{premise}」— 事实并非如此\n"
                f"   ✅ 修正：{correction}\n\n"
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
                f"- 「{premise}」是错误的，正确事实是：{correction}\n\n"
                "请在回复中先指出上述前提错误，然后基于修正后的事实回答用户的原始问题。"
            )

    # ==================================================================
    # 生命周期
    # ==================================================================

    async def terminate(self):
        for task in self._pending_tasks:
            task.cancel()
        self._pending_tasks.clear()
        self._session_sent_events.clear()
        self._group_msg_cache.clear()
