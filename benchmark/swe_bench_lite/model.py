from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from minisweagent.exceptions import FormatError
from minisweagent.models.litellm_model import LitellmModel


class SparseVLLMLitellmModel(LitellmModel):
    """Replay clean chat history and opt into per-instance Sparse-vLLM chains."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._chain_cache_enabled = os.getenv(
            "SPARSEVLLM_CHAIN_CACHE", ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._chain_id: str | None = None
        self._last_request_messages: list[dict[str, Any]] | None = None
        self._last_response_message: dict[str, Any] | None = None
        self._last_response_append_safe = False
        self._recovery_chain_id: str | None = None
        self._pending_chain_state: tuple[
            str,
            list[dict[str, Any]],
            dict[str, Any],
            str | None,
            str,
            str,
        ] | None = None
        self._force_new_chain_reason: str | None = None

    @classmethod
    def _plain_value(cls, value: Any) -> Any:
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            value = model_dump(mode="json")
        if isinstance(value, dict):
            return {
                str(key): cls._plain_value(item)
                for key, item in value.items()
                if key not in {"extra", "provider_specific_fields"}
                and item is not None
            }
        if isinstance(value, list):
            return [cls._plain_value(item) for item in value]
        return value

    @classmethod
    def _chain_message(cls, message: dict[str, Any]) -> dict[str, Any]:
        plain = cls._plain_value(message)
        return {
            key: plain[key]
            for key in (
                "role",
                "content",
                "reasoning_content",
                "tool_calls",
                "tool_call_id",
                "name",
            )
            if key in plain
        }

    @staticmethod
    def _value_digest(value: Any) -> str:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]

    def _continuation_error(
        self,
        messages: list[dict[str, Any]],
    ) -> str | None:
        if self._last_request_messages is None:
            return "missing previous request messages"
        if self._last_response_message is None:
            return "missing previous response message"
        current = [self._chain_message(message) for message in messages]
        previous = self._last_request_messages
        if len(current) <= len(previous):
            return (
                "message history did not grow: "
                f"current={len(current)} previous={len(previous)}"
            )
        for index, (expected, actual) in enumerate(
            zip(previous, current, strict=False)
        ):
            if expected == actual:
                continue
            differing_fields = sorted(
                key
                for key in set(expected) | set(actual)
                if expected.get(key) != actual.get(key)
            )
            return (
                "previous request prefix changed: "
                f"index={index} "
                f"expected_role={expected.get('role')!r} "
                f"actual_role={actual.get('role')!r} "
                f"differing_fields={differing_fields!r} "
                f"expected_digest={self._value_digest(expected)} "
                f"actual_digest={self._value_digest(actual)}"
            )
        appended_response = current[len(previous)]
        if appended_response != self._last_response_message:
            differing_fields = sorted(
                key
                for key in set(appended_response)
                | set(self._last_response_message)
                if appended_response.get(key)
                != self._last_response_message.get(key)
            )
            return (
                "previous assistant response changed: "
                f"index={len(previous)} "
                f"differing_fields={differing_fields!r} "
                "expected_digest="
                f"{self._value_digest(self._last_response_message)} "
                f"actual_digest={self._value_digest(appended_response)}"
            )
        return None

    def _query(self, messages: list[dict[str, str]], **kwargs):
        if not self._chain_cache_enabled:
            return super()._query(messages, **kwargs)

        self._pending_chain_state = None
        request_chain_id = self._recovery_chain_id or self._chain_id
        recovering_chain = self._force_new_chain_reason is not None
        if request_chain_id is not None and not recovering_chain:
            continuation_error = self._continuation_error(messages)
            if continuation_error is not None:
                raise RuntimeError(
                    "Sparse-vLLM chain transcript is not append-only: "
                    f"{continuation_error}."
                )
        chain_append_start = (
            len(self._last_request_messages) + 1
            if request_chain_id is not None
            and not recovering_chain
            and self._last_response_append_safe
            and self._last_request_messages is not None
            else None
        )
        configured_extra_body = self.config.model_kwargs.get("extra_body") or {}
        request_extra_body = kwargs.pop("extra_body", None) or {}
        extra_body = {
            **configured_extra_body,
            **request_extra_body,
            "chain_id": request_chain_id,
            "preserve_thinking": True,
        }
        if chain_append_start is not None:
            extra_body["chain_append_start"] = chain_append_start
        response = super()._query(
            messages,
            extra_body=extra_body,
            **kwargs,
        )
        chain_id = getattr(response, "chain_id", None)
        if chain_id is None:
            model_extra = getattr(response, "model_extra", None)
            if isinstance(model_extra, dict):
                chain_id = model_extra.get("chain_id")
        normalized_chain_id = str(chain_id or "").strip()
        if not normalized_chain_id:
            raise RuntimeError(
                "Sparse-vLLM chain-cache request completed without a chain_id."
            )
        pending_request_messages = [
            self._chain_message(message) for message in messages
        ]
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise RuntimeError(
                "Sparse-vLLM chain-cache response contained no choices."
            )
        response_message = getattr(choices[0], "message", None)
        if response_message is None:
            raise RuntimeError(
                "Sparse-vLLM chain-cache response contained no assistant message."
            )
        pending_response_message = self._chain_message(response_message)
        finish_reason = str(
            getattr(choices[0], "finish_reason", "") or ""
        ).strip()
        if not finish_reason:
            raise RuntimeError(
                "Sparse-vLLM chain-cache response omitted finish_reason."
            )
        chain_status = getattr(response, "chain_status", None)
        if chain_status is None:
            model_extra = getattr(response, "model_extra", None)
            if isinstance(model_extra, dict):
                chain_status = model_extra.get("chain_status")
        normalized_chain_status = str(chain_status or "").strip()
        if not normalized_chain_status:
            raise RuntimeError(
                "Sparse-vLLM chain-cache response omitted chain_status."
            )
        self._pending_chain_state = (
            normalized_chain_id,
            pending_request_messages,
            pending_response_message,
            self._force_new_chain_reason,
            finish_reason,
            normalized_chain_status,
        )
        return response

    def query(self, messages: list[dict[str, Any]], **kwargs):
        if not self._chain_cache_enabled:
            return super().query(messages, **kwargs)

        try:
            message = super().query(messages, **kwargs)
        except Exception as exc:
            if self._pending_chain_state is not None:
                self._recovery_chain_id = self._pending_chain_state[0]
                self._force_new_chain_reason = (
                    "format_error"
                    if isinstance(exc, FormatError)
                    else type(exc).__name__
                )
            self._pending_chain_state = None
            raise

        pending = self._pending_chain_state
        if pending is None:
            raise RuntimeError(
                "Sparse-vLLM chain-cache query completed without pending "
                "chain state."
            )
        (
            self._chain_id,
            self._last_request_messages,
            self._last_response_message,
            reset_reason,
            finish_reason,
            chain_status,
        ) = pending
        chain_invalidated = chain_status == "invalidated"
        self._last_response_append_safe = (
            not chain_invalidated
            and finish_reason in {"stop", "tool_calls"}
        )
        if chain_invalidated:
            self._chain_id = None
        self._pending_chain_state = None
        self._recovery_chain_id = None
        self._force_new_chain_reason = None
        if reset_reason is not None and isinstance(message, dict):
            extra = message.setdefault("extra", {})
            extra["chain_reset_reason"] = reset_reason
        return message

    def _prepare_messages_for_api(self, messages: list[dict]) -> list[dict]:
        cleaned = [
            {key: value for key, value in message.items() if key != "provider_specific_fields"}
            for message in messages
        ]
        return super()._prepare_messages_for_api(cleaned)
