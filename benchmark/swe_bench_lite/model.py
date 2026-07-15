from __future__ import annotations

from minisweagent.models.litellm_model import LitellmModel


class SparseVLLMLitellmModel(LitellmModel):
    """Remove LiteLLM-only response metadata before replaying chat history."""

    def _prepare_messages_for_api(self, messages: list[dict]) -> list[dict]:
        cleaned = [
            {key: value for key, value in message.items() if key != "provider_specific_fields"}
            for message in messages
        ]
        return super()._prepare_messages_for_api(cleaned)
