"""Canned success payloads for each provider route the matrix exercises."""

OPENAI_CHAT = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 0,
    "model": "m",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "ok"},
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}

OPENAI_TEXT = {
    "id": "cmpl-1",
    "object": "text_completion",
    "created": 0,
    "model": "m",
    "choices": [{"index": 0, "finish_reason": "stop", "text": "ok", "logprobs": None}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}

OPENAI_EMBEDDING = {
    "object": "list",
    "model": "m",
    "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
    "usage": {"prompt_tokens": 1, "total_tokens": 1},
}

OPENAI_RESPONSES = {
    "id": "resp_1",
    "object": "response",
    "created_at": 0,
    "status": "completed",
    "model": "m",
    "output": [
        {
            "type": "message",
            "id": "msg_1",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "ok", "annotations": []}],
        }
    ],
    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
}

VERTEX_EMBEDDING = {
    "predictions": [
        {
            "embeddings": {
                "values": [0.1, 0.2, 0.3],
                "statistics": {"token_count": 1, "truncated": False},
            }
        }
    ],
    "metadata": {"billableCharacterCount": 1},
}

ANTHROPIC_MESSAGE = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "m",
    "content": [{"type": "text", "text": "ok"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 1, "output_tokens": 1},
}

BEDROCK_CONVERSE = {
    "output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
    "stopReason": "end_turn",
    "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
    "metrics": {"latencyMs": 1},
}
