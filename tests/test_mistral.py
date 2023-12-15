from pytest_httpx import IteratorStream
import llm


def test_stream(httpx_mock):
    httpx_mock.add_response(
        url="https://api.mistral.ai/v1/chat/completions",
        method="POST",
        stream=IteratorStream(
            [
                b'data: {"id": "cmpl-4243ee7858634455a2153d6430719956", "model": "mistral-tiny", "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]}\n\n',
                b'data: {"id": "cmpl-4243ee7858634455a2153d6430719956", "object": "chat.completion.chunk", "created": 1702612156, "model": "mistral-tiny", "choices": [{"index": 0, "delta": {"role": null, "content": "I am an AI"}, "finish_reason": null}]}\n\n',
                b'data: {"id": "cmpl-4243ee7858634455a2153d6430719956", "object": "chat.completion.chunk", "created": 1702612156, "model": "mistral-tiny", "choices": [{"index": 0, "delta": {"role": null, "content": ""}, "finish_reason": "stop"}]}\n\n',
                b"data: [DONE]",
            ]
        ),
        headers={"content-type": "text/event-stream"},
    )
    model = llm.get_model("mistral-tiny")
    model.key = "KEY"
    response = model.prompt("How are you?")
    chunks = list(response)
    assert chunks == ["I am an AI", ""]
