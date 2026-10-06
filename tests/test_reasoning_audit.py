"""Regression probes for the reasoning spec; xfails document confirmed gaps."""

import json
from pathlib import Path

import llm
import pytest
from click.testing import CliRunner
from llm.cli import cli
from llm.parts import ReasoningPart, TextPart, ToolCallPart
from pydantic import ValidationError
from pytest_httpx import IteratorStream

from test_mistral import (
    TEST_MODELS,
    llm_user_path,
    mock_env,
    mocked_stream,
    reasoning_response,
)


@pytest.mark.parametrize("async_", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_logged_reasoning_continuation(
    reasoning_response, httpx_mock, monkeypatch, tmp_path, async_, stream
):
    monkeypatch.setenv("LLM_USER_PATH", str(tmp_path))
    (tmp_path / "mistral_models.json").write_text(json.dumps(TEST_MODELS))
    runner = CliRunner()
    flags = ([] if stream else ["--no-stream"]) + (["--async"] if async_ else [])
    reasoning_response(stream)
    first = runner.invoke(
        cli, ["-m", "mistral-tiny", "First question", "-s", "Be brief"] + flags
    )
    assert first.exit_code == 0, first.output
    logs = runner.invoke(cli, ["logs", "-n", "1", "--json"])
    assert logs.exit_code == 0, logs.output
    assert json.loads(logs.output)[0]["reasoning"] == "First thought.Second thought."
    reasoning_response(stream)
    second = runner.invoke(cli, ["-c", "Follow up"] + flags)
    assert second.exit_code == 0, second.output
    messages = json.loads(httpx_mock.get_requests()[1].content)["messages"]
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[0]["content"] == "Be brief"
    assert messages[1]["content"] == "First question"
    assert messages[3]["content"] == "Follow up"
    assert [c.get("signature") for c in messages[2]["content"]] == [
        "first-signature",
        "second-signature",
        None,
    ]
    assert messages[2]["content"][-1] == {"type": "text", "text": "The answer."}


@pytest.mark.xfail(
    strict=True, reason="Reasoning options are currently offered on every model"
)
@pytest.mark.parametrize(
    "option,value", [("reasoning_effort", "high"), ("prompt_mode", "reasoning")]
)
def test_non_reasoning_model_rejects_options(option, value, monkeypatch, tmp_path):
    models = {
        "data": [
            {
                "id": "plain-model",
                "capabilities": {"completion_chat": True, "reasoning": False},
            }
        ]
    }
    monkeypatch.setenv("LLM_USER_PATH", str(tmp_path))
    (tmp_path / "mistral_models.json").write_text(json.dumps(models))
    model = llm.get_model("mistral/plain-model")
    with pytest.raises(ValidationError):
        model.Options(**{option: value})


@pytest.mark.xfail(
    strict=True,
    reason="Empty redacted reasoning is currently replayed as an empty thinking chunk",
)
def test_redacted_reasoning_is_not_replayed():
    model = llm.get_model("mistral-tiny")
    assert (
        model._content_from_parts(
            [ReasoningPart(text="", redacted=True), TextPart(text="Answer")]
        )
        == "Answer"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("async_", [False, True])
@pytest.mark.parametrize("stream", [False, True])
async def test_thinking_then_tool_call(httpx_mock, async_, stream):
    thinking = {
        "type": "thinking",
        "thinking": [{"type": "text", "text": "Use a tool."}],
        "signature": "tool-signature",
    }
    tool_call = {
        "id": "call12345",
        "type": "function",
        "function": {"name": "lookup", "arguments": '{"value":42}'},
    }
    base = {
        "object": "chat.completion",
        "id": "tool-reasoning",
        "model": "mistral-tiny",
        "created": 1702614202,
    }
    if stream:
        deltas = [{"content": [thinking]}, {"tool_calls": [{**tool_call, "index": 0}]}]
        chunks = [
            (
                "data: "
                + json.dumps(
                    {
                        **base,
                        "choices": [
                            {"index": 0, "delta": delta, "finish_reason": None}
                        ],
                    }
                )
                + "\n\n"
            ).encode()
            for delta in deltas
        ]
        httpx_mock.add_response(
            url="https://api.mistral.ai/v1/chat/completions#stream",
            stream=IteratorStream(chunks + [b"data: [DONE]\n\n"]),
            headers={"content-type": "text/event-stream"},
        )
    else:
        httpx_mock.add_response(
            url="https://api.mistral.ai/v1/chat/completions",
            json={
                **base,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": [thinking],
                            "tool_calls": [tool_call],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 10,
                    "total_tokens": 15,
                },
            },
        )

    def lookup(value: int):
        return value

    model = (llm.get_async_model if async_ else llm.get_model)("mistral-tiny")
    response = model.prompt("Look up 42", tools=[lookup], stream=stream)
    events = (
        [event async for event in response.astream_events()]
        if async_
        else list(response.stream_events())
    )
    messages = await response.messages() if async_ else response.messages()
    assert [p.__class__ for p in messages[0].parts] == [ReasoningPart, ToolCallPart]
    assert (
        messages[0].parts[0].provider_metadata["mistral"]["signature"]
        == "tool-signature"
    )
    assert messages[0].parts[1].arguments == {"value": 42}
    assert [e.type for e in events if e.chunk] == [
        "reasoning",
        "tool_call_name",
        "tool_call_args",
    ]
    assert (await response.text() if async_ else response.text()) == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("async_", [False, True])
@pytest.mark.parametrize("stream", [False, True])
async def test_recorded_magistral_response(httpx_mock, async_, stream):
    root = Path(__file__).parent / "fixtures" / "reasoning"
    if stream:
        httpx_mock.add_response(
            url="https://api.mistral.ai/v1/chat/completions#stream",
            stream=IteratorStream([(root / "magistral-stream.sse").read_bytes()]),
            headers={"content-type": "text/event-stream"},
        )
    else:
        httpx_mock.add_response(
            url="https://api.mistral.ai/v1/chat/completions",
            json=json.loads((root / "magistral-completion.json").read_text()),
        )
    response = (llm.get_async_model if async_ else llm.get_model)(
        "mistral-tiny"
    ).prompt("What is 17 times 19? Answer briefly.", stream=stream)
    text = await response.text() if async_ else response.text()
    messages = await response.messages() if async_ else response.messages()
    assert "323" in text
    assert any(isinstance(p, TextPart) for p in messages[0].parts)


@pytest.fixture
def recorded_high(httpx_mock):
    def add(stream):
        root = Path(__file__).parent / "fixtures" / "reasoning"
        if stream:
            data = (root / "magistral-high-stream.sse").read_bytes()
            httpx_mock.add_response(
                url="https://api.mistral.ai/v1/chat/completions#stream",
                stream=IteratorStream([data]),
                headers={"content-type": "text/event-stream"},
            )
            rows = [
                json.loads(line[6:])
                for line in data.decode().splitlines()
                if line.startswith("data: {")
            ]
            chunks = [
                part
                for row in rows
                for choice in row["choices"]
                for part in (choice["delta"].get("content") or [])
                if isinstance(part, dict)
            ]
        else:
            data = json.loads((root / "magistral-high-completion.json").read_text())
            httpx_mock.add_response(
                url="https://api.mistral.ai/v1/chat/completions", json=data
            )
            chunks = data["choices"][0]["message"]["content"]
        return "".join(
            inner["text"]
            for chunk in chunks
            if chunk["type"] == "thinking"
            for inner in chunk["thinking"]
            if inner["type"] == "text"
        )

    return add


@pytest.mark.asyncio
@pytest.mark.parametrize("async_", [False, True])
@pytest.mark.parametrize("stream", [False, True])
async def test_recorded_magistral_thinking_is_preserved(recorded_high, async_, stream):
    expected = recorded_high(stream)
    response = (llm.get_async_model if async_ else llm.get_model)(
        "mistral-tiny"
    ).prompt(
        "What is 17 times 19? Answer briefly.", stream=stream, reasoning_effort="high"
    )
    text = await response.text() if async_ else response.text()
    messages = await response.messages() if async_ else response.messages()
    assert "323" in text
    assert expected
    assert (
        "".join(p.text for p in messages[0].parts if isinstance(p, ReasoningPart))
        == expected
    )
    assert expected not in text


@pytest.mark.asyncio
@pytest.mark.parametrize("async_", [False, True])
async def test_streamed_reasoning_is_not_split_on_closed_flag(recorded_high, async_):
    expected = recorded_high(True)
    response = (llm.get_async_model if async_ else llm.get_model)(
        "mistral-tiny"
    ).prompt("What is 17 times 19? Answer briefly.")
    if async_:
        await response.text()
        messages = await response.messages()
    else:
        response.text()
        messages = response.messages()
    parts = messages[0].parts
    assert len([p for p in parts if isinstance(p, ReasoningPart)]) == 1
    assert parts[0].text == expected


def test_prefix_is_last_after_history(mocked_stream):
    from llm.parts import assistant, user

    response = llm.get_model("mistral-tiny").prompt(
        messages=[user("First"), assistant("Answer"), user("Next")], prefix="Start here"
    )
    response.text()
    messages = json.loads(mocked_stream.get_request().content)["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]
    assert messages[-1] == {
        "role": "assistant",
        "content": "Start here",
        "prefix": True,
    }


def test_vision_attachment_is_preserved(mocked_stream):
    from llm_mistral import Mistral

    model = Mistral("mistral/vision-test", "vision-test", True, True, True, False)
    response = model.prompt(
        "Describe",
        attachments=[llm.Attachment(type="image/png", content=b"fake image")],
    )
    response.text()
    assert json.loads(mocked_stream.get_request().content)["messages"][0][
        "content"
    ] == [
        {"type": "text", "text": "Describe"},
        {"type": "image_url", "image_url": "data:image/png;base64,ZmFrZSBpbWFnZQ=="},
    ]
