import json
import pathlib
import pytest
from pytest_httpx import IteratorStream
import llm
from llm.parts import ReasoningPart, StreamEvent, TextPart, ToolCallPart
from llm.tools import llm_version

TEST_MODELS = {
    "data": [
        {
            "id": "mistral-tiny",
            "type": "base",
            "capabilities": {"completion_chat": True, "function_calling": True},
            "name": "Mistral Tiny",
            "description": "A tiny model",
        },
        {
            "id": "mistral-small",
            "type": "base",
            "capabilities": {"completion_chat": True, "function_calling": True},
            "name": "Mistral Small",
            "description": "A small model",
        },
        {
            "id": "mistral-medium",
            "type": "base",
            "capabilities": {"completion_chat": True, "function_calling": True},
            "name": "Mistral Small",
            "description": "A small model",
        },
        {
            "id": "mistral-large-largest",
            "type": "base",
            "capabilities": {"completion_chat": True, "function_calling": True},
            "name": "Mistral Large",
            "description": "A large model",
        },
        {
            "id": "mistral-other",
            "type": "base",
            "capabilities": {"completion_chat": True, "function_calling": True},
            "name": "Mistral Other",
            "description": "Another model",
        },
        {
            "id": "voxtral-small-2507",
            "type": "base",
            "capabilities": {"completion_chat": True, "audio": True},
            "name": "Voxtral Small",
            "description": "An audio model",
        },
    ]
}


@pytest.fixture(scope="session")
def llm_user_path(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp("llm")
    return str(tmpdir)


# Fixture that always runs
@pytest.fixture(autouse=True)
def mock_env(monkeypatch, llm_user_path):
    monkeypatch.setenv("LLM_MISTRAL_KEY", "test_key")
    monkeypatch.setenv("LLM_USER_PATH", llm_user_path)
    # Write a mistral_models.json file
    (pathlib.Path(llm_user_path) / "mistral_models.json").write_text(
        json.dumps(TEST_MODELS, indent=2)
    )


def test_caches_models(monkeypatch, tmpdir, httpx_mock):
    httpx_mock.add_response(
        url="https://api.mistral.ai/v1/models",
        method="GET",
        json=TEST_MODELS,
    )
    llm_user_path = str(tmpdir / "llm")
    monkeypatch.setenv("LLM_USER_PATH", llm_user_path)
    # Should not have llm_user_path / mistral_models.json
    path = pathlib.Path(llm_user_path) / "mistral_models.json"
    assert not path.exists()
    # Listing models should create that file
    llm.get_models_with_aliases()
    assert path.exists()
    # Should have called that API
    response = httpx_mock.get_request()
    assert response.url == "https://api.mistral.ai/v1/models"


@pytest.fixture
def mocked_stream(httpx_mock):
    httpx_mock.add_response(
        url="https://api.mistral.ai/v1/chat/completions#stream",
        method="POST",
        stream=IteratorStream(
            [
                b'data: {"id": "cmpl-4243ee7858634455a2153d6430719956", "model": "mistral-tiny", "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]}\n\n',
                b'data: {"id": "cmpl-4243ee7858634455a2153d6430719956", "object": "chat.completion.chunk", "created": 1702612156, "model": "mistral-tiny", "choices": [{"index": 0, "delta": {"role": null, "content": "I am an AI"}, "finish_reason": null}]}\n\n',
                b'data: {"id": "cmpl-4243ee7858634455a2153d6430719956", "object": "chat.completion.chunk", "created": 1702612156, "model": "mistral-tiny", "choices": [{"index": 0, "delta": {"role": null, "content": ""}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9, "prompt_tokens_details": {"cached_tokens": 2}}}\n\n',
                b"data: [DONE]",
            ]
        ),
        headers={"content-type": "text/event-stream"},
    )
    return httpx_mock


@pytest.fixture
def mocked_tool_stream(httpx_mock):
    # First response - when model is called with tools
    chunks1 = [
        b'data: {"id":"755aa30c9826400b818018ed9a0d4f62","object":"chat.completion.chunk","created":1748464326,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}\n\n',
        b'data: {"id":"755aa30c9826400b818018ed9a0d4f62","object":"chat.completion.chunk","created":1748464326,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"tool_calls":[{"id":"4dI1Jw6NX","function":{"name":"llm_version","arguments":"{}"},"index":0}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":55,"total_tokens":73,"completion_tokens":18}}\n\n',
        b"data: [DONE]",
    ]

    # Second response - when model is called with tool results
    chunks2 = [
        b'data: {"id":"394ba88c55a44d4593f7c5b57c2fa74f","object":"chat.completion.chunk","created":1748464330,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}\n\n',
        b'data: {"id":"394ba88c55a44d4593f7c5b57c2fa74f","object":"chat.completion.chunk","created":1748464330,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":"The"},"finish_reason":null}]}\n\n',
        b'data: {"id":"394ba88c55a44d4593f7c5b57c2fa74f","object":"chat.completion.chunk","created":1748464330,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" installed version of LL"},"finish_reason":null}]}\n\n',
        b'data: {"id":"394ba88c55a44d4593f7c5b57c2fa74f","object":"chat.completion.chunk","created":1748464330,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":"M is 0"},"finish_reason":null}]}\n\n',
        b'data: {"id":"394ba88c55a44d4593f7c5b57c2fa74f","object":"chat.completion.chunk","created":1748464330,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":".26."},"finish_reason":null}]}\n\n',
        b'data: {"id":"394ba88c55a44d4593f7c5b57c2fa74f","object":"chat.completion.chunk","created":1748464330,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":""},"finish_reason":"stop"}],"usage":{"prompt_tokens":105,"total_tokens":119,"completion_tokens":14}}\n\n',
        b"data: [DONE]",
    ]

    # Add first response
    httpx_mock.add_response(
        url="https://api.mistral.ai/v1/chat/completions#stream",
        method="POST",
        stream=IteratorStream(chunks1),
        headers={"content-type": "text/event-stream"},
    )

    # Add second response
    httpx_mock.add_response(
        url="https://api.mistral.ai/v1/chat/completions#stream",
        method="POST",
        stream=IteratorStream(chunks2),
        headers={"content-type": "text/event-stream"},
    )

    return httpx_mock


@pytest.fixture
def mocked_no_stream(httpx_mock):
    httpx_mock.add_response(
        url="https://api.mistral.ai/v1/chat/completions",
        method="POST",
        json={
            "id": "cmpl-362653b3050c4939bfa423af5f97709b",
            "object": "chat.completion",
            "created": 1702614202,
            "model": "mistral-tiny",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "I'm just a computer program, I don't have feelings.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 16, "total_tokens": 79, "completion_tokens": 63},
        },
    )
    return httpx_mock


def test_stream(mocked_stream):
    model = llm.get_model("mistral-tiny")
    response = model.prompt("How are you?")
    chunks = list(response)
    assert chunks == ["I am an AI"]
    request = mocked_stream.get_request()
    assert json.loads(request.content) == {
        "model": "mistral-tiny",
        "messages": [{"role": "user", "content": "How are you?"}],
        "temperature": 0.7,
        "top_p": 1,
        "stream": True,
    }


@pytest.mark.asyncio
async def test_stream_async(mocked_stream):
    model = llm.get_async_model("mistral-tiny")
    response = await model.prompt("How are you?")
    chunks = [item async for item in response]
    assert chunks == ["I am an AI"]
    request = mocked_stream.get_request()
    assert json.loads(request.content) == {
        "model": "mistral-tiny",
        "messages": [{"role": "user", "content": "How are you?"}],
        "temperature": 0.7,
        "top_p": 1,
        "stream": True,
    }


@pytest.mark.asyncio
async def test_async_no_stream(mocked_no_stream):
    model = llm.get_async_model("mistral-tiny")
    response = await model.prompt("How are you?", stream=False)
    text = await response.text()
    assert text == "I'm just a computer program, I don't have feelings."


def test_stream_with_options(mocked_stream):
    model = llm.get_model("mistral-tiny")
    model.prompt(
        "How are you?",
        temperature=0.5,
        top_p=0.8,
        random_seed=42,
        safe_prompt=True,
        max_tokens=10,
    ).text()
    request = mocked_stream.get_request()
    assert json.loads(request.content) == {
        "model": "mistral-tiny",
        "messages": [{"role": "user", "content": "How are you?"}],
        "temperature": 0.5,
        "top_p": 0.8,
        "random_seed": 42,
        "safe_prompt": True,
        "max_tokens": 10,
        "stream": True,
    }


def test_no_stream(mocked_no_stream):
    model = llm.get_model("mistral-tiny")
    response = model.prompt("How are you?", stream=False)
    assert response.text() == "I'm just a computer program, I don't have feelings."


def test_tools_stream(mocked_tool_stream):
    model = llm.get_model("mistral/mistral-medium")
    tool_calls = []
    chain_response = model.chain(
        "llm_version",
        tools=[llm_version],
        before_call=print,
        after_call=lambda *args: tool_calls.append(args),
    )
    output = chain_response.text()
    assert output == "The installed version of LLM is 0.26."


def test_stream_events_usage_and_response_json(mocked_stream):
    model = llm.get_model("mistral-tiny")
    response = model.prompt("How are you?")
    events = list(response.stream_events())
    assert all(isinstance(event, StreamEvent) for event in events)
    assert [event.chunk for event in events if event.type == "text"] == ["I am an AI"]
    assert response.resolved_model == "mistral-tiny"
    assert response.input_tokens == 5
    assert response.output_tokens == 4
    assert response.token_details == {"prompt_tokens_details": {"cached_tokens": 2}}
    assert len(response.response_json["chunks"]) == 3


def test_explicit_messages(mocked_no_stream):
    model = llm.get_model("mistral-tiny")
    response = model.prompt(
        messages=[
            llm.user("Hello"),
            llm.assistant("Hi!"),
            llm.user("What can you do?"),
        ],
        stream=False,
    )
    response.text()
    request = mocked_no_stream.get_request()
    assert json.loads(request.content)["messages"] == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!", "prefix": False},
        {"role": "user", "content": "What can you do?"},
    ]


def test_reasoning_parts_and_replay(httpx_mock):
    httpx_mock.add_response(
        url="https://api.mistral.ai/v1/chat/completions",
        method="POST",
        json={
            "id": "cmpl-reasoning",
            "object": "chat.completion",
            "created": 1702614202,
            "model": "magistral-small-latest",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "thinking",
                                "thinking": [{"type": "text", "text": "Work it out."}],
                                "signature": "reasoning-signature",
                            },
                            {"type": "text", "text": "The answer is 42."},
                        ],
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 8,
                "completion_tokens": 7,
                "total_tokens": 15,
            },
        },
    )
    httpx_mock.add_response(
        url="https://api.mistral.ai/v1/chat/completions",
        method="POST",
        json={
            "id": "cmpl-followup",
            "object": "chat.completion",
            "created": 1702614203,
            "model": "mistral-tiny",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Yes."},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 1,
                "total_tokens": 21,
            },
        },
    )
    model = llm.get_model("mistral-tiny")
    first = model.prompt("What is six times seven?", stream=False)
    assert first.text() == "The answer is 42."
    first_parts = first.messages()[0].parts
    assert isinstance(first_parts[0], ReasoningPart)
    assert first_parts[0].text == "Work it out."
    assert isinstance(first_parts[1], TextPart)

    second = first.reply("Are you sure?", stream=False)
    assert second.text() == "Yes."
    second_request = json.loads(httpx_mock.get_requests()[1].content)
    assistant_content = second_request["messages"][1]["content"]
    assert assistant_content[0] == {
        "type": "thinking",
        "thinking": [{"type": "text", "text": "Work it out."}],
        "signature": "reasoning-signature",
    }
    assert assistant_content[1] == {"type": "text", "text": "The answer is 42."}


def test_streaming_reasoning_preserves_complete_block(httpx_mock):
    chunks = [
        b'data: {"id":"reasoning-stream","model":"magistral-small-latest","choices":[{"index":0,"delta":{"content":[{"type":"thinking","thinking":[{"type":"text","text":"Work "}]}]},"finish_reason":null}]}\n\n',
        b'data: {"id":"reasoning-stream","model":"magistral-small-latest","choices":[{"index":0,"delta":{"content":[{"type":"thinking","thinking":[{"type":"text","text":"it out."}],"signature":"stream-signature","closed":true},{"type":"text","text":"The answer."}]},"finish_reason":"stop"}],"usage":{"prompt_tokens":8,"completion_tokens":6,"total_tokens":14}}\n\n',
        b"data: [DONE]\n\n",
    ]
    httpx_mock.add_response(
        url="https://api.mistral.ai/v1/chat/completions#stream",
        method="POST",
        stream=IteratorStream(chunks),
        headers={"content-type": "text/event-stream"},
    )
    model = llm.get_model("mistral-tiny")
    response = model.prompt("Think about this")
    events = list(response.stream_events())
    assert [(event.type, event.chunk) for event in events] == [
        ("reasoning", "Work "),
        ("reasoning", "it out."),
        ("reasoning", ""),
        ("text", "The answer."),
    ]
    parts = response.messages()[0].parts
    assert isinstance(parts[0], ReasoningPart)
    assert parts[0].text == "Work it out."
    assert parts[0].provider_metadata == {
        "mistral": {
            "content_chunk": {
                "type": "thinking",
                "thinking": [
                    {"type": "text", "text": "Work "},
                    {"type": "text", "text": "it out."},
                ],
                "signature": "stream-signature",
                "closed": True,
            },
            "signature": "stream-signature",
            "closed": True,
        }
    }
    assert isinstance(parts[1], TextPart)


def test_split_streaming_tool_arguments(httpx_mock):
    chunks = [
        b'data: {"id":"tool-stream","model":"mistral-medium","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":"lookup","arguments":"{\\"value\\":"}}]},"finish_reason":null}]}\n\n',
        b'data: {"id":"tool-stream","model":"mistral-medium","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":"","arguments":"42}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}\n\n',
        b"data: [DONE]\n\n",
    ]
    httpx_mock.add_response(
        url="https://api.mistral.ai/v1/chat/completions#stream",
        method="POST",
        stream=IteratorStream(chunks),
        headers={"content-type": "text/event-stream"},
    )

    def lookup(value: int):
        """Look up a value."""
        return value

    model = llm.get_model("mistral/mistral-medium")
    response = model.prompt("Look it up", tools=[lookup])
    events = list(response.stream_events())
    assert [event.type for event in events] == [
        "tool_call_name",
        "tool_call_args",
        "tool_call_args",
    ]
    assert [event.chunk for event in events] == ["lookup", '{"value":', "42}"]
    event_tool_call_ids = {event.tool_call_id for event in events}
    assert len(event_tool_call_ids) == 1
    fallback_id = event_tool_call_ids.pop()
    assert fallback_id.startswith("mistral-tool-")
    tool_call = response.tool_calls()[0]
    assert tool_call.name == "lookup"
    assert tool_call.arguments == {"value": 42}
    assert tool_call.tool_call_id == fallback_id
    tool_call_parts = [
        part for part in response.messages()[0].parts if isinstance(part, ToolCallPart)
    ]
    assert len(tool_call_parts) == 1
    assert tool_call_parts[0].tool_call_id == fallback_id


def test_zero_and_false_options(mocked_stream):
    model = llm.get_model("mistral-tiny")
    model.prompt(
        "How are you?",
        temperature=0,
        top_p=0,
        random_seed=0,
        safe_prompt=False,
        max_tokens=0,
    ).text()
    body = json.loads(mocked_stream.get_request().content)
    assert body["temperature"] == 0
    assert body["top_p"] == 0
    assert body["random_seed"] == 0
    assert body["safe_prompt"] is False
    assert body["max_tokens"] == 0


def test_safe_mode_is_a_legacy_alias(mocked_stream):
    model = llm.get_model("mistral-tiny")
    model.prompt("How are you?", safe_mode=True).text()
    body = json.loads(mocked_stream.get_request().content)
    assert body["safe_prompt"] is True
    assert "safe_mode" not in body


def test_schema_uses_sdk_without_mutating_input(mocked_stream):
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }
    model = llm.get_model("mistral-tiny")
    model.prompt("Return a name", schema=schema).text()
    body = json.loads(mocked_stream.get_request().content)
    assert "additionalProperties" not in schema
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "data",
            "schema": {**schema, "additionalProperties": False},
            "strict": True,
        },
    }


def test_local_audio_attachment_uses_sdk_shape(mocked_stream):
    model = llm.get_model("voxtral-small")
    model.prompt(
        "Transcribe this",
        attachments=[llm.Attachment(type="audio/mpeg", content=b"fake mp3")],
    ).text()
    body = json.loads(mocked_stream.get_request().content)
    assert body["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Transcribe this"},
                {"type": "input_audio", "input_audio": "ZmFrZSBtcDM="},
            ],
        }
    ]


def test_sdk_errors_are_model_errors(httpx_mock):
    httpx_mock.add_response(
        url="https://api.mistral.ai/v1/chat/completions#stream",
        method="POST",
        status_code=401,
        json={"type": "invalid_api_key", "message": "Bad API key"},
    )
    model = llm.get_model("mistral-tiny")
    with pytest.raises(llm.ModelError, match="401: invalid_api_key - Bad API key"):
        model.prompt("Hello").text()


def test_embeddings_use_sdk(httpx_mock):
    httpx_mock.add_response(
        url="https://api.mistral.ai/v1/embeddings",
        method="POST",
        json={
            "id": "embeddings-1",
            "object": "list",
            "model": "mistral-embed",
            "usage": {
                "prompt_tokens": 2,
                "completion_tokens": 0,
                "total_tokens": 2,
            },
            "data": [
                {"object": "embedding", "embedding": [0.1, 0.2], "index": 0},
                {"object": "embedding", "embedding": [0.3, 0.4], "index": 1},
            ],
        },
    )
    model = llm.get_embedding_model("mistral-embed")
    assert model.embed_batch(["one", "two"]) == [[0.1, 0.2], [0.3, 0.4]]
    request = httpx_mock.get_request()
    assert json.loads(request.content) == {
        "model": "mistral-embed",
        "input": ["one", "two"],
    }


@pytest.fixture
def reasoning_response(httpx_mock):
    def add(stream):
        blocks = [
            {
                "type": "thinking",
                "thinking": [{"type": "text", "text": "First thought."}],
                "signature": "first-signature",
                "closed": True,
            },
            {
                "type": "thinking",
                "thinking": [{"type": "text", "text": "Second thought."}],
                "signature": "second-signature",
                "closed": True,
            },
            {"type": "text", "text": "The answer."},
        ]
        base = {
            "id": "reasoning-test",
            "model": "magistral-small-latest",
            "created": 1702614202,
        }
        usage = {"prompt_tokens": 8, "completion_tokens": 12, "total_tokens": 20}
        if stream:
            contents = [
                [
                    {
                        "type": "thinking",
                        "thinking": [{"type": "text", "text": "First "}],
                        "closed": True,
                    }
                ],
                "",  # Empty deltas must not detach the eventual signature.
                [{**blocks[0], "thinking": [{"type": "text", "text": "thought."}]}],
                [blocks[1]],
                [blocks[2]],
            ]
            chunks = [
                (
                    "data: "
                    + json.dumps(
                        {
                            **base,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": content},
                                    "finish_reason": None,
                                }
                            ],
                            "usage": usage,
                        }
                    )
                    + "\n\n"
                ).encode()
                for content in contents
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
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": blocks},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": usage,
                },
            )
        return blocks

    return add


@pytest.mark.asyncio
@pytest.mark.parametrize("async_", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "effort", [None, "none", "minimal", "low", "medium", "high", "xhigh"]
)
async def test_reasoning_options_and_replay(
    reasoning_response, httpx_mock, async_, stream, effort
):
    blocks = reasoning_response(stream)
    get_model = llm.get_async_model if async_ else llm.get_model
    options = {"reasoning_effort": effort, "prompt_mode": "reasoning"} if effort else {}
    response = get_model("mistral-tiny").prompt("Think", stream=stream, **options)
    if async_:
        assert await response.text() == "The answer."
        messages = await response.messages()
    else:
        assert response.text() == "The answer."
        messages = response.messages()
    parts = messages[0].parts
    assert [type(part) for part in parts] == [ReasoningPart, ReasoningPart, TextPart]
    assert [part.text for part in parts] == [
        "First thought.",
        "Second thought.",
        "The answer.",
    ]
    assert [part.provider_metadata["mistral"]["signature"] for part in parts[:2]] == [
        "first-signature",
        "second-signature",
    ]
    request = json.loads(httpx_mock.get_requests()[0].content)
    if effort:
        assert request["reasoning_effort"] == effort
        assert request["prompt_mode"] == "reasoning"
    else:
        assert "reasoning_effort" not in request
        assert "prompt_mode" not in request

    reasoning_response(stream)
    if async_:
        followup = await response.reply("Continue", stream=stream)
        await followup.text()
    else:
        followup = response.reply("Continue", stream=stream)
        followup.text()
    replay = json.loads(httpx_mock.get_requests()[1].content)["messages"][1]["content"]
    # Streaming keeps the provider's text fragments intact for signed replay.
    if stream:
        blocks[0]["thinking"] = [
            {"type": "text", "text": "First "},
            {"type": "text", "text": "thought."},
        ]
    assert replay == blocks


@pytest.mark.parametrize(
    "option,value", [("reasoning_effort", "max"), ("prompt_mode", "normal")]
)
def test_invalid_reasoning_options(option, value):
    from pydantic import ValidationError
    from llm_mistral import Mistral

    with pytest.raises(ValidationError):
        Mistral.Options(**{option: value})


@pytest.mark.parametrize("async_", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("hide", [False, True])
def test_reasoning_cli_display(reasoning_response, async_, stream, hide):
    from click.testing import CliRunner
    from llm.cli import cli

    reasoning_response(stream)
    args = ["-m", "mistral-tiny", "Think", "-o", "reasoning_effort", "high", "--no-log"]
    if async_:
        args.append("--async")
    if not stream:
        args.append("--no-stream")
    if hide:
        args.append("-R")
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    assert result.stdout == "The answer.\n"
    if hide or not stream:
        assert result.stderr == ""
    else:
        assert "First thought.Second thought." in result.stderr
        assert "The answer." not in result.stderr
