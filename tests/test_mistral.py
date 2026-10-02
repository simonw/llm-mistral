import json
import pathlib
from types import SimpleNamespace
import pytest
from pytest_httpx import IteratorStream
import llm
from llm.tools import llm_version

canonical_api = pytest.mark.skipif(
    not hasattr(llm.models.Prompt, "messages"), reason="Requires canonical messages API"
)


def canonical_prompt(messages, prefix=None):
    return SimpleNamespace(
        messages=messages,
        prompt="current",
        system=None,
        attachments=[],
        tool_results=[],
        options=SimpleNamespace(prefix=prefix),
    )


@canonical_api
@pytest.mark.parametrize("async_", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_cli_continues_normalized_logs(
    tmp_path, monkeypatch, httpx_mock, async_, stream
):
    from click.testing import CliRunner
    from llm.cli import cli

    monkeypatch.setenv("LLM_USER_PATH", str(tmp_path))
    (tmp_path / "mistral_models.json").write_text(json.dumps(TEST_MODELS))
    for text in ("Hi from fixture", "Again"):
        if stream:
            event = {"choices": [{"delta": {"content": text}}]}
            httpx_mock.add_response(
                method="POST",
                url="https://api.mistral.ai/v1/chat/completions",
                content="data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n",
                headers={"content-type": "text/event-stream"},
            )
        else:
            httpx_mock.add_response(
                method="POST",
                url="https://api.mistral.ai/v1/chat/completions",
                json={"choices": [{"message": {"role": "assistant", "content": text}}]},
            )
    flags = (["--async"] if async_ else []) + ([] if stream else ["--no-stream"])
    runner = CliRunner()
    first = runner.invoke(cli, ["-m", "mistral-tiny", *flags, "say hi"])
    assert first.exit_code == 0, first.output
    second = runner.invoke(cli, ["-m", "mistral-tiny", *flags, "-c", "say it again"])
    assert second.exit_code == 0, second.output
    requests = httpx_mock.get_requests()
    assert len(requests) == 2
    assert json.loads(requests[1].content)["messages"] == [
        {"role": "user", "content": "say hi"},
        {"role": "assistant", "content": "Hi from fixture"},
        {"role": "user", "content": "say it again"},
    ]


@canonical_api
def test_canonical_messages_are_authoritative():
    from llm.parts import Message, TextPart

    chain = [
        Message("system", [TextPart("first instructions")]),
        Message("user", [TextPart("old question")]),
        Message("assistant", [TextPart("old answer")]),
        Message("system", [TextPart("new instructions")]),
        Message("user", [TextPart("cur"), TextPart("rent")]),
    ]
    model = llm.get_model("mistral-tiny")
    assert model.build_messages(
        canonical_prompt(chain), SimpleNamespace(responses=[])
    ) == [
        {"role": "system", "content": "first instructions"},
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "system", "content": "new instructions"},
        {"role": "user", "content": "current"},
    ]


@canonical_api
def test_canonical_attachment_order():
    from llm.parts import AttachmentPart, Message, TextPart

    image = llm.Attachment(url="https://example.test/image.png", type="image/png")
    audio = llm.Attachment(url="https://example.test/audio.mp3", type="audio/mpeg")
    prompt = canonical_prompt(
        [
            Message(
                "user",
                [
                    TextPart("before"),
                    AttachmentPart(image),
                    TextPart("after"),
                    AttachmentPart(audio),
                ],
            )
        ]
    )
    assert llm.get_model("mistral-tiny").build_messages(prompt, None) == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "before"},
                {"type": "image_url", "image_url": image.url},
                {"type": "text", "text": "after"},
                {
                    "type": "input_audio",
                    "input_audio": {"data": audio.url, "format": "mp3"},
                },
            ],
        }
    ]


@canonical_api
def test_canonical_tools_keep_text_ids_and_result_output():
    from llm.parts import Message, TextPart, ToolCallPart, ToolResultPart

    prompt = canonical_prompt(
        [
            Message(
                "assistant",
                [TextPart("Checking"), ToolCallPart("lookup", {"id": 0}, "call00001")],
            ),
            Message(
                "tool",
                [
                    ToolResultPart("lookup", '{"found":false}', "call00001"),
                    ToolResultPart("other", "", "call00002"),
                ],
            ),
            Message("user", [TextPart("current")]),
        ]
    )
    assert llm.get_model("mistral-tiny").build_messages(prompt, None) == [
        {
            "role": "assistant",
            "content": "Checking",
            "tool_calls": [
                {
                    "id": "call00001",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"id": 0}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call00001", "content": '{"found":false}'},
        {"role": "tool", "tool_call_id": "call00002", "content": ""},
        {"role": "user", "content": "current"},
    ]


@canonical_api
def test_canonical_tool_result_attachments():
    from llm.parts import Message, ToolResultPart

    image = llm.Attachment(url="https://example.test/result.png", type="image/png")
    prompt = canonical_prompt(
        [
            Message(
                "tool",
                [ToolResultPart("lookup", "found", "call00001", attachments=[image])],
            )
        ]
    )
    assert llm.get_model("mistral-tiny").build_messages(prompt, None) == [
        {
            "role": "tool",
            "tool_call_id": "call00001",
            "content": [
                {"type": "text", "text": "found"},
                {"type": "image_url", "image_url": image.url},
            ],
        }
    ]


@canonical_api
@pytest.mark.parametrize(
    "output", [{"value": 0}, [1, {"value": False}], 0, False, None]
)
def test_canonical_tool_result_json_values(output):
    from llm.parts import Message, ToolResultPart

    prompt = canonical_prompt(
        [Message("tool", [ToolResultPart("lookup", output, "call00001")])]
    )
    assert llm.get_model("mistral-tiny").build_messages(prompt, None) == [
        {"role": "tool", "tool_call_id": "call00001", "content": json.dumps(output)}
    ]


@canonical_api
def test_canonical_thinking_is_replayed_before_text():
    from llm.parts import Message, ReasoningPart, TextPart

    prompt = canonical_prompt(
        [Message("assistant", [TextPart("answer"), ReasoningPart("reason")])]
    )
    assert llm.get_model("mistral-tiny").build_messages(prompt, None) == [
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": [{"type": "text", "text": "reason"}]},
                {"type": "text", "text": "answer"},
            ],
        }
    ]


@canonical_api
def test_canonical_multiple_thinking_and_text_fragments():
    from llm.parts import Message, ReasoningPart, TextPart

    prompt = canonical_prompt(
        [
            Message(
                "assistant",
                [
                    ReasoningPart("first"),
                    TextPart("an"),
                    ReasoningPart("second"),
                    TextPart("swer"),
                ],
            )
        ]
    )
    assert llm.get_model("mistral-tiny").build_messages(prompt, None) == [
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": [{"type": "text", "text": "first"}]},
                {"type": "thinking", "thinking": [{"type": "text", "text": "second"}]},
                {"type": "text", "text": "an"},
                {"type": "text", "text": "swer"},
            ],
        }
    ]


@canonical_api
def test_canonical_redacted_and_opaque_metadata_not_invented_or_forwarded():
    from llm.parts import Message, ReasoningPart, TextPart

    prompt = canonical_prompt(
        [
            Message(
                "assistant",
                [
                    ReasoningPart("withheld", redacted=True),
                    TextPart("answer", provider_metadata={"token": "opaque"}),
                ],
                provider_metadata={"role": "system", "secret": "opaque"},
            )
        ]
    )
    assert llm.get_model("mistral-tiny").build_messages(prompt, None) == [
        {"role": "assistant", "content": "answer"}
    ]


@canonical_api
def test_canonical_prefix_follows_full_history():
    from llm.parts import Message, TextPart

    prompt = canonical_prompt(
        [
            Message("user", [TextPart("past")]),
            Message("assistant", [TextPart("reply")]),
            Message("user", [TextPart("current")]),
        ],
        prefix="Start",
    )
    assert llm.get_model("mistral-tiny").build_messages(prompt, None) == [
        {"role": "user", "content": "past"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "current"},
        {"role": "assistant", "content": "Start", "prefix": True},
    ]


@canonical_api
def test_canonical_unknown_part_fails_instead_of_dropping_context():
    from llm.parts import Message, Part

    with pytest.raises(ValueError, match="Unsupported"):
        llm.get_model("mistral-tiny").build_messages(
            canonical_prompt([Message("user", [Part()])]), None
        )


def test_legacy_message_fallback_preserves_history():
    previous = SimpleNamespace(
        prompt=SimpleNamespace(system="instructions", prompt="past", tool_results=[]),
        attachments=[],
        tool_calls_or_raise=lambda: [],
        text_or_raise=lambda: "reply",
    )
    prompt = canonical_prompt([])
    del prompt.messages
    assert llm.get_model("mistral-tiny").build_messages(
        prompt, SimpleNamespace(responses=[previous])
    ) == [
        {"role": "system", "content": "instructions"},
        {"role": "user", "content": "past"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "current"},
    ]


TEST_MODELS = {
    "data": [
        {
            "id": "mistral-tiny",
            "name": "Mistral Tiny",
            "description": "A tiny model",
        },
        {
            "id": "mistral-small",
            "name": "Mistral Small",
            "description": "A small model",
        },
        {
            "id": "mistral-medium",
            "name": "Mistral Small",
            "description": "A small model",
        },
        {
            "id": "mistral-large-largest",
            "name": "Mistral Large",
            "description": "A large model",
        },
        {
            "id": "mistral-other",
            "name": "Mistral Other",
            "description": "Another model",
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
    llm_user_path = new_tmp_dir = str(tmpdir / "llm")
    monkeypatch.setenv("LLM_USER_PATH", llm_user_path)
    # Should not have llm_user_path / mistral_models.json
    path = pathlib.Path(llm_user_path) / "mistral_models.json"
    assert not path.exists()
    # Listing models should create that file
    models_with_aliases = llm.get_models_with_aliases()
    assert path.exists()
    # Should have called that API
    response = httpx_mock.get_request()
    assert response.url == "https://api.mistral.ai/v1/models"


@pytest.fixture
def mocked_stream(httpx_mock):
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
        url="https://api.mistral.ai/v1/chat/completions",
        method="POST",
        stream=IteratorStream(chunks1),
        headers={"content-type": "text/event-stream"},
    )

    # Add second response
    httpx_mock.add_response(
        url="https://api.mistral.ai/v1/chat/completions",
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
    assert chunks == ["I am an AI", ""]
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
    assert chunks == ["I am an AI", ""]
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
        safe_mode=True,
        max_tokens=10,
    ).text()
    request = mocked_stream.get_request()
    assert json.loads(request.content) == {
        "model": "mistral-tiny",
        "messages": [{"role": "user", "content": "How are you?"}],
        "temperature": 0.5,
        "top_p": 0.8,
        "random_seed": 42,
        "safe_mode": True,
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
