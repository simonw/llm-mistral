import json
import pathlib
import pytest
from pytest_httpx import IteratorStream
import llm

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
