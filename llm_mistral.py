import copy
import json
from itertools import count
import uuid
from typing import Literal, Optional

import click
import llm
from llm.parts import (
    AttachmentPart,
    ReasoningPart,
    StreamEvent,
    TextPart,
    ToolCallPart,
    ToolResultPart,
)
from mistralai.client import Mistral as MistralClient
from mistralai.client.errors import MistralError
from pydantic import Field

DEFAULT_ALIASES = {
    "mistral/mistral-tiny": "mistral-tiny",
    "mistral/open-mistral-nemo": "mistral-nemo",
    "mistral/mistral-small-2312": "mistral-small-2312",
    "mistral/mistral-small-2402": "mistral-small-2402",
    "mistral/mistral-small-2409": "mistral-small-2409",
    "mistral/mistral-small-2501": "mistral-small-2501",
    "mistral/magistral-small-2506": "magistral-small-2506",
    "mistral/magistral-small-latest": "magistral-small",
    "mistral/mistral-small-latest": "mistral-small",
    "mistral/mistral-medium-2312": "mistral-medium-2312",
    "mistral/mistral-medium-2505": "mistral-medium-2505",
    "mistral/magistral-medium-2506": "magistral-medium-2506",
    "mistral/magistral-medium-latest": "magistral-medium",
    "mistral/mistral-medium-latest": "mistral-medium",
    "mistral/mistral-large-latest": "mistral-large",
    "mistral/codestral-mamba-latest": "codestral-mamba",
    "mistral/codestral-latest": "codestral",
    "mistral/ministral-3b-latest": "ministral-3b",
    "mistral/ministral-8b-latest": "ministral-8b",
    "mistral/pixtral-12b-latest": "pixtral-12b",
    "mistral/pixtral-large-latest": "pixtral-large",
    "mistral/devstral-small-latest": "devstral-small",
    "mistral/voxtral-mini-2507": "voxtral-mini",
    "mistral/voxtral-small-2507": "voxtral-small",
}

tool_models = {
    "mistral/mistral-large-latest",
    "mistral/mistral-medium-2312",
    "mistral/mistral-medium-2505",
    "mistral/mistral-medium-latest",
    "mistral/mistral-small-2312",
    "mistral/mistral-small-2402",
    "mistral/mistral-small-2409",
    "mistral/mistral-small-2501",
    "mistral/mistral-small-latest",
    "mistral/mistral-small",
    "mistral/mistral-medium",
    "mistral/devstral-small-latest",
    "mistral/codestral-latest",
    "mistral/ministral-8b-latest",
    "mistral/ministral-3b-latest",
    "mistral/pixtral-12b-latest",
    "mistral/pixtral-large-latest",
    "mistral/open-mistral-nemo",
}


@llm.hookimpl
def register_models(register):
    for model in get_model_details():
        model_id = model["id"]
        capabilities = model.get("capabilities") or {}
        vision = capabilities.get("vision", False)
        audio = capabilities.get("audio", "voxtral" in model_id)
        our_model_id = "mistral/" + model_id
        alias = DEFAULT_ALIASES.get(our_model_id)
        aliases = [alias] if alias else []
        schemas = "codestral-mamba" not in model_id
        tools = capabilities.get("function_calling", our_model_id in tool_models)
        reasoning = capabilities.get("reasoning", model_id.startswith("magistral-"))
        register(
            Mistral(our_model_id, model_id, vision, schemas, tools, audio, reasoning),
            AsyncMistral(
                our_model_id, model_id, vision, schemas, tools, audio, reasoning
            ),
            aliases=aliases,
        )


@llm.hookimpl
def register_embedding_models(register):
    # alias here to avoid breaking backwards compatibility
    register(
        MistralEmbed(model_id="mistral/mistral-embed", model_name="mistral-embed"),
        aliases=("mistral-embed",),
    )
    # These don't get the alias
    for i in (256, 512, 1024, 1536, 3072):
        model_id = "mistral/codestral-embed-{}".format(i)
        aliases = None
        if i == 1536:
            aliases = ("codestral-embed",)
        register(
            MistralEmbed(
                model_id=model_id, model_name="codestral-embed", output_dimension=i
            ),
            aliases=aliases,
        )


def _sdk_dump(value):
    return value.model_dump(mode="json", by_alias=True, exclude_unset=True)


def _model_error(error):
    try:
        decoded = json.loads(error.body)
    except (json.JSONDecodeError, TypeError):
        decoded = {}
    if not isinstance(decoded, dict):
        decoded = {}
    error_type = decoded.get("type", type(error).__name__)
    message = decoded.get("message") or decoded.get("detail") or str(error)
    if not isinstance(message, str):
        message = json.dumps(message)
    # Avoid echoing huge base64 values or other unwieldy request details.
    message = " ".join(word[:30] for word in message.split())[:500]
    return llm.ModelError(f"{error.status_code}: {error_type} - {message}")


def refresh_models():
    user_dir = llm.user_dir()
    mistral_models = user_dir / "mistral_models.json"
    key = llm.get_key("", "mistral", "LLM_MISTRAL_KEY")
    if not key:
        raise click.ClickException(
            "You must set the 'mistral' key or the LLM_MISTRAL_KEY environment variable."
        )
    try:
        with MistralClient(api_key=key) as client:
            models = _sdk_dump(client.models.list())
    except MistralError as error:
        raise _model_error(error) from error
    mistral_models.write_text(json.dumps(models, indent=2))
    return models


def get_model_details():
    user_dir = llm.user_dir()
    models = {
        "data": [
            {"id": model_id.replace("mistral/", "")}
            for model_id in DEFAULT_ALIASES.keys()
        ]
    }
    mistral_models = user_dir / "mistral_models.json"
    if mistral_models.exists():
        models = json.loads(mistral_models.read_text())
    elif llm.get_key("", "mistral", "LLM_MISTRAL_KEY"):
        try:
            models = refresh_models()
        except llm.ModelError:
            pass
    details = []
    for model in models.get("data", []):
        if model.get("is_unknown") and isinstance(model.get("raw"), dict):
            model = model["raw"]
        model_id = model.get("id")
        if not model_id:
            continue
        capabilities = model.get("capabilities") or {}
        if capabilities.get("completion_chat", "embed" not in model_id):
            details.append(model)
    return details


def get_model_ids():
    return [model["id"] for model in get_model_details()]


@llm.hookimpl
def register_commands(cli):
    @cli.group()
    def mistral():
        "Commands relating to the llm-mistral plugin"

    @mistral.command()
    def models():
        "List of available Mistral models in JSON"
        click.echo(json.dumps(get_model_details(), indent=2))

    @mistral.command()
    def refresh():
        "Refresh the list of available Mistral models"
        before = set(get_model_ids())
        refresh_models()
        after = set(get_model_ids())
        added = after - before
        removed = before - after
        if added:
            click.echo(f"Added models: {', '.join(added)}", err=True)
        if removed:
            click.echo(f"Removed models: {', '.join(removed)}", err=True)
        if added or removed:
            click.echo("New list of models:", err=True)
            for model_id in get_model_ids():
                click.echo(model_id, err=True)
        else:
            click.echo("No changes", err=True)


def _attachment_chunk(attachment):
    attachment_type = attachment.resolve_type()
    if attachment_type.startswith("audio/"):
        return {
            "type": "input_audio",
            "input_audio": attachment.url or attachment.base64_content(),
        }
    return {
        "type": "image_url",
        "image_url": attachment.url
        or f"data:{attachment_type};base64,{attachment.base64_content()}",
    }


class _PartIndexes:
    """Keep streamed text together and signed reasoning blocks separate."""

    def __init__(self):
        self.indexes = count()
        self.text_index = None
        self.text_message_index = None

    def __next__(self):
        self.text_index = None
        return next(self.indexes)

    def text(self, message_index):
        if self.text_index is None or self.text_message_index != message_index:
            self.text_index = next(self.indexes)
            self.text_message_index = message_index
        return self.text_index


class _Shared:
    can_stream = True
    needs_key = "mistral"
    key_env_var = "LLM_MISTRAL_KEY"

    class Options(llm.Options):
        temperature: Optional[float] = Field(
            description=(
                "Determines the sampling temperature. Higher values like 0.8 increase randomness, "
                "while lower values like 0.2 make the output more focused and deterministic."
            ),
            ge=0,
            le=1,
            default=0.7,
        )
        top_p: Optional[float] = Field(
            description=(
                "Nucleus sampling, where the model considers the tokens with top_p probability mass. "
                "For example, 0.1 means considering only the tokens in the top 10% probability mass."
            ),
            ge=0,
            le=1,
            default=1,
        )
        max_tokens: Optional[int] = Field(
            description="The maximum number of tokens to generate in the completion.",
            ge=0,
            default=None,
        )
        safe_prompt: Optional[bool] = Field(
            description="Whether to inject a safety prompt before all conversations.",
            default=None,
        )
        random_seed: Optional[int] = Field(
            description="Sets the seed for random sampling to generate deterministic results.",
            default=None,
        )
        prefix: Optional[str] = Field(
            description="A prefix to prepend to the response.",
            default=None,
        )

    class ReasoningOptions(Options):
        reasoning_effort: Optional[
            Literal["none", "minimal", "low", "medium", "high", "xhigh"]
        ] = Field(
            description="Reasoning effort for supported models: none, minimal, low, medium, high or xhigh.",
            default=None,
        )

    def __init__(
        self,
        our_model_id,
        mistral_model_id,
        vision,
        schemas,
        tools,
        audio,
        reasoning=False,
    ):
        self.model_id = our_model_id
        self.mistral_model_id = mistral_model_id
        if reasoning:
            self.Options = self.ReasoningOptions
        attachment_types = set()
        if vision:
            attachment_types.update(
                {
                    "image/jpeg",
                    "image/png",
                    "image/gif",
                    "image/webp",
                }
            )
        if audio:
            attachment_types.update(
                {
                    "audio/mpeg",
                }
            )
        self.attachment_types = attachment_types
        self.supports_schema = schemas
        self.supports_tools = tools

    def _reasoning_chunk(self, part):
        metadata = (part.provider_metadata or {}).get("mistral", {})
        raw_chunk = metadata.get("content_chunk", {})
        if not isinstance(raw_chunk, dict):
            raw_chunk = {}
        raw_thinking = raw_chunk.get("thinking", [])
        raw_text = "".join(
            item.get("text", "")
            for item in raw_thinking
            if isinstance(item, dict) and item.get("type") == "text"
        )
        if raw_chunk.get("type") == "thinking" and raw_text == part.text:
            return copy.deepcopy(raw_chunk)
        non_text_thinking = [
            copy.deepcopy(item)
            for item in raw_thinking
            if isinstance(item, dict) and item.get("type") != "text"
        ]
        chunk = {
            "type": "thinking",
            "thinking": (
                ([{"type": "text", "text": part.text}] if part.text else [])
                + non_text_thinking
            ),
        }
        signature = metadata.get("signature", raw_chunk.get("signature"))
        if signature is not None:
            chunk["signature"] = signature
        closed = metadata.get("closed", raw_chunk.get("closed"))
        if closed is not None:
            chunk["closed"] = closed
        return chunk

    def _content_from_parts(self, parts):
        chunks = []
        for part in parts:
            if isinstance(part, TextPart):
                chunks.append({"type": "text", "text": part.text})
            elif isinstance(part, ReasoningPart):
                if part.redacted and not part.text:
                    continue
                chunks.append(self._reasoning_chunk(part))
            elif isinstance(part, AttachmentPart) and part.attachment:
                chunks.append(_attachment_chunk(part.attachment))
        if not chunks:
            return None
        if all(chunk["type"] == "text" for chunk in chunks):
            return "".join(chunk["text"] for chunk in chunks)
        return chunks

    def build_messages(self, prompt, conversation):
        messages = []
        for message in prompt.messages:
            if message.role == "tool":
                tool_results = [
                    part for part in message.parts if isinstance(part, ToolResultPart)
                ]
                for tool_result in tool_results:
                    content_parts = [TextPart(text=tool_result.output)] + [
                        AttachmentPart(attachment=attachment)
                        for attachment in tool_result.attachments
                    ]
                    tool_message = {
                        "role": "tool",
                        "content": self._content_from_parts(content_parts) or "",
                    }
                    if tool_result.tool_call_id is not None:
                        tool_message["tool_call_id"] = tool_result.tool_call_id
                    if tool_result.name:
                        tool_message["name"] = tool_result.name
                    messages.append(tool_message)
                if not tool_results:
                    messages.append(
                        {
                            "role": "tool",
                            "content": self._content_from_parts(message.parts) or "",
                        }
                    )
                continue

            provider_message = {
                "role": message.role,
                "content": self._content_from_parts(message.parts) or "",
            }
            if message.role == "assistant":
                tool_calls = []
                for part in message.parts:
                    if not isinstance(part, ToolCallPart):
                        continue
                    tool_call = {
                        "type": "function",
                        "function": {
                            "name": part.name,
                            "arguments": json.dumps(part.arguments),
                        },
                    }
                    if part.tool_call_id is not None:
                        tool_call["id"] = part.tool_call_id
                    tool_calls.append(tool_call)
                if tool_calls:
                    provider_message["tool_calls"] = tool_calls
                    if not provider_message["content"]:
                        provider_message["content"] = None
            messages.append(provider_message)

        if prompt.options.prefix:
            messages.append(
                {"role": "assistant", "content": prompt.options.prefix, "prefix": True}
            )
        return messages

    def build_kwargs(self, prompt, messages):
        kwargs = {
            "model": self.mistral_model_id,
            "messages": messages,
        }
        if getattr(prompt.options, "reasoning_effort", None) is not None:
            kwargs["reasoning_effort"] = prompt.options.reasoning_effort
        if prompt.options.temperature is not None:
            kwargs["temperature"] = prompt.options.temperature
        if prompt.options.top_p is not None:
            kwargs["top_p"] = prompt.options.top_p
        if prompt.options.max_tokens is not None:
            kwargs["max_tokens"] = prompt.options.max_tokens
        if prompt.options.safe_prompt is not None:
            kwargs["safe_prompt"] = prompt.options.safe_prompt
        if prompt.options.random_seed is not None:
            kwargs["random_seed"] = prompt.options.random_seed
        if prompt.schema:
            # Mistral complains if additionalProperties: False is missing
            schema = copy.deepcopy(prompt.schema)
            schema["additionalProperties"] = False
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "schema": schema,
                    "strict": True,
                    "name": "data",
                },
            }
        if prompt.tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                    },
                }
                for tool in prompt.tools
            ]
            kwargs["tool_choice"] = "auto"
        return kwargs

    def set_usage(self, response, usage):
        usage_dict = _sdk_dump(usage)
        details = {
            key: value
            for key, value in usage_dict.items()
            if key not in {"prompt_tokens", "completion_tokens", "total_tokens"}
            and value is not None
        }
        response.set_usage(
            input=usage_dict.get("prompt_tokens"),
            output=usage_dict.get("completion_tokens"),
            details=details or None,
        )

    def _reasoning_event(
        self, raw_chunk, message_index, part_index, chunk_override=None
    ):
        thinking_text = (
            "".join(
                item.get("text", "")
                for item in raw_chunk.get("thinking", [])
                if isinstance(item, dict) and item.get("type") == "text"
            )
            if chunk_override is None
            else chunk_override
        )
        metadata = {"mistral": {"content_chunk": copy.deepcopy(raw_chunk)}}
        if "signature" in raw_chunk:
            metadata["mistral"]["signature"] = raw_chunk["signature"]
        if "closed" in raw_chunk:
            metadata["mistral"]["closed"] = raw_chunk["closed"]
        return StreamEvent(
            type="reasoning",
            chunk=thinking_text,
            part_index=part_index,
            provider_metadata=metadata,
            message_index=message_index,
        )

    def content_events(self, content, part_indexes, message_index=0):
        if isinstance(content, str):
            if content:
                yield StreamEvent(
                    type="text",
                    chunk=content,
                    part_index=part_indexes.text(message_index),
                    message_index=message_index,
                )
            return
        if not isinstance(content, list):
            return
        for content_chunk in content:
            chunk_type = getattr(content_chunk, "type", None)
            if chunk_type == "text" and content_chunk.text:
                yield StreamEvent(
                    type="text",
                    chunk=content_chunk.text,
                    part_index=part_indexes.text(message_index),
                    message_index=message_index,
                )
            elif chunk_type == "thinking":
                yield self._reasoning_event(
                    _sdk_dump(content_chunk), message_index, next(part_indexes)
                )

    def _accumulate_reasoning(
        self, reasoning_states, content_chunk, message_index, part_indexes
    ):
        incoming = _sdk_dump(content_chunk)
        if message_index not in reasoning_states:
            reasoning_states[message_index] = {
                "chunk": {"type": "thinking", "thinking": []},
                "part_index": next(part_indexes),
            }
        state = reasoning_states[message_index]["chunk"]
        state["thinking"].extend(copy.deepcopy(incoming.get("thinking", [])))
        if "signature" in incoming:
            state["signature"] = incoming["signature"]
        if "closed" in incoming:
            state["closed"] = incoming["closed"]
        return incoming

    def flush_reasoning(self, reasoning_states, message_index=None):
        indexes = (
            list(reasoning_states)
            if message_index is None
            else ([message_index] if message_index in reasoning_states else [])
        )
        for index in indexes:
            state = reasoning_states.pop(index)
            yield self._reasoning_event(
                state["chunk"], index, state["part_index"], chunk_override=""
            )

    def stream_content_events(
        self, content, reasoning_states, part_indexes, message_index=0
    ):
        for index in list(reasoning_states):
            if index != message_index:
                yield from self.flush_reasoning(reasoning_states, index)
        if isinstance(content, str):
            if content:
                yield from self.flush_reasoning(reasoning_states, message_index)
                yield StreamEvent(
                    type="text",
                    chunk=content,
                    part_index=part_indexes.text(message_index),
                    message_index=message_index,
                )
            return
        if not isinstance(content, list):
            return
        for content_chunk in content:
            chunk_type = getattr(content_chunk, "type", None)
            if chunk_type == "thinking":
                incoming = self._accumulate_reasoning(
                    reasoning_states, content_chunk, message_index, part_indexes
                )
                thinking_text = "".join(
                    item.get("text", "")
                    for item in incoming.get("thinking", [])
                    if isinstance(item, dict) and item.get("type") == "text"
                )
                if thinking_text:
                    yield StreamEvent(
                        type="reasoning",
                        chunk=thinking_text,
                        part_index=reasoning_states[message_index]["part_index"],
                        message_index=message_index,
                    )
                # closed is a prefixing flag, repeated on ordinary text deltas.
                # A signature seals a block; preserve it separately for replay.
                if incoming.get("signature") is not None:
                    yield from self.flush_reasoning(reasoning_states, message_index)
            else:
                yield from self.flush_reasoning(reasoning_states, message_index)
                if chunk_type == "text" and content_chunk.text:
                    yield StreamEvent(
                        type="text",
                        chunk=content_chunk.text,
                        part_index=part_indexes.text(message_index),
                        message_index=message_index,
                    )

    def gather_tool_call_events(
        self, gathered, tool_calls, part_indexes, message_index=0
    ):
        if not isinstance(tool_calls, list):
            return
        for position, tool_call in enumerate(tool_calls):
            index = getattr(tool_call, "index", position)
            if not isinstance(index, int):
                index = position
            key = (message_index, index)
            entry = gathered.get(key)
            if entry is None:
                provider_id = getattr(tool_call, "id", None)
                if not provider_id or provider_id == "null":
                    provider_id = f"mistral-tool-{uuid.uuid4().hex}"
                entry = {
                    "id": provider_id,
                    "part_index": next(part_indexes),
                    "name": "",
                    "arguments": "",
                    "arguments_object": None,
                    "message_index": message_index,
                }
                gathered[key] = entry
            function = tool_call.function
            name = function.name
            name_fragment = ""
            if name:
                if not entry["name"]:
                    entry["name"] = name
                    name_fragment = name
                elif name.startswith(entry["name"]):
                    name_fragment = name[len(entry["name"]) :]
                    entry["name"] = name
                elif not entry["name"].endswith(name):
                    entry["name"] += name
                    name_fragment = name
            if name_fragment:
                yield StreamEvent(
                    type="tool_call_name",
                    chunk=name_fragment,
                    tool_call_id=entry["id"],
                    part_index=entry["part_index"],
                    message_index=message_index,
                )
            arguments = function.arguments
            if isinstance(arguments, dict):
                entry["arguments_object"] = arguments
                arguments_fragment = json.dumps(arguments)
            elif isinstance(arguments, str):
                entry["arguments"] += arguments
                arguments_fragment = arguments
            else:
                arguments_fragment = ""
            if arguments_fragment:
                yield StreamEvent(
                    type="tool_call_args",
                    chunk=arguments_fragment,
                    tool_call_id=entry["id"],
                    part_index=entry["part_index"],
                    message_index=message_index,
                )

    def finalize_tool_calls(self, response, gathered):
        for tool_call in gathered.values():
            if not tool_call["name"]:
                continue
            if tool_call["arguments_object"] is not None:
                arguments = tool_call["arguments_object"]
                arguments_json = json.dumps(arguments)
            else:
                arguments_json = tool_call["arguments"] or "{}"
                try:
                    arguments = json.loads(arguments_json)
                except json.JSONDecodeError as error:
                    raise llm.ModelError(
                        f"Invalid JSON arguments for tool {tool_call['name']}: {arguments_json}"
                    ) from error
            response.add_tool_call(
                llm.ToolCall(
                    name=tool_call["name"],
                    arguments=arguments,
                    tool_call_id=tool_call["id"],
                )
            )

    def completion_events(self, completion, response):
        gathered = {}
        part_indexes = _PartIndexes()
        if not completion.choices:
            return
        choice = completion.choices[0]
        provider_messages = []
        if choice.message is not None:
            provider_messages.append(choice.message)
        if choice.messages:
            provider_messages.extend(choice.messages)
        for position, message in enumerate(provider_messages):
            index = getattr(message, "index", position)
            if not isinstance(index, int):
                index = position
            yield from self.content_events(message.content, part_indexes, index)
            yield from self.gather_tool_call_events(
                gathered, message.tool_calls, part_indexes, index
            )
        self.finalize_tool_calls(response, gathered)


class Mistral(_Shared, llm.KeyModel):
    def execute(self, prompt, stream, response, conversation, key):
        messages = self.build_messages(prompt, conversation)
        response._prompt_json = {"messages": messages}
        kwargs = self.build_kwargs(prompt, messages)
        try:
            with MistralClient(api_key=key) as client:
                if stream:
                    chunks = []
                    gathered_tool_calls = {}
                    reasoning_states = {}
                    part_indexes = _PartIndexes()
                    usage = None
                    resolved_model = None
                    with client.chat.stream(**kwargs) as event_stream:
                        for sdk_event in event_stream:
                            chunk = sdk_event.data
                            chunks.append(_sdk_dump(chunk))
                            resolved_model = chunk.model or resolved_model
                            if chunk.usage is not None:
                                usage = chunk.usage
                            for choice in chunk.choices:
                                delta = choice.delta
                                message_index = getattr(delta, "index", choice.index)
                                if not isinstance(message_index, int):
                                    message_index = choice.index
                                yield from self.stream_content_events(
                                    delta.content,
                                    reasoning_states,
                                    part_indexes,
                                    message_index,
                                )
                                if (
                                    isinstance(delta.tool_calls, list)
                                    and delta.tool_calls
                                ):
                                    yield from self.flush_reasoning(reasoning_states)
                                yield from self.gather_tool_call_events(
                                    gathered_tool_calls,
                                    delta.tool_calls,
                                    part_indexes,
                                    message_index,
                                )
                    yield from self.flush_reasoning(reasoning_states)
                    self.finalize_tool_calls(response, gathered_tool_calls)
                    response.response_json = {"chunks": chunks}
                    if resolved_model:
                        response.set_resolved_model(resolved_model)
                    if usage is not None:
                        self.set_usage(response, usage)
                else:
                    completion = client.chat.complete(**kwargs)
                    response.response_json = _sdk_dump(completion)
                    response.set_resolved_model(completion.model)
                    yield from self.completion_events(completion, response)
                    self.set_usage(response, completion.usage)
        except MistralError as error:
            raise _model_error(error) from error


class AsyncMistral(_Shared, llm.AsyncKeyModel):
    async def execute(self, prompt, stream, response, conversation, key):
        messages = self.build_messages(prompt, conversation)
        response._prompt_json = {"messages": messages}
        kwargs = self.build_kwargs(prompt, messages)
        try:
            async with MistralClient(api_key=key) as client:
                if stream:
                    chunks = []
                    gathered_tool_calls = {}
                    reasoning_states = {}
                    part_indexes = _PartIndexes()
                    usage = None
                    resolved_model = None
                    event_stream = await client.chat.stream_async(**kwargs)
                    async with event_stream:
                        async for sdk_event in event_stream:
                            chunk = sdk_event.data
                            chunks.append(_sdk_dump(chunk))
                            resolved_model = chunk.model or resolved_model
                            if chunk.usage is not None:
                                usage = chunk.usage
                            for choice in chunk.choices:
                                delta = choice.delta
                                message_index = getattr(delta, "index", choice.index)
                                if not isinstance(message_index, int):
                                    message_index = choice.index
                                for event in self.stream_content_events(
                                    delta.content,
                                    reasoning_states,
                                    part_indexes,
                                    message_index,
                                ):
                                    yield event
                                if (
                                    isinstance(delta.tool_calls, list)
                                    and delta.tool_calls
                                ):
                                    for event in self.flush_reasoning(reasoning_states):
                                        yield event
                                for event in self.gather_tool_call_events(
                                    gathered_tool_calls,
                                    delta.tool_calls,
                                    part_indexes,
                                    message_index,
                                ):
                                    yield event
                    for event in self.flush_reasoning(reasoning_states):
                        yield event
                    self.finalize_tool_calls(response, gathered_tool_calls)
                    response.response_json = {"chunks": chunks}
                    if resolved_model:
                        response.set_resolved_model(resolved_model)
                    if usage is not None:
                        self.set_usage(response, usage)
                else:
                    completion = await client.chat.complete_async(**kwargs)
                    response.response_json = _sdk_dump(completion)
                    response.set_resolved_model(completion.model)
                    for event in self.completion_events(completion, response):
                        yield event
                    self.set_usage(response, completion.usage)
        except MistralError as error:
            raise _model_error(error) from error


class MistralEmbed(llm.EmbeddingModel):
    batch_size = 10
    needs_key = "mistral"
    key_env_var = "LLM_MISTRAL_KEY"

    def __init__(self, model_id, model_name, output_dimension=None):
        self.model_id = model_id
        self.model_name = model_name
        self.output_dimension = output_dimension

    def embed_batch(self, texts):
        key = self.get_key()
        kwargs = {
            "model": self.model_name,
            "inputs": list(texts),
        }
        if self.output_dimension is not None:
            kwargs["output_dimension"] = self.output_dimension
        try:
            with MistralClient(api_key=key) as client:
                api_response = client.embeddings.create(**kwargs)
        except MistralError as error:
            raise _model_error(error) from error
        return [item.embedding for item in api_response.data]
