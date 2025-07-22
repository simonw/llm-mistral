import click
from httpx_sse import connect_sse, aconnect_sse
import httpx
import json
import llm
from pydantic import Field
from typing import Optional


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
        vision = model.get("capabilities", {}).get("vision")
        audio = "voxtral" in model_id
        our_model_id = "mistral/" + model_id
        alias = DEFAULT_ALIASES.get(our_model_id)
        aliases = [alias] if alias else []
        schemas = "codestral-mamba" not in model_id
        tools = our_model_id in tool_models
        register(
            Mistral(our_model_id, model_id, vision, schemas, tools, audio),
            AsyncMistral(our_model_id, model_id, vision, schemas, tools, audio),
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


def refresh_models():
    user_dir = llm.user_dir()
    mistral_models = user_dir / "mistral_models.json"
    key = llm.get_key("", "mistral", "LLM_MISTRAL_KEY")
    if not key:
        raise click.ClickException(
            "You must set the 'mistral' key or the LLM_MISTRAL_KEY environment variable."
        )
    response = httpx.get(
        "https://api.mistral.ai/v1/models", headers={"Authorization": f"Bearer {key}"}
    )
    response.raise_for_status()
    models = response.json()
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
        except httpx.HTTPStatusError:
            pass
    return [model for model in models["data"] if "embed" not in model["id"]]


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

    @mistral.command()
    @click.argument("input_path")
    @click.option(
        "-m",
        "--model",
        default="voxtral-mini",
        type=click.Choice(["voxtral-mini", "voxtral-small"]),
        help="Model to use for transcription (default: voxtral-mini)",
    )
    @click.option(
        "-s",
        "--segments",
        is_flag=True,
        help="Include segment-level timestamps in the transcription",
    )
    @click.option("--key", help="API key to use")
    def transcribe(input_path, model, segments, key):
        "Transcribe audio from URL or local file path"
        api_key = llm.get_key(key, "mistral", "LLM_MISTRAL_KEY")
        if not api_key:
            raise click.ClickException(
                "You must use --key, set the 'mistral' key or set the LLM_MISTRAL_KEY environment variable."
            )

        # Convert model name to full model ID
        model_id = f"{model}-2507"

        # Determine if input is URL or file path
        is_url = input_path.startswith(("http://", "https://"))

        # Prepare multipart form data using files parameter for proper multipart/form-data
        files = {"model": (None, model_id)}

        if segments:
            files["timestamp_granularities"] = (None, "segment")

        if is_url:
            files["file_url"] = (None, input_path)
        else:
            # For file uploads
            try:
                files["file"] = open(input_path, "rb")
            except FileNotFoundError:
                raise click.ClickException(f"File not found: {input_path}")
            except Exception as e:
                raise click.ClickException(f"Error opening file: {e}")

        try:
            with httpx.Client() as client:
                response = client.post(
                    "https://api.mistral.ai/v1/audio/transcriptions",
                    headers={"x-api-key": api_key},
                    files=files,
                    timeout=None,
                )
                response.raise_for_status()
                result = response.json()
                click.echo(json.dumps(result, indent=2))
        except httpx.HTTPStatusError as e:
            try:
                error_details = e.response.json()
                raise click.ClickException(
                    f"API Error {e.response.status_code}: {error_details}"
                )
            except:
                raise click.ClickException(
                    f"API Error {e.response.status_code}: {e.response.text}"
                )
        except Exception as e:
            raise click.ClickException(f"Error: {e}")
        finally:
            # Close file if it was opened for file uploads
            if not is_url and "file" in files:
                files["file"].close()


def _build_message_content(text, attachments):
    """Build message content handling attachments consistently.

    Args:
        text: The text content for the message
        attachments: List of attachment objects or None

    Returns:
        String if no attachments, or array with text + attachment objects
    """
    if not attachments:
        return text

    content = [{"type": "text", "text": text or ""}]
    for attachment in attachments:
        # audio only handles URLs at the moment
        if attachment.resolve_type() == "audio/mpeg":
            if not attachment.url:
                raise ValueError("Audio attachment must use a URL")
            content.append(
                {
                    "type": "input_audio",
                    "input_audio": {"data": attachment.url, "format": "mp3"},
                }
            )
        else:
            # Images
            content.append(
                {
                    "type": "image_url",
                    "image_url": attachment.url
                    or f"data:{attachment.resolve_type()};base64,{attachment.base64_content()}",
                }
            )
    return content


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
        safe_mode: Optional[bool] = Field(
            description="Whether to inject a safety prompt before all conversations.",
            default=False,
        )
        random_seed: Optional[int] = Field(
            description="Sets the seed for random sampling to generate deterministic results.",
            default=None,
        )
        prefix: Optional[str] = Field(
            description="A prefix to prepend to the response.",
            default=None,
        )

    def __init__(self, our_model_id, mistral_model_id, vision, schemas, tools, audio):
        self.model_id = our_model_id
        self.mistral_model_id = mistral_model_id
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

    def build_messages(self, prompt, conversation):
        messages = []

        # If no conversation history, build initial messages
        if not conversation:
            if prompt.system:
                messages.append({"role": "system", "content": prompt.system})

            # Add user message if we have content and no tool results
            if not prompt.tool_results and prompt.prompt is not None:
                messages.append(
                    {
                        "role": "user",
                        "content": _build_message_content(
                            prompt.prompt, prompt.attachments
                        ),
                    }
                )

            # Add tool results if present
            if prompt.tool_results:
                for tool_result in prompt.tool_results:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_result.tool_call_id,
                            "content": json.dumps(tool_result.output),
                        }
                    )

            # Add prefix if specified
            if prompt.options.prefix:
                messages.append(
                    {
                        "role": "assistant",
                        "content": prompt.options.prefix,
                        "prefix": True,
                    }
                )

            return messages

        # Process conversation history
        current_system = None
        for i, prev_response in enumerate(conversation.responses):
            # Add system message if changed
            if (
                prev_response.prompt.system
                and prev_response.prompt.system != current_system
            ):
                messages.append(
                    {"role": "system", "content": prev_response.prompt.system}
                )
                current_system = prev_response.prompt.system

            # Add user message only if not a tool result response
            if not prev_response.prompt.tool_results:
                if (
                    prev_response.prompt.prompt is not None
                ):  # Only add if there's content
                    messages.append(
                        {
                            "role": "user",
                            "content": _build_message_content(
                                prev_response.prompt.prompt, prev_response.attachments
                            ),
                        }
                    )

            # If this response's prompt had tool results, add them before the assistant message
            if prev_response.prompt.tool_results:
                for tool_result in prev_response.prompt.tool_results:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_result.tool_call_id,
                            "content": json.dumps(tool_result.output),
                        }
                    )

            # Add assistant response
            assistant_message = {"role": "assistant"}

            # Check if response contains tool calls
            tool_calls = prev_response.tool_calls_or_raise()
            if tool_calls:
                # If there are tool calls, format them according to Mistral spec
                assistant_message["content"] = None
                assistant_message["tool_calls"] = [
                    {
                        "id": tool_call.tool_call_id,
                        "type": "function",
                        "function": {
                            "name": tool_call.name,
                            "arguments": json.dumps(tool_call.arguments),
                        },
                    }
                    for tool_call in tool_calls
                ]
            else:
                # Regular text response
                assistant_message["content"] = prev_response.text_or_raise()

            messages.append(assistant_message)

        # Add system message for current prompt if different
        if prompt.system and prompt.system != current_system:
            messages.append({"role": "system", "content": prompt.system})

        # Add current user message if not a tool result response
        if not prompt.tool_results and prompt.prompt is not None:
            messages.append(
                {
                    "role": "user",
                    "content": _build_message_content(
                        prompt.prompt, prompt.attachments
                    ),
                }
            )

        # Add current tool results if present
        if prompt.tool_results:
            for tool_result in prompt.tool_results:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_result.tool_call_id,
                        "content": json.dumps(tool_result.output),
                    }
                )

        # Add prefix if specified
        if prompt.options.prefix:
            messages.append(
                {"role": "assistant", "content": prompt.options.prefix, "prefix": True}
            )

        return messages

    def build_body(self, prompt, messages):
        body = {
            "model": self.mistral_model_id,
            "messages": messages,
        }
        if prompt.options.temperature:
            body["temperature"] = prompt.options.temperature
        if prompt.options.top_p:
            body["top_p"] = prompt.options.top_p
        if prompt.options.max_tokens:
            body["max_tokens"] = prompt.options.max_tokens
        if prompt.options.safe_mode:
            body["safe_mode"] = prompt.options.safe_mode
        if prompt.options.random_seed:
            body["random_seed"] = prompt.options.random_seed
        if prompt.schema:
            # Mistral complains if additionalProperties: False is missing
            prompt.schema["additionalProperties"] = False
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "schema": prompt.schema,
                    "strict": True,
                    "name": "data",
                },
            }
        if prompt.tools:
            body["tools"] = [
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
            body["tool_choice"] = "auto"
        return body

    def set_usage(self, response, usage):
        response.set_usage(
            input=usage["prompt_tokens"],
            output=usage["completion_tokens"],
        )

    def extract_tool_calls(self, response, thing_with_tool_calls):
        if thing_with_tool_calls.get("tool_calls"):
            for tool_call in thing_with_tool_calls["tool_calls"]:
                response.add_tool_call(
                    llm.ToolCall(
                        name=tool_call["function"]["name"],
                        arguments=json.loads(tool_call["function"]["arguments"]),
                        tool_call_id=tool_call["id"],
                    )
                )


class Mistral(_Shared, llm.KeyModel):
    def execute(self, prompt, stream, response, conversation, key):
        messages = self.build_messages(prompt, conversation)
        response._prompt_json = {"messages": messages}
        body = self.build_body(prompt, messages)
        if stream:
            body["stream"] = True
            with httpx.Client() as client:
                with connect_sse(
                    client,
                    "POST",
                    "https://api.mistral.ai/v1/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                        "Authorization": f"Bearer {key}",
                    },
                    json=body,
                    timeout=None,
                ) as event_source:
                    # In case of unauthorized:
                    if event_source.response.status_code != 200:
                        # Try to make this a readable error, it may have a base64 chunk
                        try:
                            decoded = json.loads(event_source.response.read())
                            type = decoded.get("type", "UnknownType")
                            words = decoded["message"].split()
                        except (json.JSONDecodeError, KeyError):
                            click.echo(
                                event_source.response.read().decode()[:200], err=True
                            )
                            event_source.response.raise_for_status()
                        # Truncate any words longer than 30 characters
                        words = [word[:30] for word in words]
                        message = " ".join(words)
                        raise llm.ModelError(
                            f"{event_source.response.status_code}: {type} - {message}"
                        )
                    usage = None
                    event_source.response.raise_for_status()
                    for sse in event_source.iter_sse():
                        if sse.data != "[DONE]":
                            try:
                                event = sse.json()
                                if "usage" in event:
                                    usage = event["usage"]
                                delta = event["choices"][0]["delta"]
                                self.extract_tool_calls(response, delta)
                                if "content" in delta:
                                    yield delta["content"]
                            except KeyError:
                                pass
                    if usage:
                        self.set_usage(response, usage)
        else:
            with httpx.Client() as client:
                api_response = client.post(
                    "https://api.mistral.ai/v1/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                        "Authorization": f"Bearer {key}",
                    },
                    json=body,
                    timeout=None,
                )
                api_response.raise_for_status()
                yield api_response.json()["choices"][0]["message"]["content"]
                details = api_response.json()
                usage = details.pop("usage", None)
                self.extract_tool_calls(response, details["choices"][0]["message"])
                response.response_json = details
                if usage:
                    self.set_usage(response, usage)

    def build_body(self, prompt, messages):
        body = {
            "model": self.mistral_model_id,
            "messages": messages,
        }
        if prompt.options.temperature:
            body["temperature"] = prompt.options.temperature
        if prompt.options.top_p:
            body["top_p"] = prompt.options.top_p
        if prompt.options.max_tokens:
            body["max_tokens"] = prompt.options.max_tokens
        if prompt.options.safe_mode:
            body["safe_mode"] = prompt.options.safe_mode
        if prompt.options.random_seed:
            body["random_seed"] = prompt.options.random_seed
        if prompt.schema:
            # Mistral complains if additionalProperties: False is missing
            prompt.schema["additionalProperties"] = False
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "schema": prompt.schema,
                    "strict": True,
                    "name": "data",
                },
            }
        if prompt.tools:
            body["tools"] = [
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
            body["tool_choice"] = "auto"
        return body


class AsyncMistral(_Shared, llm.AsyncKeyModel):
    async def execute(self, prompt, stream, response, conversation, key):
        messages = self.build_messages(prompt, conversation)
        response._prompt_json = {"messages": messages}
        body = self.build_body(prompt, messages)
        if stream:
            body["stream"] = True
            async with httpx.AsyncClient() as client:
                async with aconnect_sse(
                    client,
                    "POST",
                    "https://api.mistral.ai/v1/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                        "Authorization": f"Bearer {key}",
                    },
                    json=body,
                    timeout=None,
                ) as event_source:
                    # In case of unauthorized:
                    if event_source.response.status_code != 200:
                        # Try to make this a readable error, it may have a base64 chunk
                        try:
                            decoded = json.loads(await event_source.response.aread())
                            type = decoded.get("type", "UnknownType")
                            words = decoded["message"].split()
                        except (json.JSONDecodeError, KeyError):
                            click.echo(
                                (await event_source.response.aread()).decode()[:200],
                                err=True,
                            )
                            event_source.response.raise_for_status()
                        # Truncate any words longer than 30 characters
                        words = [word[:30] for word in words]
                        message = " ".join(words)
                        raise llm.ModelError(
                            f"{event_source.response.status_code}: {type} - {message}"
                        )
                    event_source.response.raise_for_status()
                    usage = None
                    async for sse in event_source.aiter_sse():
                        if sse.data != "[DONE]":
                            try:
                                event = sse.json()
                                if "usage" in event:
                                    usage = event["usage"]
                                delta = event["choices"][0]["delta"]
                                self.extract_tool_calls(response, delta)
                                if "content" in delta:
                                    yield delta["content"]
                            except KeyError:
                                pass
                    if usage:
                        self.set_usage(response, usage)
        else:
            async with httpx.AsyncClient() as client:
                api_response = await client.post(
                    "https://api.mistral.ai/v1/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                        "Authorization": f"Bearer {key}",
                    },
                    json=body,
                    timeout=None,
                )
                api_response.raise_for_status()
                message = api_response.json()["choices"][0]["message"]
                self.extract_tool_calls(response, message)
                yield message["content"]
                details = api_response.json()
                usage = details.pop("usage", None)
                response.response_json = details
                if usage:
                    self.set_usage(response, usage)


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
        body = {
            "model": self.model_name,
            "input": list(texts),
        }
        if self.output_dimension:
            body["output_dimension"] = self.output_dimension
        with httpx.Client() as client:
            api_response = client.post(
                "https://api.mistral.ai/v1/embeddings",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Authorization": f"Bearer {key}",
                },
                json=body,
                timeout=None,
            )
            api_response.raise_for_status()
            return [item["embedding"] for item in api_response.json()["data"]]
