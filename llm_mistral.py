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
    "mistral/mistral-small": "mistral-small",
    "mistral/mistral-medium": "mistral-medium",
    "mistral/mistral-large-latest": "mistral-large",
    "mistral/codestral-mamba-latest": "codestral-mamba",
    "mistral/codestral-latest": "codestral",
    "mistral/ministral-3b-latest": "ministral-3b",
    "mistral/ministral-8b-latest": "ministral-8b",
    "mistral/pixtral-12b-latest": "pixtral-12b",
    "mistral/pixtral-large-latest": "pixtral-large",
}


@llm.hookimpl
def register_models(register):
    for model in get_model_details():
        model_id = model["id"]
        vision = model.get("capabilities", {}).get("vision")
        our_model_id = "mistral/" + model_id
        alias = DEFAULT_ALIASES.get(our_model_id)
        aliases = [alias] if alias else []
        register(
            Mistral(our_model_id, model_id, vision),
            AsyncMistral(our_model_id, model_id, vision),
            aliases=aliases,
        )


@llm.hookimpl
def register_embedding_models(register):
    register(MistralEmbed())


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

    def __init__(self, our_model_id, mistral_model_id, vision):
        self.model_id = our_model_id
        self.mistral_model_id = mistral_model_id
        if vision:
            self.attachment_types = {
                "image/jpeg",
                "image/png",
                "image/gif",
                "image/webp",
            }

    def build_messages(self, prompt, conversation):
        messages = []
        latest_message = None
        if prompt.attachments:
            latest_message = {
                "role": "user",
                "content": [{"type": "text", "text": prompt.prompt}]
                + [
                    {
                        "type": "image_url",
                        "image_url": attachment.url
                        or f"data:{attachment.resolve_type()};base64,{attachment.base64_content()}",
                    }
                    for attachment in prompt.attachments
                ],
            }
        else:
            latest_message = {"role": "user", "content": prompt.prompt}
        if not conversation:
            if prompt.system:
                messages.append({"role": "system", "content": prompt.system})
            messages.append(latest_message)
            return messages

        current_system = None
        for prev_response in conversation.responses:
            if (
                prev_response.prompt.system
                and prev_response.prompt.system != current_system
            ):
                messages.append(
                    {"role": "system", "content": prev_response.prompt.system}
                )
                current_system = prev_response.prompt.system
            if prev_response.attachments:
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prev_response.prompt.prompt,
                            }
                        ]
                        + [
                            {
                                "type": "image_url",
                                "image_url": attachment.url
                                or f"data:{attachment.resolve_type()};base64,{attachment.base64_content()}",
                            }
                            for attachment in prev_response.attachments
                        ],
                    }
                )
            else:
                messages.append(
                    {"role": "user", "content": prev_response.prompt.prompt}
                )
            messages.append(
                {"role": "assistant", "content": prev_response.text_or_raise()}
            )
        if prompt.system and prompt.system != current_system:
            messages.append({"role": "system", "content": prompt.system})

        messages.append(latest_message)
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
        return body

    def set_usage(self, response, usage):
        response.set_usage(
            input=usage["prompt_tokens"],
            output=usage["completion_tokens"],
        )


class Mistral(_Shared, llm.Model):
    def execute(self, prompt, stream, response, conversation):
        key = self.get_key()
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
                            type = decoded["type"]
                            words = decoded["message"].split()
                        except (json.JSONDecodeError, KeyError):
                            click.echo(
                                event_source.response.read().decode()[:200], err=True
                            )
                            event_source.response.raise_for_status()
                        # Truncate any words longer than 30 characters
                        words = [word[:30] for word in words]
                        message = " ".join(words)
                        raise click.ClickException(
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
                                yield event["choices"][0]["delta"]["content"]
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
                response.response_json = details
                if usage:
                    self.set_usage(response, usage)


class AsyncMistral(_Shared, llm.AsyncModel):
    async def execute(self, prompt, stream, response, conversation):
        key = self.get_key()
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
                            decoded = json.loads(event_source.response.read())
                            type = decoded["type"]
                            words = decoded["message"].split()
                        except (json.JSONDecodeError, KeyError):
                            click.echo(
                                event_source.response.read().decode()[:200], err=True
                            )
                            event_source.response.raise_for_status()
                        # Truncate any words longer than 30 characters
                        words = [word[:30] for word in words]
                        message = " ".join(words)
                        raise click.ClickException(
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
                                yield event["choices"][0]["delta"]["content"]
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
                yield api_response.json()["choices"][0]["message"]["content"]
                details = api_response.json()
                usage = details.pop("usage", None)
                response.response_json = details
                if usage:
                    self.set_usage(response, usage)


class MistralEmbed(llm.EmbeddingModel):
    model_id = "mistral-embed"
    batch_size = 10
    needs_key = "mistral"
    key_env_var = "LLM_MISTRAL_KEY"

    def embed_batch(self, texts):
        key = self.get_key()
        with httpx.Client() as client:
            api_response = client.post(
                "https://api.mistral.ai/v1/embeddings",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Authorization": f"Bearer {key}",
                },
                json={
                    "model": "mistral-embed",
                    "input": list(texts),
                    "encoding_format": "float",
                },
                timeout=None,
            )
            api_response.raise_for_status()
            return [item["embedding"] for item in api_response.json()["data"]]
