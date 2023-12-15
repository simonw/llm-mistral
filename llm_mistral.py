import httpx
from httpx_sse import connect_sse
import llm



@llm.hookimpl
def register_models(register):
    register(Mistral("mistral-tiny"))
    register(Mistral("mistral-small"))
    register(Mistral("mistral-medium"))


class Mistral(llm.Model):
    can_stream = True

    def __init__(self, model_id):
        self.model_id = model_id

    def execute(self, prompt, stream, response, conversation):
        key = llm.get_key("", "mistral", "LLM_MISTRAL_KEY")
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
                json={
                    "model": self.model_id,
                    "messages": [
                        {"role": "user", "content": prompt.prompt}
                    ],
                    "stream": True,
                },
            ) as event_source:
                # In case of unauthorized:
                event_source.response.raise_for_status()
                for sse in event_source.iter_sse():
                    if sse.data != '[DONE]':
                        try:
                            yield sse.json()["choices"][0]["delta"]["content"]
                        except KeyError:
                            pass
