from collections.abc import Mapping, Sequence

from ollama import Client


class OllamaLLM:
    """Wrapper around the Ollama chat API."""

    def __init__(
        self,
        model: str = "qwen3:4b",
        host: str = "http://localhost:11434",
    ) -> None:
        self.model = model
        self.host = host

        self.client = Client(
            host=self.host,
        )

    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
    ) -> str:
        """Generate an LLM response from chat messages."""
        if not messages:
            raise ValueError("messages cannot be empty")

        response = self.client.chat(
            model=self.model,
            messages=[
                dict(message)
                for message in messages
            ],
        )

        content = response.message.content

        if not content:
            raise RuntimeError(
                f"Ollama model '{self.model}' returned an empty response"
            )

        return content.strip()