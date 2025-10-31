"""Launch a Gradio chat powered by the Biomni A1 agent using a local Ollama model."""

from __future__ import annotations

import os
from typing import Iterable, Sequence

import gradio as gr

from biomni.agent import A1
from biomni.config import default_config

DATA_ROOT = "./biomni_full"
OLLAMA_MODEL = "gpt-oss:20b"

# Configure Biomni defaults for the local environment.
default_config.path = DATA_ROOT
default_config.llm = OLLAMA_MODEL
default_config.source = "Ollama"
default_config.use_tool_retriever = False

# Instantiate once so the Gradio app reuses the agent across turns.
AGENT = A1(
    path=DATA_ROOT,
    llm=OLLAMA_MODEL,
    source="Ollama",
    use_tool_retriever=False,
)


def _iter_history(history: Sequence) -> Iterable[tuple[str, str]]:
    """Yield (role, content) pairs from Gradio chat history in either format."""
    for item in history:
        if isinstance(item, dict):
            role = item.get("role", "assistant")
            content = item.get("content", "") or ""
            yield role, content
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            user_msg, bot_msg = item
            yield "user", user_msg or ""
            yield "assistant", bot_msg or ""


def _build_prompt(message: str, history: Sequence) -> str:
    """Construct a prompt that includes minimal conversation context and instructions."""
    dialogue_lines = []
    for role, content in _iter_history(history):
        role_pretty = "User" if role == "user" else "Assistant"
        dialogue_lines.append(f"{role_pretty}: {content}")
    context = "\n".join(dialogue_lines)
    if context:
        context = "\n\nPrior conversation:\n" + context

    instructions = (
        "Respond with your reasoning, then wrap the final summary in <solution> tags. "
        "Stay grounded in biomedical knowledge and cite pathways or markers when helpful."
    )
    latest = f"\n\nLatest user request:\n{message}"
    return f"{instructions}{context}{latest}"


def respond(message: str, history: Sequence) -> str:
    prompt = _build_prompt(message, history)
    _log, answer = AGENT.go(prompt)
    return answer


def _resolve_launch_args() -> tuple[str, int | None]:
    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    port_env = os.getenv("GRADIO_SERVER_PORT")
    port_value: int | None = None
    if port_env:
        try:
            port_value = int(port_env)
        except ValueError:
            print(f"[gradio_app] Warning: invalid GRADIO_SERVER_PORT='{port_env}', falling back to auto port.")
            port_value = None
    return server_name, port_value


demo = gr.ChatInterface(
    respond,
    title="Biomni (Local Ollama gpt-oss:20b)",
    description=(
        "Chat with Biomni's A1 agent using your local Ollama model. Each reply includes an explicit "
        "reasoning section followed by a <solution> summary."
    ),
    type="messages",
)


if __name__ == "__main__":
    hostname, port = _resolve_launch_args()
    try:
        demo.launch(server_name=hostname, server_port=port, share=False)
    except OSError as exc:
        if port is not None:
            print(
                f"[gradio_app] Port {port} unavailable ({exc}). Falling back to automatic port selection."
            )
            demo.launch(server_name=hostname, server_port=0, share=False)
        else:
            raise
