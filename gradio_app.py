"""Launch a Gradio chat powered by the Biomni A1 agent using a local Ollama model."""

from __future__ import annotations

from typing import List, Tuple

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


def _build_prompt(message: str, history: List[Tuple[str, str]]) -> str:
    """Construct a prompt that includes minimal conversation context and format instructions."""
    dialogue = []
    for user_msg, bot_msg in history:
        dialogue.append(f"User: {user_msg}")
        dialogue.append(f"Assistant: {bot_msg}")
    context = "\n".join(dialogue)
    if context:
        context = "\n\nPrior conversation:\n" + context

    instructions = (
        "Respond with your reasoning, then wrap the final summary in <solution> tags. "
        "Stay grounded in biomedical knowledge and cite pathways or markers when helpful."
    )
    latest = f"\n\nLatest user request:\n{message}"
    return f"{instructions}{context}{latest}"


def respond(message: str, history: List[Tuple[str, str]]) -> str:
    prompt = _build_prompt(message, history)
    _log, answer = AGENT.go(prompt)
    return answer


demo = gr.ChatInterface(
    respond,
    title="Biomni (Local Ollama gpt-oss:20b)",
    description=(
        "Chat with Biomni's A1 agent using your local Ollama model. Each reply includes an explicit "
        "reasoning section followed by a <solution> summary."
    ),
)


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
