"""Convenience wrapper to exercise Biomni with a local Ollama model."""

import argparse

from biomni.agent.a1 import A1
from biomni.config import default_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Biomni A1 agent with the local Ollama gpt-oss:20b model."
    )
    parser.add_argument(
        "--prompt",
        default=(
            "Respond with your reasoning, then wrap the final summary in <solution> tags.\n"
            "Summarize the main regulatory checkpoints of the eukaryotic cell cycle."
        ),
        help="Task prompt to send to the agent.",
    )
    args = parser.parse_args()

    data_root = "./biomni_full"

    default_config.path = data_root
    default_config.llm = "gpt-oss:20b"
    default_config.source = "Ollama"
    default_config.use_tool_retriever = False

    agent = A1(
        path=data_root,
        llm="gpt-oss:20b",
        source="Ollama",
        use_tool_retriever=False,
    )

    print(f"Using prompt: {args.prompt}")
    log, answer = agent.go(args.prompt)

    print("---- Agent transcript ----")
    for entry in log:
        print(entry)
    print("---- Final answer ----")
    print(answer)


if __name__ == "__main__":
    main()
