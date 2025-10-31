"""
Minimal example showing how to run Biomni's A1 agent with a local Ollama model
while keeping all storage on the local filesystem (no AWS S3 downloads).

Steps performed:
1. Configure default Biomni settings to point at a local data directory.
2. Pre-create the directory structure expected by Biomni so the agent skips S3 sync.
3. Switch the LLM provider to Ollama and pick an installed model name.
4. Run a tiny prompt through the agent to verify the wiring.

Make sure an Ollama server is running locally (e.g. `ollama serve`) and that the
chosen model is pulled (`ollama pull mistral`) before executing this script.
"""

from pathlib import Path

from biomni.agent import A1
from biomni.config import default_config

LOCAL_DATA_ROOT = Path("./local_biomni")
OLLAMA_MODEL_NAME = "mistral"  # Change to any model available via `ollama list`


def prepare_local_storage(root: Path) -> None:
    """Create the minimal folder structure Biomni expects so no AWS downloads run."""
    benchmark_dir = root / "biomni_data" / "benchmark" / "hle"
    data_lake_dir = root / "biomni_data" / "data_lake"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    data_lake_dir.mkdir(parents=True, exist_ok=True)


def configure_for_ollama(root: Path, model_name: str) -> None:
    """Point global defaults at the local storage and Ollama model."""
    default_config.path = str(root)
    default_config.source = "Ollama"
    default_config.llm = model_name
    default_config.use_tool_retriever = False  # Speeds up the demo
    # No API keys required when using Ollama


def run_demo() -> None:
    """Instantiate the agent and run a quick prompt."""
    prepare_local_storage(LOCAL_DATA_ROOT)
    configure_for_ollama(LOCAL_DATA_ROOT, OLLAMA_MODEL_NAME)

    agent = A1(
        timeout_seconds=90,
        use_tool_retriever=False,
        expected_data_lake_files=[],  # Ensures no S3 lookups
    )

    prompt = "Give me a two-sentence fun fact about CRISPR."
    history, final_reply = agent.go(prompt)

    print("=== Conversation Trace (last step) ===")
    print(history[-1] if history else "No history recorded.")
    print("\n=== Final Reply ===")
    print(final_reply)


if __name__ == "__main__":
    try:
        run_demo()
    except Exception as exc:
        print("Demo failed. Common causes include:")
        print("- Ollama server not running locally.")
        print("- The selected model has not been pulled in Ollama.")
        print("- Additional Python dependencies required by a specific tool.")
        print(f"\nUnderlying error: {exc}")
