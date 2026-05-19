"""Minimal Groq model invocation using LangChain.

Run:
    python model_invoke.py "Explain tool calling in one paragraph."
"""

from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq


DEFAULT_MODEL = "llama-3.1-8b-instant"


def build_model() -> ChatGroq:
    """Create the Groq chat model from environment configuration."""
    load_dotenv()

    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to a .env file or export it in your shell."
        )

    return ChatGroq(
        model=os.getenv("GROQ_MODEL", DEFAULT_MODEL),
        temperature=0,
        max_retries=2,
        timeout=30,
    )


def invoke_model(prompt: str) -> str:
    """Invoke the model and return the text response."""
    model = build_model()
    response = model.invoke(prompt)
    return response.content


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Invoke a Groq model through LangChain.")
    parser.add_argument(
        "prompt",
        nargs="?",
        default="Explain what a LangChain tool-calling agent does in 3 sentences.",
        help="Prompt to send to the model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(invoke_model(args.prompt))


if __name__ == "__main__":
    main()
