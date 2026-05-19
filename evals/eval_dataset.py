"""Small fixed RAG evaluation dataset for the default indexed article."""

from __future__ import annotations


RAG_EVAL_CASES = [
    {
        "question": "What is task decomposition in LLM-powered agents?",
        "reference": (
            "Task decomposition breaks a complex task into smaller, manageable subgoals or steps "
            "that an agent can solve more reliably."
        ),
    },
    {
        "question": "What are common ways to perform task decomposition?",
        "reference": (
            "Task decomposition can be done by prompting the LLM directly, using task-specific "
            "instructions, or with human input."
        ),
    },
    {
        "question": "How does the article describe reflection for autonomous agents?",
        "reference": (
            "Reflection lets agents review past actions, identify mistakes, and improve future plans "
            "through self-criticism or feedback."
        ),
    },
    {
        "question": "What role does memory play in LLM-powered agents?",
        "reference": (
            "Memory helps agents retain short-term context and long-term knowledge so they can use "
            "past information during planning and execution."
        ),
    },
    {
        "question": "What is the purpose of tool use in LLM-powered agents?",
        "reference": (
            "Tool use lets agents call external APIs, search, code interpreters, or other systems to "
            "access capabilities and information beyond the base model."
        ),
    },
]
