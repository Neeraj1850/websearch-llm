"""Diverse scorecard queries for the tool agent and RAG agent."""

from __future__ import annotations


TOOL_AGENT_CASES = [
    {
        "id": "memory_update",
        "question": "My name is Neeraj and I am testing this agent.",
        "category": "memory",
        "expected_tool": "no_tool",
        "must_contain_any": ["Neeraj", "testing"],
    },
    {
        "id": "memory_followup",
        "question": "What is my name?",
        "category": "memory",
        "expected_tool": "no_tool",
        "must_contain_any": ["Neeraj"],
    },
    {
        "id": "factual_wiki",
        "question": "Who is Ada Lovelace?",
        "category": "factual",
        "expected_tool": "wikipedia_search",
        "must_contain_any": ["Lovelace", "mathematician", "programmer"],
    },
    {
        "id": "research_arxiv",
        "question": "Find research papers about retrieval augmented generation.",
        "category": "research",
        "expected_tool": "arxiv_search",
        "must_contain_any": ["arxiv", "retrieval", "generation"],
    },
    {
        "id": "current_web",
        "question": "What is the current temperature in Edmonton, AB in Celsius?",
        "category": "current",
        "expected_tool": "web_search",
        "must_contain_any": ["Edmonton", "Celsius", "temperature"],
    },
]


RAG_SCORECARD_CASES = [
    {
        "id": "rag_task_decomposition",
        "question": "What is task decomposition?",
        "category": "rag_factual",
        "expected_tool": "hybrid_retrieval",
        "must_contain_any": ["subtask", "steps", "decomposition", "complex"],
    },
    {
        "id": "rag_reflection",
        "question": "How does reflection help autonomous agents?",
        "category": "rag_factual",
        "expected_tool": "hybrid_retrieval",
        "must_contain_any": ["reflect", "mistake", "feedback", "improve"],
    },
    {
        "id": "rag_memory",
        "question": "What does the article say about memory in agents?",
        "category": "rag_factual",
        "expected_tool": "hybrid_retrieval",
        "must_contain_any": ["memory", "short-term", "long-term", "context"],
    },
    {
        "id": "rag_out_of_scope",
        "question": "What does the article say about the nutritional value of mangoes?",
        "category": "rag_out_of_scope",
        "expected_tool": "hybrid_retrieval",
        "must_contain_any": ["don't know", "not contain", "not mention", "context"],
    },
    {
        "id": "rag_tool_use",
        "question": "What is the purpose of tool use in LLM-powered agents?",
        "category": "rag_factual",
        "expected_tool": "hybrid_retrieval",
        "must_contain_any": ["tool", "external", "api", "capability"],
    },
]
