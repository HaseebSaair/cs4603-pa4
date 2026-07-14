"""Offline smoke test for the Document Analyst graph (Bonus A test target).

This is the target the Bonus A CI pipeline runs to prove the graph wires up
before any deploy. Fill it in once your nodes are implemented.

TODO (Task 1.7 / Bonus A):
  - Build fake LLM / retriever / tool objects (no Databricks, no network).
  - Call `build_graph(llm=FakeLLM(), retriever=FakeRetriever(), tools=[FakeTool()])`.
  - Invoke it on a combined retrieval+calculation query and assert that a plan was
    produced, both specialists ran, and the final answer surfaced on messages[-1].

Run:  uv run pytest -q
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.graph import build_graph

class FakeMessage:
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class FakeLLM:
    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        system = messages[0].content
        user = messages[-1].content
        if "planning stage" in system:
            return FakeMessage(content='["Look up net income", "Calculate 10 percent of it"]')
        if "Classify the given step" in system:
            return FakeMessage(content="mcp_tools" if "Calculate" in user else "rag_agent")
        if "calculator tools" in system:
            return FakeMessage(tool_calls=[{"name": "calculate", "args": {"expression": "100 * 0.1"}}])
        if "ONLY the provided document" in system:
            return FakeMessage(content="Net income was $100 million [source: fake.pdf, p.1]")
        if "final answer" in system:
            return FakeMessage(content="Net income was $100 million; 10% of it is $10 million.")
        return FakeMessage(content="ok")


class FakeDoc:
    def __init__(self, content, metadata):
        self.page_content = content
        self.metadata = metadata


class FakeRetriever:
    def invoke(self, query):
        return [
            FakeDoc(
                "Net income was $100 million.",
                {"chunk_to_retrieve": "Net income was $100 million.", "source": "fake.pdf", "page": 1},
            )
        ]


class FakeTool:
    name = "calculate"


def test_graph_module_imports():
    """Minimal collection guard: the graph module must import cleanly."""
    from agent.graph import build_graph  # noqa: F401


def test_graph_compiles_and_runs_offline():
    graph = build_graph(llm=FakeLLM(), retriever=FakeRetriever(), tools=[FakeTool()])

    result = graph.invoke(
        {
            "messages": [{"role": "user", "content": "What was net income, and what is 10% of it?"}],
            "plan": [],
            "current_step_index": 0,
            "step_results": [],
            "next_agent": "",
            "final_answer": "",
        }
    )

    assert len(result["plan"]) >= 2
    assert len(result["step_results"])== len(result["plan"])
    assert result["final_answer"]
    assert result["messages"][-1].content == result["final_answer"]