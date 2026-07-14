"""Planner node (Task 1.2).

TODO: Implement `make_planner(llm)` returning a node that:
  - reads the user question from state["messages"],
  - asks the LLM (PLANNER_PROMPT) for a JSON list of 2-5 steps,
  - parses it robustly (fallback to a single step on parse failure),
  - returns {"plan": [...], "current_step_index": 0, "step_results": []}.
"""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from agent.prompts import PLANNER_PROMPT
from agent.state import AnalystState


def parseplan(text: str, query: str) -> list[str]:
    text =text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        plan=json.loads(text)
        if isinstance(plan, list) and plan and all(isinstance(s, str) for s in plan):
            return plan
    except json.JSONDecodeError:
        pass
    return [query]

def make_planner(llm):
    def planner(state: AnalystState) -> dict:
        #raise NotImplementedError("Task 1.2: implement the planner node")
        query =state["messages"][-1].content
        response = llm.invoke([SystemMessage(content=PLANNER_PROMPT), HumanMessage(content=query)])
        plan= parseplan(response.content, query)
        return {"plan": plan, "current_step_index": 0, "step_results": []}

    return planner
