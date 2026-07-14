"""Supervisor node + routing edge (Task 1.3).

TODO:
  - `make_supervisor(llm)`: if current_step_index >= len(plan) -> next_agent =
    'synthesizer'; else classify the current step to 'rag_agent' or 'mcp_tools'.
  - `route_from_supervisor(state)`: return state["next_agent"] for the
    conditional edge.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from agent.prompts import SUPERVISOR_PROMPT
from agent.state import AnalystState

RAG = "rag_agent"
MCP = "mcp_tools"
SYNTH = "synthesizer"


def make_supervisor(llm):
    def supervisor(state: AnalystState) -> dict:
        #raise NotImplementedError("Task 1.3: implement the supervisor node")
        if state["current_step_index"] >=len(state["plan"]):
            output= {"next_agent": SYNTH}
            return output

        step =state["plan"][state["current_step_index"]]
        response= llm.invoke([SystemMessage(content=SUPERVISOR_PROMPT), HumanMessage(content=step)])
        text=response.content.strip().lower()
        next_agent = MCP if MCP in text or "calc" in text else RAG
        return {"next_agent": next_agent}

    return supervisor


def route_from_supervisor(state: AnalystState) -> str:
    #raise NotImplementedError("Task 1.3: return state['next_agent']")
    return state["next_agent"]
