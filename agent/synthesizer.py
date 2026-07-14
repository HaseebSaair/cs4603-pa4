"""Synthesizer node (Task 1.6).

TODO: Implement `make_synthesizer(llm)` returning a node that combines
step_results into one cited answer and writes it to BOTH `final_answer` AND
the `messages` channel as an AIMessage (required for the OpenAI-compatible
serving contract — see spec Task 1.6).
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.prompts import SYNTHESIZER_PROMPT
from agent.state import AnalystState


def make_synthesizer(llm):
    def synthesizer(state: AnalystState) -> dict:
        #raise NotImplementedError("Task 1.6: implement the synthesizer node")
        query= state["messages"][0].content
        steps ="\n".join(
            f"Step {i + 1}: {step}\nResult: {result}"
            for i, (step, result) in enumerate(zip(state["plan"], state["step_results"], strict=True))
        )
        context=f"Original question: {query}\n\n{steps}"

        response= llm.invoke([SystemMessage(content=SYNTHESIZER_PROMPT), HumanMessage(content=context)])
        answer= response.content.strip()

        return {"final_answer": answer, "messages": [AIMessage(content=answer)]}


    return synthesizer
