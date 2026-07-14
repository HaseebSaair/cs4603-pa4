"""RAG agent node (Task 1.4) — retrieves from Databricks Vector Search.

TODO: Implement `make_rag_agent(retriever, llm)` returning a node that:
  - retrieves top-k chunks for the current step,
  - formats them with [source: file, p.N] citations,
  - extracts a single cited fact via the LLM (or 'not found in documents'),
  - appends the fact to step_results and increments current_step_index.
Reuse `rag/store.py::get_retriever()` so local and deployed retrieval match.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from agent.prompts import RAG_EXTRACT_PROMPT
from agent.state import AnalystState


def format_docs(docs) -> str:
    if not docs:
        return ""
    parts=[]
    for d in docs:
        text= d.metadata.get("chunk_to_retrieve", d.page_content)
        source= d.metadata.get("source", "unknown")
        page= d.metadata.get("page", "?")
        parts.append(f"[source: {source}, p.{page}]\n{text}")
    return "\n\n".join(parts)
    #raise NotImplementedError("Task 1.4: format retrieved docs with citations")


def make_rag_agent(retriever, llm):
    def rag_agent(state: AnalystState) -> dict:
        #raise NotImplementedError("Task 1.4: implement the RAG node")
        step =state["plan"][state["current_step_index"]]
        documents= retriever.invoke(step)
        context= format_docs(documents)

        if not context:
            fact= "not found in documents"
        else:
            response =llm.invoke([
                SystemMessage(content=RAG_EXTRACT_PROMPT),
                HumanMessage(content=f"Step: {step}\n\nContext:\n{context}"),
            ])
            fact=response.content.strip()

        return {
            "step_results": state["step_results"] + [fact],
            "current_step_index": state["current_step_index"] + 1,
        }


    return rag_agent
