"""Full Document Analyst graph (Tasks 1.5 + 1.7).

TODO:
  - `load_mcp_tools(server_path=None)`: connect the GIVEN MCP server over stdio
    (see langchain-mcp-adapters) and return its tools.
  - `make_mcp_node(tools, llm)`: execute one calculation step by letting the LLM
    call exactly one MCP tool, then append the result and increment the index.
  - `build_graph(llm=None, retriever=None, tools=None)`: assemble
    planner -> supervisor -> {rag_agent | mcp_tools} -> ... -> synthesizer.
    Inject dependencies so the graph can be unit-tested offline with fakes.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import tempfile
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agent.planner import make_planner
from agent.prompts import MCP_STEP_PROMPT
from agent.rag_agent import make_rag_agent
from agent.state import AnalystState
from agent.supervisor import MCP, RAG, SYNTH, make_supervisor, route_from_supervisor
from agent.synthesizer import make_synthesizer


def _run_async(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()
    
_DEFAULT_SERVER_PATH = str(Path(__file__).resolve().parent.parent / "tools" / "mcp_server.py")


def callmcptool(name: str, args: dict, server_path: str) -> str:
    params = StdioServerParameters(command="python", args=[server_path])

    async def _call():
        with open(tempfile.gettempdir() + "/mcp_stderr.log", "w") as errlog:
            async with stdio_client(params, errlog=errlog) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(name, args)
                    return result.content[0].text

    return _run_async(_call())   


def load_mcp_tools(server_path: str | None = None):
    #raise NotImplementedError("Task 1.5: connect the MCP server and return its tools")
    from langchain_mcp_adapters.tools import load_mcp_tools as _load_mcp_tools

    server_path = server_path or _DEFAULT_SERVER_PATH
    parameters=StdioServerParameters(command="python", args=[server_path])

    async def _fetch():
        with open(tempfile.gettempdir() + "/mcp_stderr.log", "w") as errlog:
            async with stdio_client(parameters, errlog=errlog) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    return await _load_mcp_tools(session)

    return _run_async(_fetch())  


def make_mcp_node(tools, llm):
    bound_llm= llm.bind_tools(tools)
    def mcp_tools(state: AnalystState) -> dict:
        #raise NotImplementedError("Task 1.5: implement the MCP tool node")
        step= state["plan"][state["current_step_index"]]
        prior= "\n".join(
            f"Step {i + 1} result: {r}" for i, r in enumerate(state["step_results"])
        )
        content=f"{prior}\n\nCurrent step: {step}" if prior else step

        response= bound_llm.invoke([SystemMessage(content=MCP_STEP_PROMPT), HumanMessage(content=content)])

        if response.tool_calls:
            call= response.tool_calls[0]
            result=callmcptool(call["name"], call["args"], _DEFAULT_SERVER_PATH)
            fact =f"{call['name']}({call['args']}) = {result}"
        else:
            fact= response.content.strip()

        return {
            "step_results": state["step_results"] + [fact],
            "current_step_index": state["current_step_index"] + 1,
        }

    return mcp_tools


def build_graph(llm=None, retriever=None, tools=None):
    #raise NotImplementedError("Task 1.7: wire and compile the full graph")
    if llm is None:
        from config import get_chat_llm

        llm=get_chat_llm()
    if retriever is None:
        from rag.store import get_retriever

        retriever= get_retriever()
    if tools is None:
        tools =load_mcp_tools()

    planner= make_planner(llm)
    supervisor= make_supervisor(llm)
    rag_agent =make_rag_agent(retriever, llm)
    mcp_tools_node= make_mcp_node(tools, llm)
    synthesizer =make_synthesizer(llm)

    builder=StateGraph(AnalystState)
    builder.add_node("planner", planner)
    builder.add_node("supervisor", supervisor)
    builder.add_node("rag_agent", rag_agent)
    builder.add_node("mcp_tools", mcp_tools_node)
    builder.add_node("synthesizer", synthesizer)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "supervisor")
    builder.add_conditional_edges(
        "supervisor", route_from_supervisor, {RAG: "rag_agent", MCP: "mcp_tools", SYNTH: "synthesizer"}
    )
    builder.add_edge("rag_agent", "supervisor")
    builder.add_edge("mcp_tools", "supervisor")
    builder.add_edge("synthesizer", END)

    return builder.compile()
