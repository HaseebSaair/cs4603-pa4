"""MLflow models-from-code definition (Task 2.1).

TODO: Make this file self-contained so MLflow can serialise it:
  - validate DATABRICKS_HOST/TOKEN/MODEL at import time (clear error if missing),
  - rebuild the graph with production clients (LLM, Vector Search retriever,
    MCP tools),
  - end with `mlflow.models.set_model(graph)`.

Must import cleanly:  python -c "import deployment.agent_model"
"""

from __future__ import annotations

import mlflow

from config import get_settings
from agent.graph import build_graph

get_settings()

graph = build_graph()

mlflow.models.set_model(graph)

# TODO: import os, mlflow, build_graph, get_chat_llm, get_retriever, load_mcp_tools
# TODO: validate env vars
# TODO: graph = build_graph(...)
# TODO: mlflow.models.set_model(graph)
