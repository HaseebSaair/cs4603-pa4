"""Vector Search retriever factory (Task 1.4 support / rag/store.py).

TODO: Implement `get_retriever(k=4)` that returns a LangChain retriever over the
Databricks Vector Search index built by `ingest.py`, using
`DatabricksVectorSearch` from `databricks_langchain`. Read endpoint/index names
from config.get_settings(). This exact retriever is reused by the deployed model.
"""

from __future__ import annotations

from config import get_settings

TEXT_COLUMN = "chunk_to_retrieve"
CITATION_COLUMNS =  ["chunk_id", "chunk_to_retrieve", "source", "page"] # added chunk to retreieve


def get_vector_store():
    from databricks_langchain import DatabricksVectorSearch

    s =get_settings()
    return DatabricksVectorSearch(
        index_name=s["vs_index"],
        endpoint=s["vs_endpoint"],
        # text_column= TEXT_COLUMN, # bcs fo that embedding chunk issue 
        columns=CITATION_COLUMNS,
    )
    #raise NotImplementedError("Task 1.4: return a DatabricksVectorSearch handle")


def get_retriever(k: int = 4):
    return get_vector_store().as_retriever(search_kwargs={"k": k})
    #raise NotImplementedError("Task 1.4: return a top-k retriever over the index")
    