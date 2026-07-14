"""Corpus ingestion into Databricks Vector Search (Task 0.3 / rag/ingest.py).

Run inside a Databricks notebook (needs Spark + ai_parse_document/ai_prep_search).
Mirror PA2 Part 1:

TODO:
  - `build_chunks_table(spark, volume_path, chunks_table)`: parse the PDF with
    ai_parse_document, chunk with ai_prep_search into a Delta table with columns
    chunk_id, chunk_to_retrieve, chunk_to_embed, source, page. Enable Change Data
    Feed on the table.
  - `create_index()`: create a STANDARD Vector Search endpoint and a TRIGGERED
    Delta Sync index (primary_key='chunk_id',
    embedding_source_column='chunk_to_retrieve',
    embedding_model_endpoint_name=$EMBEDDINGS_ENDPOINT).
"""

from __future__ import annotations
from config import get_settings
import os


def build_chunks_table(spark, volume_path: str, chunks_table: str) -> None:
    spark.sql(f"""
        CREATE OR REPLACE TEMP VIEW parsed AS
        SELECT path, ai_parse_document(content) AS parsed
        FROM READ_FILES('{volume_path}', format => 'binaryFile')
    """)

    spark.sql("""
        CREATE OR REPLACE TEMP VIEW prepped AS
        SELECT path, ai_prep_search(parsed) AS result
        FROM parsed
    """)

    df = spark.sql("""
        SELECT
          chunk.value:chunk_id::STRING AS chunk_id,
          chunk.value:chunk_to_retrieve::STRING AS chunk_to_retrieve,
          chunk.value:chunk_to_embed::STRING AS chunk_to_embed,
          path AS source,
          int(chunk.value:pages[0]:page_id) + 1 AS page
        FROM prepped, LATERAL variant_explode(  result:document.contents) AS chunk
    """)
    df.write.mode("overwrite").saveAsTable(chunks_table)
    spark.sql(f"ALTER TABLE {chunks_table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
    #raise NotImplementedError("Task 0.3: parse + chunk into a Delta table")


def create_index() -> None:
    from databricks.vector_search.client import VectorSearchClient

    s =get_settings()
    source_table = os.environ["SOURCE_TABLE"]
    vsc = VectorSearchClient()

    existing = [e["name"] for e in vsc.list_endpoints().get("endpoints", [])]
    if s["vs_endpoint"] not in existing:
        vsc.create_endpoint(name=s["vs_endpoint"], endpoint_type="STANDARD")

    vsc.create_delta_sync_index_and_wait(
        endpoint_name=s["vs_endpoint"],
        index_name=s["vs_index"],
        source_table_name=source_table,
        pipeline_type="TRIGGERED",
        primary_key="chunk_id",
        embedding_source_column="chunk_to_embed",
        embedding_model_endpoint_name=s["embeddings"],
    )
    #raise NotImplementedError("Task 0.3: create the Vector Search Delta Sync index")
