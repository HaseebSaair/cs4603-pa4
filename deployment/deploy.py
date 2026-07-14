"""Log, register, and serve the Document Analyst (Tasks 2.2 + 2.3).

Run:  uv run python deployment/deploy.py

TODO:
  - `log_and_register()`: set registry uri to 'databricks-uc', log the model via
    `mlflow.langchain.log_model(lc_model="deployment/agent_model.py", name=...,
    code_paths=[...], pip_requirements=[...], input_example={...})`, then
    `mlflow.register_model(...)` into $UC_CATALOG.$UC_SCHEMA.<model>.
  - `create_or_update_endpoint(uc_name, version)`: create/update a Model Serving
    endpoint with `WorkspaceClient().serving_endpoints`, workload_size='Small',
    scale_to_zero_enabled=True, and environment_vars supplied as secret refs
    ({{secrets/cs4603-deploy/...}}). Wait for READY and print the URL.
"""

from __future__ import annotations
import os
import mlflow
from dotenv import load_dotenv

load_dotenv()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def log_and_register():
    #raise NotImplementedError("Task 2.2: log + register the model in Unity Catalog")
    from databricks.sdk import WorkspaceClient

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")

    email=WorkspaceClient().current_user.me().user_name
    mlflow.set_experiment(f"/Users/{email}/pa4-document-analyst")

    catalog= os.environ["UC_CATALOG"]
    schema=os.environ["UC_SCHEMA"]
    model_name = os.environ["SERVING_ENDPOINT_NAME"].replace("-", "_")

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            lc_model="deployment/agent_model.py",
            name="agent",
            code_paths=["agent", "rag", "tools", "config.py"],
            pip_requirements=[
                "mlflow",
                "langgraph",
                "langchain-openai",
                "langchain-core",
                "databricks-langchain",
                "databricks-vectorsearch",
                "mcp",
                "langchain-mcp-adapters",
                "python-dotenv",
            ],
            input_example={"messages": [{"role": "user", "content": "What was the revenue?"}]},
        )

    uc_name=f"{catalog}.{schema}.{model_name}"
    registered=mlflow.register_model(model_info.model_uri, uc_name)
    print(f"Registered {uc_name} version {registered.version}")
    return uc_name, registered.version

def secret_ref(scope: str, key: str) -> str:
    return "{{secrets/%s/%s}}" % (scope, key)

def create_or_update_endpoint(uc_name: str, version: str) -> str:
    #raise NotImplementedError("Task 2.3: create/update the serving endpoint")
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

    w=WorkspaceClient()
    endpointname= os.environ["SERVING_ENDPOINT_NAME"]
    scope =os.environ.get("SECRET_SCOPE", "cs4603-deploy")

    served_entities=[
        ServedEntityInput(
            entity_name=uc_name,
            entity_version=version,
            workload_size="Small",
            scale_to_zero_enabled=True,
            environment_vars={
                "DATABRICKS_HOST": secret_ref(scope, "DATABRICKS_HOST"),
                "DATABRICKS_TOKEN": secret_ref(scope, "DATABRICKS_TOKEN"),
                "DATABRICKS_MODEL": secret_ref(scope, "DATABRICKS_MODEL"),
                "VECTOR_SEARCH_ENDPOINT": os.environ["VECTOR_SEARCH_ENDPOINT"],
                "VECTOR_SEARCH_INDEX": os.environ["VECTOR_SEARCH_INDEX"],
                "EMBEDDINGS_ENDPOINT": os.environ["EMBEDDINGS_ENDPOINT"],
            },
        )
    ]

    existing_names = [e.name for e in w.serving_endpoints.list()]
    if endpointname in existing_names:
        w.serving_endpoints.update_config_and_wait(name=endpointname, served_entities=served_entities)
    else:
        w.serving_endpoints.create_and_wait(
            name=endpointname,
            config=EndpointCoreConfigInput(name=endpointname, served_entities=served_entities),
        )

    url = f"{os.environ['DATABRICKS_HOST'].rstrip('/')}/serving-endpoints/{endpointname}/invocations"
    print(f"Endpoint READY: {url}")
    return url



if __name__ == "__main__":
    name, ver = log_and_register()
    create_or_update_endpoint(name, ver)
