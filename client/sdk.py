"""Python client SDK for the deployed Document Analyst (Part 3).

TODO: Implement `DocumentAnalystClient` and `AnalystClientError` per Task 3.1:
  - __init__(endpoint_name, host=None, token=None, timeout=120.0, max_retries=3):
    read DATABRICKS_HOST/DATABRICKS_TOKEN from env when not provided.
  - ask(question) -> str
  - ask_streaming(question) -> Iterator[str]   (yield chunks as they arrive)
  - health_check() -> bool                      (True only when endpoint READY)
  - exponential backoff on 429/503, TimeoutError with elapsed time, and wrap HTTP
    errors in AnalystClientError(status_code, message, request_id).
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator

import requests


class AnalystClientError(Exception):
    def __init__(self, message: str, status_code=None, request_id=None):
        super().__init__(message)
        self.status_code = status_code
        self.request_id = request_id


class DocumentAnalystClient:
    def __init__(
        self,
        endpoint_name: str,
        host: str | None = None,
        token: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        #raise NotImplementedError("Task 3.1: implement the client constructor")
        self.endpoint_name=endpoint_name
        self.host =(host or os.environ["DATABRICKS_HOST"]).rstrip("/")
        self.token =token or os.environ["DATABRICKS_TOKEN"]
        self.timeout= timeout
        self.max_retries= max_retries

    def headers(self) -> dict:
        output= {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        return output

    def invocationurl(self) -> str:
        output= f"{self.host}/serving-endpoints/{self.endpoint_name}/invocations"
        return output
    def postretry(self, payload: dict, stream: bool = False):
        attempt= 0
        start =time.time()
        while True:
            try:
                response= requests.post(
                    self.invocationurl(),
                    headers=self.headers(),
                    json=payload,
                    timeout=self.timeout,
                    stream=stream,
                )
            except requests.exceptions.Timeout  as err:
                elapsed =time.time()- start
                raise TimeoutError(f"Request timed out after {elapsed:.1f}s") from err

            if response.status_code in (429, 503) and attempt <self.max_retries:
                time.sleep(2**attempt)
                attempt =attempt+ 1
                continue

            if response.status_code >= 400:
                raise AnalystClientError(
                    f"Endpoint returned {response.status_code}: {response.text[:200]}",
                    status_code=response.status_code,
                    request_id=response.headers.get("x-request-id"),
                )

            return response


    def ask(self, question: str) -> str:
        #raise NotImplementedError("Task 3.1: implement ask()")
        payload= {"messages": [{"role": "user", "content": question}]}
        response =self.postretry(payload)
        data=response.json()
        output= data[0]["messages"][-1]["content"]
        return output

    def ask_streaming(self, question: str) -> Iterator[str]:
        #raise NotImplementedError("Task 3.1: implement ask_streaming()")
        payload={"messages": [{"role": "user", "content": question}]}
        response =self.postretry(payload, stream=True)

        content_type=response.headers.get("content-type", "")
        if "text/event-stream" not in content_type:
            data=response.json()
            yield data[0]["messages"][-1]["content"]
            return

        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            rawchunk=line[len("data:") :].strip()
            if rawchunk=="[DONE]":
                break
            try:
                chunk=json.loads(rawchunk)
                delta=chunk["choices"][0]["delta"].get("content", "")
            except (KeyError, IndexError, json.JSONDecodeError):
                continue
            if delta:
                yield delta

    def health_check(self) -> bool:
        #raise NotImplementedError("Task 3.1: implement health_check()")
        url=f"{self.host}/api/2.0/serving-endpoints/{self.endpoint_name}"
        try:
            response=requests.get(url, headers=self.headers(), timeout=self.timeout)
        except requests.exceptions.RequestException:
            return False
        if response.status_code !=200:
            return False
        out=response.json().get("state", {}).get("ready") =="READY"
        return out
