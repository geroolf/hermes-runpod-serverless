from __future__ import annotations

import asyncio
import contextlib
import glob
import logging
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import AsyncIterator

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse


LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("hermes_serverless_worker")

LLAMA_HOST = os.getenv("LLAMA_HOST", "127.0.0.1")
LLAMA_PORT = int(os.getenv("LLAMA_PORT", "8080"))
LLAMA_BASE_URL = f"http://{LLAMA_HOST}:{LLAMA_PORT}"
MODEL_REPO = os.getenv(
    "MODEL_REPO", "oolfBER/Qwen3.6-27B-UD-Q4_K_XL-single-GGUF"
)
MODEL_FILENAME = os.getenv(
    "MODEL_FILENAME", "Qwen3.6-27B-UD-Q4_K_XL.gguf"
)
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "qwen3.6-27b-local")
MODEL_EXPECTED_SIZE = int(os.getenv("MODEL_EXPECTED_SIZE", "17909097600"))
MODEL_SEARCH_TIMEOUT = int(os.getenv("MODEL_SEARCH_TIMEOUT", "300"))
MODEL_LOAD_TIMEOUT = int(os.getenv("MODEL_LOAD_TIMEOUT", "600"))
PORT = int(os.getenv("PORT", "80"))

HOP_BY_HOP_HEADERS = {
    "authorization",
    "connection",
    "content-length",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}

app = FastAPI(
    title="Hermes Qwen llama.cpp Serverless Worker",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

llama_process: subprocess.Popen[bytes] | None = None
llama_ready = asyncio.Event()
llama_failed: str | None = None
started_at = time.monotonic()
ready_at: float | None = None
request_slot = asyncio.Semaphore(1)
shutdown_event = asyncio.Event()
client = httpx.AsyncClient(
    base_url=LLAMA_BASE_URL,
    timeout=httpx.Timeout(connect=10.0, read=None, write=60.0, pool=10.0),
)


def cached_repo_directory() -> str:
    return "models--" + MODEL_REPO.replace("/", "--")


def find_model() -> Path | None:
    explicit_path = os.getenv("MODEL_PATH")
    candidates: list[Path] = []

    if explicit_path:
        candidates.append(Path(explicit_path))

    cache_root = Path("/runpod-volume/huggingface-cache/hub")
    repo_root = cache_root / cached_repo_directory()
    candidates.extend(
        Path(path)
        for path in glob.glob(
            str(repo_root / "snapshots" / "*" / MODEL_FILENAME)
        )
    )

    candidates.extend(
        [
            Path("/runpod-volume") / MODEL_FILENAME,
            Path("/models") / MODEL_FILENAME,
        ]
    )

    for candidate in candidates:
        try:
            if (
                candidate.is_file()
                and candidate.stat().st_size == MODEL_EXPECTED_SIZE
            ):
                return candidate.resolve()
        except OSError:
            continue
    return None


def llama_command(model_path: Path) -> list[str]:
    return [
        "/usr/local/bin/llama-server",
        "--model",
        str(model_path),
        "--alias",
        MODEL_ALIAS,
        "--host",
        LLAMA_HOST,
        "--port",
        str(LLAMA_PORT),
        "--ctx-size",
        "114688",
        "--parallel",
        "1",
        "--n-gpu-layers",
        "999",
        "--flash-attn",
        "on",
        "--cache-type-k",
        "q4_0",
        "--cache-type-v",
        "q4_0",
        "--jinja",
        "--no-webui",
    ]


async def wait_for_model() -> Path:
    deadline = time.monotonic() + MODEL_SEARCH_TIMEOUT
    while time.monotonic() < deadline:
        model_path = find_model()
        if model_path:
            return model_path
        await asyncio.sleep(2)
    raise RuntimeError(
        f"model not found after {MODEL_SEARCH_TIMEOUT}s: "
        f"{MODEL_REPO}/{MODEL_FILENAME}"
    )


async def wait_for_llama() -> None:
    global ready_at

    deadline = time.monotonic() + MODEL_LOAD_TIMEOUT
    while time.monotonic() < deadline:
        if llama_process and llama_process.poll() is not None:
            raise RuntimeError(
                f"llama-server exited with status {llama_process.returncode}"
            )
        try:
            response = await client.get("/v1/models")
            if response.status_code == 200:
                ready_at = time.monotonic()
                llama_ready.set()
                logger.info(
                    "model ready alias=%s elapsed_seconds=%.3f",
                    MODEL_ALIAS,
                    ready_at - started_at,
                )
                return
        except httpx.HTTPError:
            pass
        await asyncio.sleep(1)
    raise RuntimeError(
        f"llama-server not ready after {MODEL_LOAD_TIMEOUT}s"
    )


async def launch_llama() -> None:
    global llama_process, llama_failed

    try:
        model_path = await wait_for_model()
        logger.info(
            "model located path=%s bytes=%d",
            model_path,
            model_path.stat().st_size,
        )
        llama_process = subprocess.Popen(
            llama_command(model_path),
            stdin=subprocess.DEVNULL,
        )
        logger.info("llama-server started pid=%s", llama_process.pid)
        await wait_for_llama()
    except Exception as exc:
        llama_failed = str(exc)
        logger.exception("worker initialization failed")


@app.on_event("startup")
async def startup() -> None:
    asyncio.create_task(launch_llama())


@app.on_event("shutdown")
async def shutdown() -> None:
    shutdown_event.set()
    await client.aclose()
    if llama_process and llama_process.poll() is None:
        llama_process.send_signal(signal.SIGTERM)
        try:
            await asyncio.to_thread(llama_process.wait, timeout=20)
        except subprocess.TimeoutExpired:
            llama_process.kill()
            await asyncio.to_thread(llama_process.wait)


@app.get("/ping")
async def ping() -> Response:
    if llama_ready.is_set():
        return Response(status_code=200)
    if llama_failed:
        return JSONResponse(
            status_code=503,
            content={"status": "failed", "error": llama_failed},
        )
    return Response(status_code=204)


@app.get("/worker/status")
async def worker_status() -> dict[str, object]:
    return {
        "ready": llama_ready.is_set(),
        "failed": llama_failed,
        "model_repo": MODEL_REPO,
        "model_filename": MODEL_FILENAME,
        "model_alias": MODEL_ALIAS,
        "initialization_seconds": (
            round(ready_at - started_at, 3) if ready_at else None
        ),
    }


def upstream_request_headers(request: Request) -> dict[str, str]:
    return {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS
    }


def downstream_response_headers(response: httpx.Response) -> dict[str, str]:
    return {
        key: value
        for key, value in response.headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS
    }


async def stream_upstream(
    response: httpx.Response,
) -> AsyncIterator[bytes]:
    try:
        async for chunk in response.aiter_raw():
            yield chunk
    finally:
        await response.aclose()
        request_slot.release()


@app.api_route(
    "/v1/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
)
async def proxy_v1(path: str, request: Request) -> Response:
    if not llama_ready.is_set():
        return JSONResponse(
            status_code=503,
            content={"error": {"message": "model is initializing"}},
        )

    await request_slot.acquire()
    try:
        body = await request.body()
        upstream_request = client.build_request(
            method=request.method,
            url=f"/v1/{path}",
            params=request.query_params,
            headers=upstream_request_headers(request),
            content=body,
        )
        upstream_response = await client.send(
            upstream_request,
            stream=True,
        )
    except Exception:
        request_slot.release()
        logger.exception("upstream request failed path=/v1/%s", path)
        return JSONResponse(
            status_code=502,
            content={"error": {"message": "upstream model request failed"}},
        )

    headers = downstream_response_headers(upstream_response)
    media_type = upstream_response.headers.get("content-type")
    return StreamingResponse(
        stream_upstream(upstream_response),
        status_code=upstream_response.status_code,
        headers=headers,
        media_type=media_type,
    )


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=PORT,
            access_log=True,
            log_level=LOG_LEVEL.lower(),
        )

