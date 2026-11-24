#!/usr/bin/env python3
"""TabPFN v2 tabular prediction API for Railway, CPU only.

TabPFN v2 (Prior Labs, published in Nature, 2025) is a pretrained transformer that classifies or regresses a small
table in one forward pass: no training loop, no hyperparameters. This file puts the `tabpfn` package behind one
authenticated HTTP API:

    POST /v1/predict   a training table and rows to score in, predictions (and probabilities / quantiles) out
    GET  /v1/models    the model, its licence, the limits in force
    GET  /health       503 until both checkpoints are loaded and warmed, then 200; the only unauthenticated route

What this file adds around the library, because upstream ships a Python package, not a server:

  1. Bearer API_KEY on every route except GET /health. An empty API_KEY refuses to start.
  2. Only the TabPFN-2 checkpoints, loaded from files baked into the image. The package's default weights
     (TabPFN-2.5 to 3.5) are non-commercial and gated behind a login; they are never fetched.
  3. Hard limits sized for CPU (rows, features, cells, classes), so a request cannot be accepted that the
     container could not finish inside Railway's roughly five minute edge timeout.
  4. torch's thread pool is sized from the container's cgroup quota (the entrypoint exports the count).
  5. One prediction at a time (CPU bound), a short queue, 429 when it is full. Data is held in memory for the
     request only and is never logged or written to disk.
  6. The attribution the Prior Labs licence asks for ("Built with PriorLabs-TabPFN") on every response.
"""

import asyncio
import hmac
import math
import os
import socket
import sys
import threading
import time
import warnings
from typing import Any, Literal

warnings.filterwarnings("ignore")

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

API_KEY = os.environ.get("API_KEY") or ""
THREADS = max(1, int(os.environ.get("TORCH_THREADS") or os.environ.get("THREADS") or os.cpu_count() or 1))
MODEL_ROOT = os.environ.get("MODEL_ROOT") or "/models"
MAX_TRAIN_ROWS = int(os.environ.get("MAX_TRAIN_ROWS") or 1000)      # TabPFN-2's CPU limit
MAX_TEST_ROWS = int(os.environ.get("MAX_TEST_ROWS") or 1000)
MAX_FEATURES = int(os.environ.get("MAX_FEATURES") or 100)
MAX_CELLS = int(os.environ.get("MAX_CELLS") or 120000)              # (train + test rows) x features
MAX_CLASSES = 10                                                    # what the TabPFN-2 classifier was trained for
MAX_SECONDS = float(os.environ.get("MAX_SECONDS") or 240)
MAX_PENDING = int(os.environ.get("MAX_PENDING") or 3)
MAX_BODY_BYTES = int(os.environ.get("MAX_BODY_BYTES") or 8 * 1024 * 1024)
PUBLIC_PATHS = {"/health"}
MODEL_ID = "tabpfn-v2"
BUILT_WITH = "PriorLabs-TabPFN"
CLF_CKPT = "tabpfn-v2-classifier.ckpt"
REG_CKPT = "tabpfn-v2-regressor.ckpt"

_state: dict[str, Any] = {"ready": False, "error": None, "loaded_in": None}
_run_lock = threading.Lock()      # one prediction at a time
_pending = 0
_pending_lock = threading.Lock()

app = FastAPI(title="TabPFN v2 prediction API", docs_url=None, redoc_url=None, openapi_url=None)


class Predict(BaseModel):
    task: Literal["classification", "regression"]
    X_train: list[list[Any]] = Field(description="training rows, each a list of numbers, strings (categorical) or null")
    y_train: list[Any] = Field(description="one target per training row")
    X_test: list[list[Any]] = Field(description="rows to score, same width as X_train")
    quantiles: list[float] | None = Field(default=None, description="regression only: e.g. [0.1, 0.5, 0.9]")
    n_estimators: int = Field(default=4, ge=1, le=16, description="ensemble size; more is slower and slightly better")
    seed: int = Field(default=0, ge=0, le=2**31 - 1)


class BadRequest(Exception):
    def __init__(self, status: int, message: str):
        self.status, self.message = status, message


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float, bool)) and not (isinstance(v, float) and not math.isfinite(v))


def _matrix(rows: list[list[Any]], name: str) -> int:
    if not rows:
        raise BadRequest(422, f"{name} has no rows")
    width = len(rows[0])
    if width == 0:
        raise BadRequest(422, f"{name} rows have no columns")
    for i, row in enumerate(rows):
        if len(row) != width:
            raise BadRequest(422, f"{name} row {i} has {len(row)} values, row 0 has {width}")
    return width


def _encode(train: list[list[Any]], test: list[list[Any]]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Numbers stay numbers, null becomes NaN, a column holding any string is categorical and is ordinal-coded
    from the training rows (a test value the training rows never held becomes NaN)."""
    width = len(train[0])
    xtr = np.full((len(train), width), np.nan, dtype=np.float64)
    xte = np.full((len(test), width), np.nan, dtype=np.float64)
    categorical: list[int] = []
    for j in range(width):
        col_tr = [r[j] for r in train]
        col_te = [r[j] for r in test]
        vals = [v for v in col_tr + col_te if v is not None]
        is_str = [isinstance(v, str) for v in vals]
        if any(is_str):
            if not all(is_str):
                raise BadRequest(422, f"column {j} mixes strings and numbers")
            codes = {v: float(k) for k, v in enumerate(sorted({v for v in col_tr if v is not None}))}
            xtr[:, j] = [codes.get(v, np.nan) if v is not None else np.nan for v in col_tr]
            xte[:, j] = [codes.get(v, np.nan) if v is not None else np.nan for v in col_te]
            categorical.append(j)
        else:
            for v in vals:
                if not _is_number(v):
                    raise BadRequest(422, f"column {j} holds a value that is not a finite number, string or null")
            xtr[:, j] = [float(v) if v is not None else np.nan for v in col_tr]
            xte[:, j] = [float(v) if v is not None else np.nan for v in col_te]
    return xtr, xte, categorical


def _check_limits(n_train: int, n_test: int, width: int) -> None:
    if n_train > MAX_TRAIN_ROWS:
        raise BadRequest(413, f"X_train has {n_train} rows; this CPU deployment accepts at most {MAX_TRAIN_ROWS}")
    if n_test > MAX_TEST_ROWS:
        raise BadRequest(413, f"X_test has {n_test} rows; this CPU deployment accepts at most {MAX_TEST_ROWS} per request")
    if width > MAX_FEATURES:
        raise BadRequest(413, f"{width} columns; this CPU deployment accepts at most {MAX_FEATURES}")
    if (n_train + n_test) * width > MAX_CELLS:
        raise BadRequest(413, f"(X_train + X_test rows) x columns = {(n_train + n_test) * width} exceeds the "
                              f"{MAX_CELLS} cell budget; send fewer rows or columns")


def _jsonable(a: np.ndarray) -> list:
    return np.where(np.isfinite(a), a, np.nan).astype(float).tolist() if a.dtype.kind == "f" else a.tolist()


def _run(req: Predict) -> dict[str, Any]:
    """Validate, fit, predict. Runs in a worker thread holding _run_lock."""
    from tabpfn import TabPFNClassifier, TabPFNRegressor

    width = _matrix(req.X_train, "X_train")
    if _matrix(req.X_test, "X_test") != width:
        raise BadRequest(422, "X_test has a different number of columns than X_train")
    if len(req.y_train) != len(req.X_train):
        raise BadRequest(422, f"y_train has {len(req.y_train)} values for {len(req.X_train)} X_train rows")
    _check_limits(len(req.X_train), len(req.X_test), width)
    xtr, xte, categorical = _encode(req.X_train, req.X_test)
    t0 = time.time()
    out: dict[str, Any] = {"model": MODEL_ID, "task": req.task, "built_with": BUILT_WITH,
                           "n_train": len(xtr), "n_test": len(xte), "n_features": width}
    common = dict(device="cpu", n_estimators=req.n_estimators, random_state=req.seed,
                  categorical_features_indices=categorical or None, n_jobs=THREADS)
    if req.task == "classification":
        if req.quantiles is not None:
            raise BadRequest(422, "quantiles apply to regression only")
        labels = req.y_train
        if any(v is None or isinstance(v, (list, dict)) for v in labels):
            raise BadRequest(422, "y_train must hold one label (string, number or boolean) per row, no nulls")
        kinds = {type(v) is str for v in labels}
        if len(kinds) > 1:
            raise BadRequest(422, "y_train mixes string and non-string labels")
        classes = sorted(set(labels))
        if len(classes) < 2:
            raise BadRequest(422, "y_train needs at least two distinct labels")
        if len(classes) > MAX_CLASSES:
            raise BadRequest(413, f"y_train has {len(classes)} classes; TabPFN v2 supports at most {MAX_CLASSES}")
        index = {c: k for k, c in enumerate(classes)}
        y = np.array([index[v] for v in labels], dtype=np.int64)
        model = TabPFNClassifier(model_path=os.path.join(MODEL_ROOT, CLF_CKPT), **common)
        model.fit(xtr, y)
        proba = model.predict_proba(xte)
        out["classes"] = classes
        out["predictions"] = [classes[int(k)] for k in proba.argmax(axis=1)]
        out["probabilities"] = _jsonable(proba)
    else:
        if not all(_is_number(v) and not isinstance(v, bool) for v in req.y_train):
            raise BadRequest(422, "y_train must be finite numbers for regression")
        y = np.array(req.y_train, dtype=np.float64)
        qs = req.quantiles
        if qs is not None:
            if not qs or len(qs) > 9 or any(not (0.0 < q < 1.0) for q in qs):
                raise BadRequest(422, "quantiles must be 1 to 9 numbers strictly between 0 and 1")
        model = TabPFNRegressor(model_path=os.path.join(MODEL_ROOT, REG_CKPT), **common)
        model.fit(xtr, y)
        out["predictions"] = _jsonable(model.predict(xte, output_type="mean"))
        if qs:
            full = model.predict(xte, output_type="quantiles", quantiles=list(qs))
            out["quantiles"] = {str(q): _jsonable(np.asarray(col)) for q, col in zip(qs, full)}
    out["seconds"] = round(time.time() - t0, 3)
    return out


def _guarded_run(req: Predict) -> dict[str, Any]:
    global _pending
    try:
        with _run_lock:
            return _run(req)
    finally:
        with _pending_lock:
            _pending -= 1


def _boot() -> None:
    t0 = time.time()
    try:
        torch.set_num_threads(THREADS)
        for ckpt in (CLF_CKPT, REG_CKPT):
            if not os.path.isfile(os.path.join(MODEL_ROOT, ckpt)):
                raise FileNotFoundError(os.path.join(MODEL_ROOT, ckpt))
        rng = np.random.default_rng(0)
        x = rng.normal(size=(40, 3)).tolist()
        with _run_lock:
            _run(Predict(task="classification", X_train=x[:30], y_train=[int(r[0] > 0) for r in x[:30]], X_test=x[30:]))
            _run(Predict(task="regression", X_train=x[:30], y_train=[r[0] * 2 for r in x[:30]], X_test=x[30:]))
        _state["ready"] = True
        _state["loaded_in"] = round(time.time() - t0, 1)
        print(f"[tabpfn] ready threads={THREADS} in {_state['loaded_in']}s", flush=True)
    except Exception as exc:  # noqa: BLE001 - reported over /health, never fatal
        _state["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[tabpfn] LOAD FAILED: {_state['error']}", flush=True)


def _reply(body: dict, status: int = 200, headers: dict | None = None) -> JSONResponse:
    h = {"X-Built-With": BUILT_WITH, "X-Model": MODEL_ID}
    h.update(headers or {})
    return JSONResponse(body, status_code=status, headers=h)


@app.middleware("http")
async def guards(request: Request, call_next):
    path = request.url.path.rstrip("/") or "/"
    if path in PUBLIC_PATHS:
        return await call_next(request)
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if not (scheme.lower() == "bearer" and API_KEY and hmac.compare_digest(token.encode(), API_KEY.encode())):
        return JSONResponse({"error": "Missing or invalid Authorization: Bearer <API_KEY>."},
                            status_code=401, headers={"WWW-Authenticate": "Bearer"})
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > MAX_BODY_BYTES:
        return JSONResponse({"error": f"request body is over {MAX_BODY_BYTES} bytes"}, status_code=413)
    return await call_next(request)


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError):
    first = exc.errors()[0] if exc.errors() else {}
    where = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
    return JSONResponse({"error": f"invalid request: {where}: {first.get('msg', 'bad value')}"}, status_code=422)


@app.get("/health")
async def health():
    if _state["error"]:
        return JSONResponse({"status": "error", "error": _state["error"]}, status_code=500)
    if not _state["ready"]:
        return JSONResponse({"status": "loading"}, status_code=503)
    return {"status": "ok", "model": MODEL_ID, "built_with": BUILT_WITH}


@app.get("/v1/models")
async def models():
    return _reply({"object": "list", "data": [{
        "id": MODEL_ID, "tasks": ["classification", "regression"], "built_with": BUILT_WITH,
        "weights": {"classifier": "Prior-Labs/TabPFN-v2-clf", "regressor": "Prior-Labs/TabPFN-v2-reg",
                    "license": "Prior Labs License 1.1 (Apache 2.0 with an attribution requirement)"},
        "limits": {"max_train_rows": MAX_TRAIN_ROWS, "max_test_rows": MAX_TEST_ROWS, "max_features": MAX_FEATURES,
                   "max_cells": MAX_CELLS, "max_classes": MAX_CLASSES, "max_seconds": MAX_SECONDS},
    }]})


@app.post("/v1/predict")
async def predict(req: Predict):
    global _pending
    if not _state["ready"]:
        return JSONResponse({"error": "model is loading"}, status_code=503, headers={"Retry-After": "5"})
    try:  # reject before queueing, so a bad request never waits behind a good one
        width = _matrix(req.X_train, "X_train")
        if _matrix(req.X_test, "X_test") != width:
            raise BadRequest(422, "X_test has a different number of columns than X_train")
        _check_limits(len(req.X_train), len(req.X_test), width)
    except BadRequest as exc:
        return JSONResponse({"error": exc.message}, status_code=exc.status)
    with _pending_lock:
        if _pending >= MAX_PENDING:
            return JSONResponse({"error": "busy: too many predictions queued"}, status_code=429,
                                headers={"Retry-After": "30"})
        _pending += 1
    fut = asyncio.get_running_loop().run_in_executor(None, _guarded_run, req)
    try:
        result = await asyncio.wait_for(asyncio.shield(fut), timeout=MAX_SECONDS)
    except asyncio.TimeoutError:
        return JSONResponse({"error": f"did not finish within {MAX_SECONDS:.0f} s; send fewer rows or columns"},
                            status_code=504)
    except BadRequest as exc:
        return JSONResponse({"error": exc.message}, status_code=exc.status)
    return _reply(result, headers={"X-Processing-Seconds": str(result["seconds"])})


def main() -> int:
    if not API_KEY:
        print("[tabpfn] FATAL: API_KEY is empty; refusing to publish the API to anyone who finds the URL.",
              file=sys.stderr, flush=True)
        return 1
    threading.Thread(target=_boot, name="load", daemon=True).start()
    # Railway injects PORT and its healthcheck dials that injected port. asyncio sets
    # IPV6_V6ONLY on a "::" listener, so one socket per family: IPv4 for the edge and
    # healthcheck, IPv6 for the private network (<service>.railway.internal).
    port = int(os.environ.get("PORT") or 8080)
    socks = []
    for family, addr in ((socket.AF_INET, "0.0.0.0"), (socket.AF_INET6, "::")):
        try:
            sock = socket.socket(family, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if family == socket.AF_INET6:
                sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            sock.bind((addr, port))
            sock.set_inheritable(True)
            socks.append(sock)
        except OSError as exc:
            print(f"[tabpfn] cannot bind {addr}:{port}: {exc}", flush=True)
    if not socks:
        return 1
    uvicorn.Server(uvicorn.Config(app, log_level=os.environ.get("LOG_LEVEL", "warning"))).run(sockets=socks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
