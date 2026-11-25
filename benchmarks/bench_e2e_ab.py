#!/usr/bin/env python3
"""Rigorous Stock SGLang vs sglang-sm89 E2E benchmark & A/B/A comparator.

Measures:
  - Concurrency c=1 and c=8 across temperatures (default: 0.0 and 0.6)
  - Workloads:
      * json: strict XGrammar JSON schema routing with ground-truth label checks
      * sql: executable SQLite analytics queries verified against a deterministic
        6-table seeded sqlite3 database (`orders`, `refunds`, `events`,
        `line_items`, `catalog`, `deployments`)
      * eng: unconstrained multi-paragraph technical & creative prose
  - Task accuracy & throughput: `valid_count`, `valid_rate_pct`, `valid_tasks_per_s`,
    `agg_tok_s`, `per_stream_median_tok_s`, `ttft_p50_ms`, and `tok_per_j`
  - Server telemetry: /get_server_info + /metrics (`max_total_num_tokens`, spec accept length,
    active request guard before benchmarking)
  - GPU power & energy telemetry (NVML / nvidia-smi polling -> average Watts, Joules, tok/J, peak VRAM)

Usage:
  # Run one arm against a live SGLang server (default --max-tokens 512)
  python3 benchmarks/bench_e2e_ab.py run --arm stock --base-url http://127.0.0.1:30002 --out results/stock.json

  # Compare two or three saved arms (e.g., control-a vs candidate vs control-b)
  python3 benchmarks/bench_e2e_ab.py compare results/control_a.json results/candidate.json results/control_b.json
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import shlex
import sqlite3
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
import uuid


ENGLISH_PROMPTS = [
    "Write a Python implementation of an LRU cache with O(1) get and put, including docstrings and edge-case tests.",
    "Explain how TCP congestion control works, covering slow start, congestion avoidance, fast retransmit, and fast recovery.",
    "Compare B-trees and LSM-trees for write-heavy storage engines, including compaction tradeoffs and read amplification.",
    "Write a concise short story about an astronomer tracking an anomalous signal across three nights in an alpine observatory.",
    "Explain CUDA warp-level primitives (__ballot_sync, __shfl_down_sync) and how they eliminate shared-memory reductions.",
    "Design a fault-tolerant job queue backed by PostgreSQL SKIP LOCKED with idempotency keys and exponential backoff.",
    "Summarize the architectural differences between chunked delta-rule linear attention and grouped-query softmax attention.",
    "Write a POSIX shell script that validates SHA-256 manifests atomically and rolls back symlinks on verification failure.",
]

SCHEMA_DDL = """
CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    region TEXT NOT NULL,
    net_amount REAL NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE refunds (
    refund_id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL,
    refund_amount REAL NOT NULL,
    refunded_at TEXT NOT NULL
);
CREATE TABLE events (
    session_id TEXT NOT NULL,
    user_id INTEGER NOT NULL,
    event_name TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    ts TEXT NOT NULL
);
CREATE TABLE line_items (
    order_id INTEGER NOT NULL,
    sku TEXT NOT NULL,
    qty INTEGER NOT NULL,
    unit_price REAL NOT NULL
);
CREATE TABLE catalog (
    sku TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    margin_rate REAL NOT NULL
);
CREATE TABLE deployments (
    service TEXT NOT NULL,
    revision TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    duration_s REAL NOT NULL
);
"""

SQL_CASES = [
    {
        "id": "regional_net_revenue",
        "prompt": (
            "Tables: `orders(order_id INTEGER, customer_id INTEGER, region TEXT, net_amount REAL, created_at TEXT)` "
            "and `refunds(refund_id INTEGER, order_id INTEGER, refund_amount REAL, refunded_at TEXT)`. "
            "Write one SQLite SQL query computing per-region net revenue after refunds "
            "(SUM(orders.net_amount) - COALESCE(SUM(r.total_refund), 0.0), aggregating refunds by order_id first to avoid row multiplication) "
            "for orders where created_at >= '2026-11-01'. Return ONLY columns (region, net_revenue) ordered by net_revenue DESC, region ASC."
        ),
        "reference_sql": (
            "SELECT o.region, SUM(o.net_amount) - COALESCE(SUM(r.total_refund), 0.0) AS net_revenue "
            "FROM orders o "
            "LEFT JOIN (SELECT order_id, SUM(refund_amount) AS total_refund FROM refunds GROUP BY order_id) r "
            "ON o.order_id = r.order_id "
            "WHERE o.created_at >= '2026-11-01' "
            "GROUP BY o.region "
            "ORDER BY net_revenue DESC, o.region ASC"
        ),
        "ordered": True,
    },
    {
        "id": "top3_refunded_regions",
        "prompt": (
            "Tables: `orders(order_id INTEGER, customer_id INTEGER, region TEXT, net_amount REAL, created_at TEXT)` "
            "and `refunds(refund_id INTEGER, order_id INTEGER, refund_amount REAL, refunded_at TEXT)`. "
            "Write one SQLite SQL query returning the top 3 regions by distinct refunded orders "
            "(COUNT(DISTINCT refunds.order_id) joined to orders on order_id). "
            "Return ONLY columns (region, refunded_orders) ordered by refunded_orders DESC, region ASC LIMIT 3."
        ),
        "reference_sql": (
            "SELECT o.region, COUNT(DISTINCT r.order_id) AS refunded_orders "
            "FROM refunds r JOIN orders o ON r.order_id = o.order_id "
            "GROUP BY o.region "
            "ORDER BY refunded_orders DESC, o.region ASC LIMIT 3"
        ),
        "ordered": True,
    },
    {
        "id": "qualifying_session_latency",
        "prompt": (
            "Table: `events(session_id TEXT, user_id INTEGER, event_name TEXT, latency_ms REAL, ts TEXT)`. "
            "Write one SQLite SQL query keeping ONLY rows whose session_id has at least 3 total events, "
            "then grouping by event_name to compute AVG(latency_ms) and COUNT(*). "
            "Return ONLY columns (event_name, avg_latency_ms, event_count) ordered by event_name ASC."
        ),
        "reference_sql": (
            "SELECT e.event_name, AVG(e.latency_ms) AS avg_latency_ms, COUNT(*) AS event_count "
            "FROM events e "
            "WHERE e.session_id IN (SELECT session_id FROM events GROUP BY session_id HAVING COUNT(*) >= 3) "
            "GROUP BY e.event_name "
            "ORDER BY e.event_name ASC"
        ),
        "ordered": True,
    },
    {
        "id": "top3_active_users",
        "prompt": (
            "Table: `events(session_id TEXT, user_id INTEGER, event_name TEXT, latency_ms REAL, ts TEXT)`. "
            "Write one SQLite SQL query finding the top 3 users by total event count. "
            "Return ONLY columns (user_id, total_events) ordered by total_events DESC, user_id ASC LIMIT 3."
        ),
        "reference_sql": (
            "SELECT user_id, COUNT(*) AS total_events "
            "FROM events "
            "GROUP BY user_id "
            "ORDER BY total_events DESC, user_id ASC LIMIT 3"
        ),
        "ordered": True,
    },
    {
        "id": "category_gross_margin",
        "prompt": (
            "Tables: `line_items(order_id INTEGER, sku TEXT, qty INTEGER, unit_price REAL)` and "
            "`catalog(sku TEXT, category TEXT, margin_rate REAL)`. "
            "Write one SQLite SQL query joining line_items and catalog on sku and computing "
            "total gross margin SUM(qty * unit_price * margin_rate) per category. "
            "Return ONLY columns (category, gross_margin) ordered by gross_margin DESC, category ASC."
        ),
        "reference_sql": (
            "SELECT c.category, SUM(l.qty * l.unit_price * c.margin_rate) AS gross_margin "
            "FROM line_items l JOIN catalog c ON l.sku = c.sku "
            "GROUP BY c.category "
            "ORDER BY gross_margin DESC, c.category ASC"
        ),
        "ordered": True,
    },
    {
        "id": "top5_skus_by_revenue",
        "prompt": (
            "Table: `line_items(order_id INTEGER, sku TEXT, qty INTEGER, unit_price REAL)`. "
            "Write one SQLite SQL query computing SUM(qty) and SUM(qty * unit_price) per sku, "
            "returning the top 5 SKUs as ONLY columns (sku, total_qty, total_revenue) "
            "ordered by total_revenue DESC, sku ASC LIMIT 5."
        ),
        "reference_sql": (
            "SELECT sku, SUM(qty) AS total_qty, SUM(qty * unit_price) AS total_revenue "
            "FROM line_items "
            "GROUP BY sku "
            "ORDER BY total_revenue DESC, sku ASC LIMIT 5"
        ),
        "ordered": True,
    },
    {
        "id": "deployment_failures_nov",
        "prompt": (
            "Table: `deployments(service TEXT, revision TEXT, status TEXT, started_at TEXT, duration_s REAL)`. "
            "Write one SQLite SQL query for rows where started_at >= '2026-11-10', computing per-service "
            "failed count SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) and average duration_s AVG(duration_s). "
            "Return ONLY columns (service, failed_count, avg_duration_s) ordered by failed_count DESC, service ASC."
        ),
        "reference_sql": (
            "SELECT service, SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed_count, "
            "AVG(duration_s) AS avg_duration_s "
            "FROM deployments "
            "WHERE started_at >= '2026-11-10' "
            "GROUP BY service "
            "ORDER BY failed_count DESC, service ASC"
        ),
        "ordered": True,
    },
    {
        "id": "high_refund_ratio_orders",
        "prompt": (
            "Tables: `orders(order_id INTEGER, customer_id INTEGER, region TEXT, net_amount REAL, created_at TEXT)` "
            "and `refunds(refund_id INTEGER, order_id INTEGER, refund_amount REAL, refunded_at TEXT)`. "
            "Write one SQLite SQL query finding all orders whose total refund_amount in refunds strictly exceeds "
            "0.5 * orders.net_amount. Return ONLY columns (order_id, net_amount, total_refund) ordered by order_id ASC."
        ),
        "reference_sql": (
            "SELECT o.order_id, o.net_amount, SUM(r.refund_amount) AS total_refund "
            "FROM orders o JOIN refunds r ON o.order_id = r.order_id "
            "GROUP BY o.order_id, o.net_amount "
            "HAVING SUM(r.refund_amount) > 0.5 * o.net_amount "
            "ORDER BY o.order_id ASC"
        ),
        "ordered": True,
    },
]

JSON_CASES = [
    {
        "text": "Analyst asks: Return monthly net revenue and refund totals by warehouse region for Q4.",
        "route": "sql_query",
        "needs_db": True,
    },
    {
        "text": "On-call asks: Why did the checkout service emit HTTP 503 trace spikes between 02:14 and 02:19 UTC?",
        "route": "ops_trace",
        "needs_db": False,
    },
    {
        "text": "Finance asks: Query our orders and refunds tables to list enterprise cohorts ranked by net retention.",
        "route": "sql_query",
        "needs_db": True,
    },
    {
        "text": "SRE asks: Inspect live GPU cluster telemetry for workers exceeding 85% VRAM utilization right now.",
        "route": "ops_trace",
        "needs_db": False,
    },
    {
        "text": "Billing asks: Aggregate overdue vendor invoice rows above 5000 USD from the ledger tables ordered by days past due.",
        "route": "sql_query",
        "needs_db": True,
    },
    {
        "text": "Support asks: Summarize the written refund SLA policy document for damaged freight claims.",
        "route": "knowledge_base",
        "needs_db": False,
    },
    {
        "text": "Developer asks: Explain our documented index design guidelines and migration changelog entry 0.14.",
        "route": "knowledge_base",
        "needs_db": False,
    },
    {
        "text": "Planner asks: Query deployment duration and failure rate statistics per microservice from the deployments table.",
        "route": "sql_query",
        "needs_db": True,
    },
]

JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "route_decision",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "route": {"type": "string", "enum": ["sql_query", "ops_trace", "knowledge_base"]},
                "needs_db": {"type": "boolean"},
                "confidence": {"type": "number"},
                "entities": {"type": "array", "items": {"type": "string"}},
                "rationale": {"type": "string"},
            },
            "required": ["route", "needs_db", "confidence", "entities", "rationale"],
            "additionalProperties": False,
        },
    },
}

SQL_JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "sql_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "dialect": {"type": "string", "enum": ["sqlite"]},
                "tables_used": {"type": "array", "items": {"type": "string"}},
                "sql": {"type": "string"},
            },
            "required": ["dialect", "tables_used", "sql"],
            "additionalProperties": False,
        },
    },
}


def seed_analytics_db() -> sqlite3.Connection:
    """Create a deterministic 6-table in-memory SQLite analytics database."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.executescript(SCHEMA_DDL)

    orders = [
        (1, 101, "us-east", 420.50, "2026-10-15"),
        (2, 102, "us-west", 310.00, "2026-10-28"),
        (3, 103, "us-east", 500.00, "2026-11-02"),
        (4, 104, "eu-central", 750.00, "2026-11-03"),
        (5, 105, "us-west", 620.00, "2026-11-05"),
        (6, 101, "ap-south", 280.00, "2026-11-07"),
        (7, 106, "us-east", 900.00, "2026-11-08"),
        (8, 107, "eu-central", 410.00, "2026-11-10"),
        (9, 108, "ap-east", 530.00, "2026-11-12"),
        (10, 102, "us-west", 240.00, "2026-11-14"),
        (11, 109, "ap-south", 690.00, "2026-11-16"),
        (12, 110, "us-east", 350.00, "2026-11-18"),
        (13, 103, "eu-central", 880.00, "2026-11-20"),
        (14, 104, "ap-east", 470.00, "2026-11-22"),
    ]
    conn.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?)", orders)

    refunds = [
        (1, 3, 300.00, "2026-11-04"),
        (2, 3, 50.00, "2026-11-05"),
        (3, 5, 120.00, "2026-11-06"),
        (4, 6, 200.00, "2026-11-09"),
        (5, 8, 90.00, "2026-11-11"),
        (6, 10, 150.00, "2026-11-15"),
        (7, 12, 210.00, "2026-11-19"),
        (8, 2, 100.00, "2026-10-30"),
    ]
    conn.executemany("INSERT INTO refunds VALUES (?, ?, ?, ?)", refunds)

    events = [
        ("s1", 101, "login", 25.0, "2026-11-10T01:00:00Z"),
        ("s1", 101, "search", 64.0, "2026-11-10T01:01:00Z"),
        ("s1", 101, "checkout", 142.0, "2026-11-10T01:02:00Z"),
        ("s1", 101, "report", 210.0, "2026-11-10T01:03:00Z"),
        ("s2", 102, "login", 30.0, "2026-11-10T02:00:00Z"),
        ("s2", 102, "search", 80.0, "2026-11-10T02:01:00Z"),
        ("s3", 103, "login", 20.0, "2026-11-11T03:00:00Z"),
        ("s3", 103, "search", 50.0, "2026-11-11T03:01:00Z"),
        ("s3", 103, "checkout", 118.0, "2026-11-11T03:02:00Z"),
        ("s4", 101, "search", 70.0, "2026-11-12T04:00:00Z"),
        ("s4", 101, "checkout", 160.0, "2026-11-12T04:01:00Z"),
        ("s4", 101, "report", 190.0, "2026-11-12T04:02:00Z"),
        ("s5", 104, "login", 45.0, "2026-11-13T05:00:00Z"),
        ("s5", 104, "search", 55.0, "2026-11-13T05:01:00Z"),
        ("s5", 104, "search", 65.0, "2026-11-13T05:02:00Z"),
        ("s6", 105, "report", 300.0, "2026-11-14T06:00:00Z"),
    ]
    conn.executemany("INSERT INTO events VALUES (?, ?, ?, ?, ?)", events)

    catalog = [
        ("SKU-A1", "compute", 0.45),
        ("SKU-A2", "compute", 0.40),
        ("SKU-B1", "storage", 0.35),
        ("SKU-B2", "storage", 0.30),
        ("SKU-C1", "network", 0.50),
        ("SKU-D1", "support", 0.70),
    ]
    conn.executemany("INSERT INTO catalog VALUES (?, ?, ?)", catalog)

    line_items = [
        (3, "SKU-A1", 2, 180.0),
        (3, "SKU-B1", 1, 140.0),
        (4, "SKU-A2", 3, 250.0),
        (5, "SKU-C1", 4, 155.0),
        (6, "SKU-D1", 2, 140.0),
        (7, "SKU-A1", 5, 180.0),
        (8, "SKU-B2", 2, 205.0),
        (9, "SKU-C1", 3, 176.67),
        (11, "SKU-A2", 2, 345.0),
        (13, "SKU-D1", 4, 220.0),
        (14, "SKU-B1", 3, 156.67),
    ]
    conn.executemany("INSERT INTO line_items VALUES (?, ?, ?, ?)", line_items)

    deployments = [
        ("auth-api", "r101", "success", "2026-11-08", 42.0),
        ("auth-api", "r102", "failed", "2026-11-10", 120.0),
        ("auth-api", "r103", "success", "2026-11-12", 40.0),
        ("billing", "r201", "failed", "2026-11-11", 95.0),
        ("billing", "r202", "failed", "2026-11-13", 105.0),
        ("billing", "r203", "success", "2026-11-15", 50.0),
        ("router", "r301", "success", "2026-11-11", 30.0),
        ("router", "r302", "success", "2026-11-14", 36.0),
        ("worker", "r401", "failed", "2026-11-12", 80.0),
        ("worker", "r402", "success", "2026-11-16", 60.0),
        ("worker", "r403", "success", "2026-11-19", 64.0),
    ]
    conn.executemany("INSERT INTO deployments VALUES (?, ?, ?, ?, ?)", deployments)
    conn.commit()
    return conn


def _normalize_cell(val: object) -> object:
    if isinstance(val, float):
        return round(val, 2)
    if isinstance(val, int) and not isinstance(val, bool):
        return float(val) if False else val
    return val


def _normalize_rows(rows: list[tuple[object, ...]], *, ordered: bool) -> list[tuple[object, ...]]:
    norm = []
    for row in rows:
        out_row = []
        for cell in row:
            if isinstance(cell, (int, float)) and not isinstance(cell, bool):
                # Normalize numeric scalars consistently when SQL returns int vs float sums
                rounded = round(float(cell), 2)
                out_row.append(int(rounded) if rounded.is_integer() else rounded)
            else:
                out_row.append(cell)
        norm.append(tuple(out_row))
    if not ordered:
        norm.sort(key=lambda r: tuple(str(x) for x in r))
    return norm


def _readonly_authorizer(action: int, arg1: object, arg2: object, dbname: object, source: object) -> int:
    allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}
    return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY


def execute_readonly_sql(conn: sqlite3.Connection, sql: str) -> list[tuple[object, ...]]:
    stmt = (sql or "").strip().rstrip(";")
    if not stmt or ";" in stmt:
        raise ValueError("Empty or multi-statement SQL rejected")
    deadline = time.monotonic() + 0.5
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    conn.set_authorizer(_readonly_authorizer)
    try:
        cur = conn.execute(stmt)
        rows = cur.fetchall()
    finally:
        conn.set_authorizer(None)
        conn.set_progress_handler(None, 0)
    return rows


def build_sql_ground_truth(conn: sqlite3.Connection) -> dict[str, list[tuple[object, ...]]]:
    truth: dict[str, list[tuple[object, ...]]] = {}
    for case in SQL_CASES:
        rows = execute_readonly_sql(conn, case["reference_sql"])
        if not rows:
            raise RuntimeError(f"Reference SQL case {case['id']} returned empty rows")
        truth[case["id"]] = _normalize_rows(rows, ordered=bool(case["ordered"]))
    return truth


def detect_model(base_url: str, timeout: float = 10.0) -> str:
    url = base_url.rstrip("/") + "/v1/models"
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    models = data.get("data") or []
    if not models or not models[0].get("id"):
        raise RuntimeError(f"Could not detect model from {url}: {data}")
    return str(models[0]["id"])


def fetch_server_snapshot(base_url: str, timeout: float = 10.0) -> dict:
    base = base_url.rstrip("/")
    info: dict = {}
    for path in ("/get_server_info", "/v1/models"):
        try:
            with urllib.request.urlopen(base + path, timeout=timeout) as resp:
                info[path] = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            info[path] = {"error": repr(exc)}

    metrics_summary: dict[str, float] = {}
    try:
        with urllib.request.urlopen(base + "/metrics", timeout=timeout) as resp:
            text = resp.read().decode("utf-8", "replace")
        for line in text.splitlines():
            if line.startswith("#"):
                continue
            for key in (
                "sglang:num_running_reqs",
                "sglang:num_queue_reqs",
                "sglang:spec_verify_ct",
                "sglang:spec_accepted_tokens",
                "sglang:spec_draft_tokens",
                "sglang:cache_hit_rate",
                "sglang:token_usage",
                "sglang:max_total_num_tokens",
            ):
                if line.startswith(key):
                    match = re.search(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$", line)
                    if match:
                        metrics_summary[key] = float(match.group(1))
    except Exception as exc:
        metrics_summary["error"] = -1.0
        info["metrics_error"] = repr(exc)

    srv = info.get("/get_server_info") if isinstance(info.get("/get_server_info"), dict) else {}
    return {
        "max_total_num_tokens": srv.get("max_total_num_tokens") or metrics_summary.get("sglang:max_total_num_tokens"),
        "max_prefill_tokens": srv.get("max_prefill_tokens"),
        "chunked_prefill_size": srv.get("chunked_prefill_size"),
        "speculative_algorithm": srv.get("speculative_algorithm"),
        "speculative_num_draft_tokens": srv.get("speculative_num_draft_tokens"),
        "metrics": metrics_summary,
    }


class PowerTelemetrySampler(threading.Thread):
    """Samples GPU power draw (W) and used VRAM (MiB) via nvidia-smi (local or custom cmd)."""

    def __init__(self, interval_s: float = 0.25, query_cmd: str | None = None) -> None:
        super().__init__(daemon=True)
        self.interval_s = interval_s
        if query_cmd:
            self.cmd = shlex.split(query_cmd)
        else:
            self.cmd = [
                "nvidia-smi",
                "--query-gpu=power.draw,power.limit,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ]
        self.samples: list[tuple[float, float, float, float, float]] = []
        self.available = True
        self._stop_event = threading.Event()

    def _poll_once(self) -> tuple[float, float, float, float, float] | None:
        try:
            proc = subprocess.run(self.cmd, capture_output=True, text=True, timeout=3.0)
            if proc.returncode != 0 or not proc.stdout.strip():
                self.available = False
                return None
            parts = [float(x.strip()) for x in proc.stdout.strip().splitlines()[0].split(",")]
            if len(parts) < 4:
                return None
            return (time.monotonic(), parts[0], parts[1], parts[2], parts[3])
        except Exception:
            self.available = False
            return None

    def run(self) -> None:
        first = self._poll_once()
        if first is not None:
            self.samples.append(first)
        while not self._stop_event.wait(self.interval_s):
            if not self.available:
                break
            row = self._poll_once()
            if row is not None:
                self.samples.append(row)

    def stop_and_summarize(self, total_completion_tokens: int, wall_s: float) -> dict:
        self._stop_event.set()
        self.join(timeout=2.0)
        if not self.samples:
            return {"available": False}
        powers = [s[1] for s in self.samples]
        limits = [s[2] for s in self.samples]
        vrams = [s[3] for s in self.samples]
        utils = [s[4] for s in self.samples]
        avg_w = statistics.mean(powers)
        energy_j = avg_w * wall_s
        tok_per_j = (total_completion_tokens / energy_j) if energy_j > 0 else 0.0
        return {
            "available": True,
            "sample_count": len(self.samples),
            "avg_power_w": round(avg_w, 2),
            "peak_power_w": round(max(powers), 2),
            "power_limit_w": round(statistics.median(limits), 2),
            "peak_vram_mib": round(max(vrams), 1),
            "avg_gpu_util_pct": round(statistics.mean(utils), 1),
            "energy_j": round(energy_j, 2),
            "tok_per_j": round(tok_per_j, 4),
        }


def run_one_request(
    *,
    url: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    response_format: dict | None,
    enable_thinking: bool,
    timeout: float,
) -> dict:
    body: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if temperature > 0:
        body["top_p"] = 0.95
    if response_format is not None:
        body["response_format"] = response_format

    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    t_first: float | None = None
    chunk_times: list[float] = []
    pieces: list[str] = []
    usage: dict = {}
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if ev.get("error"):
                raise RuntimeError(f"SSE error: {ev['error']}")
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                delta = ch.get("delta") or {}
                piece = (delta.get("content") or "") + (delta.get("reasoning_content") or delta.get("reasoning") or "")
                if piece:
                    now = time.monotonic()
                    if t_first is None:
                        t_first = now
                    else:
                        chunk_times.append(now)
                    pieces.append(piece)
    t_end = time.monotonic()
    if t_first is None:
        t_first = t_end

    text = "".join(pieces)
    ct = int(usage.get("completion_tokens") or 0)
    pt = int(usage.get("prompt_tokens") or 0)
    decode_tps = (ct - 1) / (t_end - t_first) if ct > 1 and t_end > t_first else None
    all_times = [t_first] + chunk_times
    itls_ms = [(b - a) * 1000.0 for a, b in zip(all_times, all_times[1:])]
    return {
        "t0": t0,
        "t_end": t_end,
        "ttft_ms": round((t_first - t0) * 1000.0, 2),
        "wall_ms": round((t_end - t0) * 1000.0, 2),
        "decode_tok_s": round(decode_tps, 2) if decode_tps is not None else None,
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "itls_ms": itls_ms,
        "text": text,
    }


def _validate_workload_output(
    workload: str,
    text: str,
    meta: dict,
    db_conn: sqlite3.Connection | None,
    sql_truth: dict[str, list[tuple[object, ...]]] | None,
) -> bool:
    if workload == "eng":
        return len(text.strip()) > 0
    try:
        obj = json.loads(text)
    except ValueError:
        return False
    if workload == "json":
        return (
            isinstance(obj, dict)
            and set(obj.keys()) == {"route", "needs_db", "confidence", "entities", "rationale"}
            and obj["route"] == meta["route"]
            and obj["needs_db"] is meta["needs_db"]
            and isinstance(obj["entities"], list)
        )
    if workload == "sql":
        if not (
            isinstance(obj, dict)
            and set(obj.keys()) == {"dialect", "tables_used", "sql"}
            and obj["dialect"] == "sqlite"
            and isinstance(obj["tables_used"], list)
            and isinstance(obj["sql"], str)
        ):
            return False
        if db_conn is None or sql_truth is None:
            return "select" in obj["sql"].lower()
        try:
            actual_rows = execute_readonly_sql(db_conn, obj["sql"])
        except Exception:
            return False
        actual_norm = _normalize_rows(actual_rows, ordered=bool(meta["ordered"]))
        return actual_norm == sql_truth[meta["id"]]
    return True


def build_workload_spec(workload: str, index: int, run_nonce: str) -> tuple[str, dict | None, dict]:
    salt = f"[bench {run_nonce}-{index}] "
    if workload == "eng":
        return salt + ENGLISH_PROMPTS[index % len(ENGLISH_PROMPTS)], None, {}
    if workload == "json":
        case = JSON_CASES[index % len(JSON_CASES)]
        prompt = (
            f"{salt}Classify the request into strict JSON (`route`, `needs_db`, `confidence`, `entities`, `rationale`). "
            "Rules: use route='sql_query' and needs_db=true for structured ledger/orders/refunds/cohorts/deployments SQL analytics queries; "
            "use route='ops_trace' and needs_db=false for live runtime HTTP 5xx/latency/GPU VRAM incident diagnostics; "
            "use route='knowledge_base' and needs_db=false for written policy docs, SLAs, and migration changelogs. "
            f"Request: {case['text']}"
        )
        return prompt, JSON_SCHEMA, case
    if workload == "sql":
        case = SQL_CASES[index % len(SQL_CASES)]
        prompt = (
            f"{salt}{case['prompt']} "
            "Return strict JSON matching `dialect` ('sqlite'), `tables_used`, and `sql`."
        )
        return prompt, SQL_JSON_SCHEMA, case
    raise ValueError(f"Unknown workload {workload}")


def run_benchmark_cell(
    *,
    url: str,
    model: str,
    workload: str,
    concurrency: int,
    temperature: float,
    reps: int,
    max_tokens: int,
    enable_thinking: bool,
    timeout: float,
    run_nonce: str,
    power_cmd: str | None,
    db_conn: sqlite3.Connection | None,
    sql_truth: dict[str, list[tuple[object, ...]]] | None,
) -> dict:
    # Single warmup outside timed window
    warmup_prompt, warmup_rf, _ = build_workload_spec(workload, 0, f"{run_nonce}-warm")
    run_one_request(
        url=url,
        model=model,
        prompt=warmup_prompt,
        temperature=temperature,
        max_tokens=min(32, max_tokens),
        response_format=warmup_rf,
        enable_thinking=enable_thinking,
        timeout=timeout,
    )

    before_metrics = fetch_server_snapshot(url.rsplit("/v1/", 1)[0], timeout=timeout).get("metrics", {})
    sampler = PowerTelemetrySampler(interval_s=0.25, query_cmd=power_cmd)
    sampler.start()

    results: list[dict] = []
    if concurrency == 1:
        for i in range(reps):
            prompt, rf, meta = build_workload_spec(workload, i, f"{run_nonce}-r{i}")
            res = run_one_request(
                url=url,
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=rf,
                enable_thinking=enable_thinking,
                timeout=timeout,
            )
            res["valid"] = _validate_workload_output(workload, res["text"], meta, db_conn, sql_truth)
            results.append(res)
    else:
        for batch_idx in range(reps):
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futs = []
                for slot in range(concurrency):
                    idx = batch_idx * concurrency + slot
                    prompt, rf, meta = build_workload_spec(workload, idx, f"{run_nonce}-b{batch_idx}-s{slot}")
                    futs.append((
                        meta,
                        pool.submit(
                            run_one_request,
                            url=url,
                            model=model,
                            prompt=prompt,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            response_format=rf,
                            enable_thinking=enable_thinking,
                            timeout=timeout,
                        ),
                    ))
                for meta, fut in futs:
                    res = fut.result()
                    res["valid"] = _validate_workload_output(workload, res["text"], meta, db_conn, sql_truth)
                    results.append(res)

    wall_s = max(r["t_end"] for r in results) - min(r["t0"] for r in results)
    total_ct = sum(r["completion_tokens"] for r in results)
    valid_count = sum(1 for r in results if r["valid"])
    power_summary = sampler.stop_and_summarize(total_ct, wall_s)
    after_metrics = fetch_server_snapshot(url.rsplit("/v1/", 1)[0], timeout=timeout).get("metrics", {})

    decodes = [r["decode_tok_s"] for r in results if r["decode_tok_s"] is not None]
    ttfts = [r["ttft_ms"] for r in results]
    all_itls = sorted(x for r in results for x in r["itls_ms"])
    agg_tok_s = round(total_ct / wall_s, 2) if wall_s > 0 else 0.0
    valid_tasks_per_s = round(valid_count / wall_s, 3) if wall_s > 0 else 0.0
    valid_rate_pct = round((valid_count / len(results)) * 100.0, 2) if results else 0.0

    accept_len = None
    d_accept = after_metrics.get("sglang:spec_accepted_tokens", 0.0) - before_metrics.get("sglang:spec_accepted_tokens", 0.0)
    d_verify = after_metrics.get("sglang:spec_verify_ct", 0.0) - before_metrics.get("sglang:spec_verify_ct", 0.0)
    if d_verify > 0:
        accept_len = round(d_accept / d_verify, 3)

    return {
        "workload": workload,
        "concurrency": concurrency,
        "temperature": temperature,
        "requests": len(results),
        "valid_count": valid_count,
        "valid_rate_pct": valid_rate_pct,
        "valid_tasks_per_s": valid_tasks_per_s,
        "total_completion_tokens": total_ct,
        "wall_s": round(wall_s, 3),
        "agg_tok_s": agg_tok_s,
        "per_stream_median_tok_s": round(statistics.median(decodes), 2) if decodes else None,
        "ttft_p50_ms": round(statistics.median(ttfts), 2),
        "ttft_max_ms": round(max(ttfts), 2),
        "itl_p99_ms": round(all_itls[(99 * len(all_itls)) // 100], 2) if all_itls else None,
        "itl_max_ms": round(all_itls[-1], 2) if all_itls else None,
        "spec_accept_len": accept_len,
        "telemetry": power_summary,
    }


def cmd_run(args: argparse.Namespace) -> None:
    base = args.base_url.rstrip("/")
    snap = fetch_server_snapshot(base, timeout=args.timeout)
    running = snap.get("metrics", {}).get("sglang:num_running_reqs", 0.0)
    queued = snap.get("metrics", {}).get("sglang:num_queue_reqs", 0.0)
    if not args.force and ((running or 0.0) > 0 or (queued or 0.0) > 0):
        raise SystemExit(f"Aborting: server has active load (running={running}, queued={queued}); pass --force to override.")

    db_conn = seed_analytics_db()
    sql_truth = build_sql_ground_truth(db_conn)

    model = args.model or detect_model(base, timeout=args.timeout)
    workloads = [w.strip() for w in args.workloads.split(",") if w.strip()]
    concurrencies = [int(c.strip()) for c in args.concurrencies.split(",") if c.strip()]
    temps = [float(t.strip()) for t in args.temps.split(",") if t.strip()]
    run_nonce = uuid.uuid4().hex[:10]

    cells = []
    for c in concurrencies:
        reps = args.c1_reps if c == 1 else args.cn_reps
        for temp in temps:
            for w in workloads:
                cell = run_benchmark_cell(
                    url=base + "/v1/chat/completions",
                    model=model,
                    workload=w,
                    concurrency=c,
                    temperature=temp,
                    reps=reps,
                    max_tokens=args.max_tokens,
                    enable_thinking=args.enable_thinking,
                    timeout=args.timeout,
                    run_nonce=f"{run_nonce}-c{c}-t{temp}-{w}",
                    power_cmd=args.power_cmd,
                    db_conn=db_conn,
                    sql_truth=sql_truth,
                )
                cells.append(cell)
                print(
                    f"[{args.arm}] c={c} temp={temp} {w:4s} | "
                    f"agg={cell['agg_tok_s']:7.2f} tok/s | valid={cell['valid_count']}/{cell['requests']} "
                    f"({cell['valid_rate_pct']:5.1f}%, {cell['valid_tasks_per_s']:.3f} tasks/s) | "
                    f"p50_stream={cell['per_stream_median_tok_s']} | ttft_p50={cell['ttft_p50_ms']:6.1f}ms | "
                    f"tok/J={cell['telemetry'].get('tok_per_j')}",
                    file=sys.stderr,
                    flush=True,
                )

    report = {
        "arm": args.arm,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "base_url": base,
        "model": model,
        "max_tokens": args.max_tokens,
        "enable_thinking": args.enable_thinking,
        "sql_cases": len(SQL_CASES),
        "json_cases": len(JSON_CASES),
        "server_snapshot": snap,
        "cells": cells,
    }
    encoded = json.dumps(report, indent=2)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


def cmd_compare(args: argparse.Namespace) -> None:
    reports = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.files]
    if len(reports) < 2:
        raise SystemExit("Provide at least two JSON report files to compare.")

    by_arm = {
        r["arm"]: {
            (c["concurrency"], c["temperature"], c["workload"]): c for c in r["cells"]
        }
        for r in reports
    }
    arm_names = [r["arm"] for r in reports]
    keys = sorted({k for mapping in by_arm.values() for k in mapping})
    comparisons = []
    for k in keys:
        c, temp, w = k
        entry = {"concurrency": c, "temperature": temp, "workload": w, "arms": {}}
        for name in arm_names:
            cell = by_arm[name].get(k)
            if cell:
                valid_rate_pct = cell.get(
                    "valid_rate_pct",
                    round((cell["valid_count"] / cell["requests"]) * 100.0, 2) if cell["requests"] else 0.0,
                )
                valid_tasks_per_s = cell.get(
                    "valid_tasks_per_s",
                    round(cell["valid_count"] / cell["wall_s"], 3) if cell.get("wall_s", 0) > 0 else 0.0,
                )
                entry["arms"][name] = {
                    "agg_tok_s": cell["agg_tok_s"],
                    "per_stream_median_tok_s": cell["per_stream_median_tok_s"],
                    "ttft_p50_ms": cell["ttft_p50_ms"],
                    "tok_per_j": cell["telemetry"].get("tok_per_j"),
                    "valid": f"{cell['valid_count']}/{cell['requests']}",
                    "valid_rate_pct": valid_rate_pct,
                    "valid_tasks_per_s": valid_tasks_per_s,
                }
        if len(arm_names) >= 2 and all(k in by_arm[n] for n in arm_names):
            cand = by_arm[arm_names[1]][k]["agg_tok_s"]
            controls = [by_arm[n][k]["agg_tok_s"] for i, n in enumerate(arm_names) if i != 1]
            best_ctrl = max(controls)
            entry["candidate_vs_best_control_pct"] = round((cand / best_ctrl - 1.0) * 100.0, 2) if best_ctrl > 0 else None

            cand_vt = entry["arms"][arm_names[1]]["valid_tasks_per_s"]
            ctrl_vts = [entry["arms"][n]["valid_tasks_per_s"] for i, n in enumerate(arm_names) if i != 1]
            best_ctrl_vt = max(ctrl_vts)
            entry["candidate_valid_tasks_vs_best_control_pct"] = (
                round((cand_vt / best_ctrl_vt - 1.0) * 100.0, 2) if best_ctrl_vt > 0 else None
            )
        comparisons.append(entry)

    out = {
        "arms": [
            {
                "arm": r["arm"],
                "max_total_num_tokens": r.get("server_snapshot", {}).get("max_total_num_tokens"),
                "spec_algo": r.get("server_snapshot", {}).get("speculative_algorithm"),
                "draft_tokens": r.get("server_snapshot", {}).get("speculative_num_draft_tokens"),
            }
            for r in reports
        ],
        "comparisons": comparisons,
    }
    print(json.dumps(out, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run E2E benchmark suite against a live server")
    p_run.add_argument("--arm", default="candidate")
    p_run.add_argument("--base-url", default="http://127.0.0.1:30002")
    p_run.add_argument("--model", default=None)
    p_run.add_argument("--workloads", default="json,sql,eng")
    p_run.add_argument("--concurrencies", default="1,8")
    p_run.add_argument("--temps", default="0.0,0.6")
    p_run.add_argument("--c1-reps", type=int, default=4)
    p_run.add_argument("--cn-reps", type=int, default=1)
    p_run.add_argument("--max-tokens", type=int, default=512)
    p_run.add_argument("--enable-thinking", action="store_true")
    p_run.add_argument("--timeout", type=float, default=180.0)
    p_run.add_argument("--power-cmd", default=None, help="Optional shell command returning nvidia-smi CSV")
    p_run.add_argument("--force", action="store_true")
    p_run.add_argument("--out", default=None)

    p_cmp = sub.add_parser("compare", help="Compare two or three saved run JSON files (e.g., A1 B A2)")
    p_cmp.add_argument("files", nargs="+")

    args = parser.parse_args()
    if args.command == "run":
        cmd_run(args)
    else:
        cmd_compare(args)


if __name__ == "__main__":
    main()
