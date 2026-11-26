#!/usr/bin/env python3
"""Reproducible warm-prefix 2-turn tool-calling benchmark with radix cache flushing.

Evaluates volatile-first vs catalog-first prompt envelopes over paired 2-turn
trajectories (`call` with strict JSON schema -> `answer` grounded in tool output),
reporting `initial_group`, `subsequent_group`, and combined `all_groups` metrics.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ARMS = ("volatile_first", "catalog_first")
PHASES = ("call", "answer")
GROUPS = ("initial_group", "subsequent_group", "all_groups")
MAX_TOKENS = 96

SYSTEM_INSTRUCTIONS = (
    "You are an automated tool-routing assistant. Each request carries a JSON workflow "
    "envelope. Object key order carries no semantic weight. Follow the task instructions "
    "and emit only valid JSON matching the requested schema. Ground final answers strictly "
    "in the provided tool_result."
)

ENTITIES = (
    "account", "invoice", "order", "shipment",
    "product", "vendor", "customer", "ticket",
    "refund", "warehouse", "inventory", "transfer",
    "payment", "contract", "allocation", "budget",
    "campaign", "subscription",
)

TOOL_CATALOG = [
    {
        "name": f"lookup_{entity}",
        "description": (
            f"Fetch a single {entity} record from the synthetic state store using its numeric ID. "
            "Use this read-only endpoint whenever the prompt specifies that record type. "
            "It returns the exact integer identifier and integer metric value without side effects."
        ),
        "arguments": {"id": "integer"},
        "result": {"id": "integer", "value": "integer"},
    }
    for entity in ENTITIES
]


def strict_schema(fields: dict) -> dict:
    """Build a closed JSON Schema object requiring every field."""
    return {
        "properties": fields,
        "required": list(fields),
        "additionalProperties": False,
        "type": "object",
    }


CALL_SCHEMA = strict_schema(
    {
        "tool": {"type": "string", "enum": [t["name"] for t in TOOL_CATALOG[:4]]},
        "args": strict_schema({"id": {"type": "integer"}}),
    }
)
ANSWER_SCHEMA = strict_schema({"value": {"type": "integer"}})


def encode_compact(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=True, separators=(",", ":"))


def detect_model(base_url: str, timeout: float = 10.0) -> str:
    endpoint = base_url.rstrip("/") + "/v1/models"
    with urllib.request.urlopen(endpoint, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    entries = payload.get("data") or []
    if not entries or not entries[0].get("id"):
        raise RuntimeError(f"Could not detect model at {endpoint}: {payload}")
    return str(entries[0]["id"])


def flush_cache(base_url: str, timeout: float = 10.0) -> None:
    """Reset the server radix prefix cache via POST /flush_cache."""
    req = urllib.request.Request(
        base_url.rstrip("/") + "/flush_cache",
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        resp.read()


def build_step_request(
    *,
    model: str,
    namespace: str,
    arm: str,
    nonce: str,
    phase: str,
    record_id: int,
    record_value: int,
    tool_name: str,
    temperature: float,
    enable_thinking: bool,
) -> tuple[dict, dict]:
    if arm == "catalog_first":
        body_map: dict[str, object] = {"catalog": TOOL_CATALOG, "nonce": nonce}
    else:
        body_map = {"nonce": nonce, "catalog": TOOL_CATALOG}

    if phase == "call":
        body_map["task"] = f"Call {tool_name} with id {record_id}."
        schema = CALL_SCHEMA
        expected = {"args": {"id": record_id}, "tool": tool_name}
    else:
        body_map["task"] = "Read tool_result and return its integer value."
        body_map["tool_result"] = {"id": record_id, "tool": tool_name, "value": record_value}
        schema = ANSWER_SCHEMA
        expected = {"value": record_value}

    req_body: dict[str, object] = {
        "model": model,
        "messages": [
            {"role": "system", "content": f"Benchmark namespace {namespace}. {SYSTEM_INSTRUCTIONS}"},
            {"role": "user", "content": encode_compact(body_map)},
        ],
        "temperature": temperature,
        "max_tokens": MAX_TOKENS,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": phase, "schema": schema, "strict": True},
        },
        "stream_options": {"include_usage": True},
        "stream": True,
    }
    if temperature > 0:
        req_body["top_p"] = 0.95
    return req_body, expected


def _is_valid_step_output(actual: object, phase: str) -> bool:
    if not isinstance(actual, dict):
        return False
    if phase == "call":
        args_obj = actual.get("args")
        return (
            set(actual.keys()) == {"tool", "args"}
            and actual["tool"] in {t["name"] for t in TOOL_CATALOG[:4]}
            and isinstance(args_obj, dict)
            and set(args_obj.keys()) == {"id"}
            and isinstance(args_obj["id"], int)
            and not isinstance(args_obj["id"], bool)
        )
    return (
        set(actual.keys()) == {"value"}
        and isinstance(actual["value"], int)
        and not isinstance(actual["value"], bool)
    )


def _read_sse_events(resp, t0: float) -> tuple[list[float], list[str], dict]:
    frames: list[float] = []
    chunks: list[str] = []
    usage: dict = {}
    for raw_line in resp:
        text_line = raw_line.decode("utf-8", errors="replace").strip()
        if not text_line.startswith("data:"):
            continue
        payload = text_line.removeprefix("data:").strip()
        if payload == "[DONE]":
            break
        ev = json.loads(payload)
        if ev.get("error"):
            raise RuntimeError(f"SSE error: {ev['error']}")
        if ev.get("usage"):
            usage = ev["usage"]
        for ch in ev.get("choices") or ():
            piece = (ch.get("delta") or {}).get("content")
            if piece:
                frames.append(time.perf_counter() - t0)
                chunks.append(piece)
    return frames, chunks, usage


def execute_step(url: str, timeout: float, payload: dict, expected: dict, phase: str) -> dict:
    req = urllib.request.Request(
        url,
        data=encode_compact(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        frames, chunks, usage = _read_sse_events(resp, t0)
    elapsed_s = time.perf_counter() - t0
    raw_out = "".join(chunks)

    try:
        parsed = json.loads(raw_out)
        schema_ok = _is_valid_step_output(parsed, phase)
        exact_ok = json.dumps(parsed, sort_keys=True) == json.dumps(expected, sort_keys=True)
    except ValueError:
        parsed = None
        schema_ok = False
        exact_ok = False

    pt_details = usage.get("prompt_tokens_details") or {}
    cached_tok = pt_details.get("cached_tokens", usage.get("cached_tokens"))
    comp_tok = usage.get("completion_tokens")
    decode_tps = (
        round((comp_tok - 1) / (elapsed_s - frames[0]), 2)
        if comp_tok and comp_tok > 1 and frames and elapsed_s > frames[0]
        else None
    )
    return {
        "schema_valid": schema_ok,
        "exact_match": exact_ok,
        "actual": parsed,
        "ttft_ms": round(frames[0] * 1000.0, 3) if frames else None,
        "frame_gaps_ms": [round((b - a) * 1000.0, 3) for a, b in zip(frames, frames[1:])],
        "wall_ms": round(elapsed_s * 1000.0, 3),
        "decode_tok_s": decode_tps,
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": cached_tok,
        "completion_tokens": comp_tok,
    }


def prewarm_schemas(args: argparse.Namespace, temps: list[float]) -> None:
    """Warm up both strict JSON grammar FSMs untimed before clearing cache."""
    for temp in temps:
        for idx, phase in enumerate(PHASES):
            payload, _ = build_step_request(
                model=args.model,
                namespace=f"prewarm-{args.run_id}-t{temp}-{phase}",
                arm="catalog_first",
                nonce=f"pw-{idx:02d}",
                phase=phase,
                record_id=101,
                record_value=2020,
                tool_name=TOOL_CATALOG[0]["name"],
                temperature=temp,
                enable_thinking=args.enable_thinking,
            )
            execute_step(args.url, args.timeout, payload, {"value": 2020}, phase)


def run_trajectory(
    args: argparse.Namespace,
    *,
    temperature: float,
    pair: int,
    arm: str,
    namespace: str,
    iteration: int,
    slot: int,
) -> dict:
    rng = random.Random(f"{args.seed}/{temperature}/{pair}/{iteration}/{slot}")
    tool = TOOL_CATALOG[rng.randrange(4)]["name"]
    record_id = rng.randrange(100, 1000)
    record_value = rng.randrange(1000, 9000)
    t0 = time.perf_counter()
    phases: dict[str, dict] = {}
    for index, phase in enumerate(PHASES):
        nonce = f"{iteration:02d}-{slot:02d}-{index:02d}"
        payload, expected = build_step_request(
            model=args.model,
            namespace=namespace,
            arm=arm,
            nonce=nonce,
            phase=phase,
            record_id=record_id,
            record_value=record_value,
            tool_name=tool,
            temperature=temperature,
            enable_thinking=args.enable_thinking,
        )
        if args.dry_run:
            phases[phase] = {
                "schema_valid": True,
                "exact_match": True,
                "expected": expected,
                "user_prompt_bytes": len(payload["messages"][-1]["content"]),
            }
        else:
            phases[phase] = execute_step(args.url, args.timeout, payload, expected, phase)
    cache_phase = "initial_group" if iteration == 0 else "subsequent_group"
    return {
        "arm": arm,
        "pair": pair,
        "temperature": temperature,
        "iteration": iteration,
        "slot": slot,
        "cache_phase": cache_phase,
        "wall_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        "phases": phases,
    }


def _p95(values: list[float]) -> float:
    return round(sorted(values)[(95 * len(values) + 99) // 100 - 1], 3)


def _summarize_subset(subset: list[dict]) -> dict:
    exact_count = sum(all(p["exact_match"] for p in t["phases"].values()) for t in subset)
    total_wall_s = sum(t["wall_ms"] for t in subset) / 1000.0
    cached_all = [
        p["cached_tokens"]
        for t in subset
        for p in t["phases"].values()
        if p.get("cached_tokens") is not None
    ]
    row: dict[str, object] = {
        "trajectories": len(subset),
        "schema_valid_trajectories": sum(
            all(p["schema_valid"] for p in t["phases"].values()) for t in subset
        ),
        "exact_match_trajectories": exact_count,
        "valid_tasks_per_s": round(exact_count / total_wall_s, 3) if total_wall_s > 0 else 0.0,
        "trajectory_p50_ms": round(statistics.median(t["wall_ms"] for t in subset), 3),
        "trajectory_p95_ms": _p95([t["wall_ms"] for t in subset]),
        "cached_tokens_p50": round(statistics.median(cached_all), 3) if cached_all else None,
    }
    for phase in PHASES:
        steps = [t["phases"][phase] for t in subset]
        row[phase] = {}
        for metric in ("ttft_ms", "decode_tok_s", "cached_tokens", "prompt_tokens", "completion_tokens"):
            known = [s[metric] for s in steps if s[metric] is not None]
            row[phase][f"{metric}_p50"] = round(statistics.median(known), 3) if known else None
        frame_gaps = [gap for s in steps for gap in s["frame_gaps_ms"]]
        row[phase]["sse_frame_gap_p50_ms"] = round(statistics.median(frame_gaps), 3) if frame_gaps else None
    return row


def summarize(trials: list[dict], temperatures: list[float]) -> dict:
    summary: dict[str, dict] = {}
    for temp in temperatures:
        t_key = f"temp_{temp}"
        summary[t_key] = {}
        for arm in ARMS:
            arm_rows = [row for row in trials if row["temperature"] == temp and row["arm"] == arm]
            summary[t_key][arm] = {
                "initial_group": _summarize_subset([r for r in arm_rows if r["cache_phase"] == "initial_group"]),
                "subsequent_group": _summarize_subset([r for r in arm_rows if r["cache_phase"] == "subsequent_group"]),
                "all_groups": _summarize_subset(arm_rows),
            }
    return summary


def compare_reports(paths: list[str], arm_labels: list[str] | None = None) -> dict:
    """Compare two or three saved agentic benchmark JSON reports (e.g., Control A1 -> Candidate B -> Control A2)."""
    reports = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    labels = arm_labels or [f"arm_{i}" for i in range(len(reports))]
    if len(labels) != len(reports):
        raise ValueError("Number of --arm-labels must match number of report files.")

    delta: dict[str, dict] = {}
    temps = sorted({k for r in reports for k in (r.get("summary") or {}).keys()})
    for temp in temps:
        delta[temp] = {}
        for pfx in ARMS:
            delta[temp][pfx] = {}
            for grp in GROUPS:
                entry: dict[str, object] = {}
                for lbl, rep in zip(labels, reports):
                    arm_summary = (rep.get("summary") or {}).get(temp, {}).get(pfx, {})
                    grp_data = arm_summary.get(grp)
                    if grp_data is None and grp == "all_groups":
                        # Synthesize combined view if older JSON lacked precomputed all_groups
                        init_g = arm_summary.get("initial_group")
                        sub_g = arm_summary.get("subsequent_group")
                        if init_g and sub_g:
                            entry[lbl] = {
                                "trajectory_p50_ms": round(
                                    (init_g["trajectory_p50_ms"] + sub_g["trajectory_p50_ms"]) / 2.0, 3
                                ),
                                "trajectory_p95_ms": max(init_g["trajectory_p95_ms"], sub_g["trajectory_p95_ms"]),
                            }
                        continue
                    if grp_data:
                        entry[lbl] = {
                            "trajectory_p50_ms": grp_data["trajectory_p50_ms"],
                            "trajectory_p95_ms": grp_data["trajectory_p95_ms"],
                            "schema_valid": f"{grp_data['schema_valid_trajectories']}/{grp_data['trajectories']}",
                            "exact_match": f"{grp_data['exact_match_trajectories']}/{grp_data['trajectories']}",
                            "valid_tasks_per_s": grp_data.get("valid_tasks_per_s"),
                            "cached_tokens_p50": grp_data.get("cached_tokens_p50"),
                            "call_decode_tok_s_p50": grp_data["call"]["decode_tok_s_p50"],
                            "answer_ttft_p50_ms": grp_data["answer"]["ttft_ms_p50"],
                            "answer_decode_tok_s_p50": grp_data["answer"]["decode_tok_s_p50"],
                        }
                if len(labels) >= 2 and all(lbl in entry for lbl in labels):
                    cand_ms = entry[labels[1]]["trajectory_p50_ms"]
                    ctrl_ms = min(entry[lbl]["trajectory_p50_ms"] for i, lbl in enumerate(labels) if i != 1)
                    entry["candidate_vs_best_control_speedup_x"] = round(ctrl_ms / cand_ms, 3) if cand_ms > 0 else None
                delta[temp][pfx][grp] = entry
    return {"arms": labels, "delta_summary": delta}


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        cmp_parser = argparse.ArgumentParser(description="Compare saved agentic benchmark JSONs (A/B or A/B/A)")
        cmp_parser.add_argument("files", nargs="+")
        cmp_parser.add_argument("--arm-labels", default=None, help="Comma-separated labels matching files")
        cmp_args = cmp_parser.parse_args(sys.argv[2:])
        labels = [x.strip() for x in cmp_args.arm_labels.split(",")] if cmp_args.arm_labels else None
        print(json.dumps(compare_reports(cmp_args.files, labels), indent=2))
        return

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30002")
    parser.add_argument("--model", default=None, help="Defaults to first model from /v1/models")
    parser.add_argument("--temps", default="0.0,0.6", help="Comma-separated temperatures (e.g. 0.0,0.6)")
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--concurrency", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1741)
    parser.add_argument("--run-id", default=uuid.uuid4().hex[:12])
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--no-flush-cache", action="store_true", help="Skip POST /flush_cache before arms")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    if args.pairs < 1 or args.iterations < 2 or args.iterations > 99 or args.timeout <= 0:
        parser.error("Use --pairs >= 1, 2 <= --iterations <= 99, and --timeout > 0.")

    temps = [float(x.strip()) for x in args.temps.split(",") if x.strip()]
    if not temps:
        parser.error("--temps must specify at least one temperature")

    args.url = args.base_url.rstrip("/") + "/v1/chat/completions"
    if not args.model:
        args.model = "dry-run-model" if args.dry_run else detect_model(args.base_url, args.timeout)

    if not args.dry_run:
        prewarm_schemas(args, temps)
        if not args.no_flush_cache:
            flush_cache(args.base_url, args.timeout)

    rng = random.Random(args.seed)
    trials: list[dict] = []
    arm_orders: dict[str, list[tuple[str, str]]] = {}

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        for temp in temps:
            swaps = [False, True] * (args.pairs // 2)
            if args.pairs % 2:
                swaps.append(bool(rng.randrange(2)))
            rng.shuffle(swaps)
            orders: list[tuple[str, str]] = []
            for pair, swap in enumerate(swaps):
                order = ARMS[::-1] if swap else ARMS
                orders.append(order)
                tags = ["A", "B"]
                rng.shuffle(tags)
                for arm, tag in zip(order, tags):
                    if not args.dry_run and not args.no_flush_cache:
                        flush_cache(args.base_url, args.timeout)
                    namespace = f"{tag}-{args.run_id}-t{int(round(temp * 100))}-{pair:04d}"
                    for iteration in range(args.iterations):
                        futures = [
                            executor.submit(
                                run_trajectory,
                                args,
                                temperature=temp,
                                pair=pair,
                                arm=arm,
                                namespace=namespace,
                                iteration=iteration,
                                slot=slot,
                            )
                            for slot in range(args.concurrency)
                        ]
                        trials.extend(f.result() for f in futures)
            arm_orders[f"temp_{temp}"] = orders

    config_summary = {
        "base_url": args.base_url,
        "catalog_bytes": len(encode_compact(TOOL_CATALOG)),
        "concurrency": args.concurrency,
        "dry_run": args.dry_run,
        "enable_thinking": args.enable_thinking,
        "flush_cache": not args.no_flush_cache,
        "iterations": args.iterations,
        "model": args.model,
        "pairs": args.pairs,
        "requests": len(trials) * len(PHASES),
        "run_id": args.run_id,
        "seed": args.seed,
        "temperatures": temps,
    }
    report = {
        "arm_orders": arm_orders,
        "config": config_summary,
        "trials": trials,
    }
    if not args.dry_run:
        report["summary"] = summarize(trials, temps)

    encoded = json.dumps(report, indent=2)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)

    if not args.dry_run and any(not p["schema_valid"] for t in trials for p in t["phases"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
