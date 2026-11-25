#!/usr/bin/env python3
"""Reproducible warm-prefix 2-turn tool-calling benchmark (greedy and sampled).

Compares volatile-first vs catalog-first prompt layouts across repeated warm-prefix
trajectories (Turn 1: tool call with strict JSON schema; Turn 2: final structured
answer from tool output) across temperatures such as 0.0 and 0.6.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import statistics
import sys
import time
import urllib.request
import uuid


ARMS = ("volatile_first", "catalog_first")
PHASES = ("call", "answer")
MAX_TOKENS = 96

SYSTEM_PROMPT = (
    "You are an automated tool-routing assistant. Each request carries a JSON workflow "
    "envelope. Object key order carries no semantic weight. Follow the task instructions "
    "and emit only valid JSON matching the requested schema. Ground final answers strictly "
    "in the provided tool_result."
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
    for entity in (
        "account", "invoice", "order", "shipment", "product", "vendor",
        "customer", "ticket", "refund", "warehouse", "inventory", "transfer",
        "payment", "contract", "allocation", "budget", "campaign", "subscription",
    )
]


def _object_schema(properties: dict) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


CALL_SCHEMA = _object_schema({
    "tool": {"type": "string", "enum": [tool["name"] for tool in TOOL_CATALOG[:4]]},
    "args": _object_schema({"id": {"type": "integer"}}),
})
ANSWER_SCHEMA = _object_schema({"value": {"type": "integer"}})


def _compact_json(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True)


def detect_model(base_url: str, timeout: float = 10.0) -> str:
    url = base_url.rstrip("/") + "/v1/models"
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    models = data.get("data") or []
    if not models or not models[0].get("id"):
        raise RuntimeError(f"Unable to auto-detect model from {url}: {data}")
    return str(models[0]["id"])


def build_payload(
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
    fields: list[tuple[str, object]] = [("nonce", nonce), ("catalog", TOOL_CATALOG)]
    if arm == "catalog_first":
        fields.reverse()
    envelope = dict(fields)
    if phase == "call":
        envelope["task"] = f"Call {tool_name} with id {record_id}."
        schema = CALL_SCHEMA
        expected = {"tool": tool_name, "args": {"id": record_id}}
    else:
        envelope["task"] = "Read tool_result and return its integer value."
        envelope["tool_result"] = {
            "tool": tool_name,
            "id": record_id,
            "value": record_value,
        }
        schema = ANSWER_SCHEMA
        expected = {"value": record_value}

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": f"Benchmark namespace {namespace}. {SYSTEM_PROMPT}"},
            {"role": "user", "content": _compact_json(envelope)},
        ],
        "temperature": temperature,
        "max_tokens": MAX_TOKENS,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
        "stream": True,
        "stream_options": {"include_usage": True},
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": phase, "strict": True, "schema": schema},
        },
    }
    if temperature > 0:
        payload["top_p"] = 0.95
    return payload, expected


def _validate_schema(actual: object, phase: str) -> bool:
    if not isinstance(actual, dict):
        return False
    if phase == "call":
        return (
            set(actual.keys()) == {"tool", "args"}
            and actual["tool"] in {tool["name"] for tool in TOOL_CATALOG[:4]}
            and isinstance(actual["args"], dict)
            and set(actual["args"].keys()) == {"id"}
            and isinstance(actual["args"]["id"], int)
            and not isinstance(actual["args"]["id"], bool)
        )
    return (
        set(actual.keys()) == {"value"}
        and isinstance(actual["value"], int)
        and not isinstance(actual["value"], bool)
    )


def execute_step(url: str, timeout: float, payload: dict, expected: dict, phase: str) -> dict:
    request = urllib.request.Request(
        url,
        data=_compact_json(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    frames: list[float] = []
    pieces: list[str] = []
    usage: dict = {}
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            if event.get("error"):
                raise RuntimeError(f"SSE error: {event['error']}")
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices") or []:
                piece = choice.get("delta", {}).get("content")
                if piece:
                    frames.append(time.perf_counter() - start)
                    pieces.append(piece)
    duration = time.perf_counter() - start
    raw_text = "".join(pieces)
    try:
        actual = json.loads(raw_text)
        schema_valid = _validate_schema(actual, phase)
        exact_match = json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)
    except ValueError:
        actual = None
        schema_valid = False
        exact_match = False

    details = usage.get("prompt_tokens_details") or {}
    cached_tokens = details.get("cached_tokens", usage.get("cached_tokens"))
    completion_tokens = usage.get("completion_tokens")
    decode_tps = None
    if completion_tokens and completion_tokens > 1 and frames and duration > frames[0]:
        decode_tps = round((completion_tokens - 1) / (duration - frames[0]), 2)
    return {
        "schema_valid": schema_valid,
        "exact_match": exact_match,
        "actual": actual,
        "ttft_ms": round(frames[0] * 1000, 3) if frames else None,
        "frame_gaps_ms": [round((b - a) * 1000, 3) for a, b in zip(frames, frames[1:])],
        "wall_ms": round(duration * 1000, 3),
        "decode_tok_s": decode_tps,
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": cached_tokens,
        "completion_tokens": completion_tokens,
    }


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
    start = time.perf_counter()
    phases: dict[str, dict] = {}
    for index, phase in enumerate(PHASES):
        nonce = f"{iteration:02d}-{slot:02d}-{index:02d}"
        payload, expected = build_payload(
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
    return {
        "temperature": temperature,
        "pair": pair,
        "arm": arm,
        "iteration": iteration,
        "slot": slot,
        "cache_phase": "initial_group" if iteration == 0 else "subsequent_group",
        "wall_ms": round((time.perf_counter() - start) * 1000, 3),
        "phases": phases,
    }


def _p95(values: list[float]) -> float:
    return round(sorted(values)[(95 * len(values) + 99) // 100 - 1], 3)


def summarize(trials: list[dict], temperatures: list[float]) -> dict:
    summary: dict[str, dict] = {}
    for temp in temperatures:
        t_key = f"temp_{temp}"
        summary[t_key] = {}
        for arm in ARMS:
            summary[t_key][arm] = {}
            for group in ("initial_group", "subsequent_group"):
                subset = [
                    row for row in trials
                    if row["temperature"] == temp and row["arm"] == arm and row["cache_phase"] == group
                ]
                exact_count = sum(
                    all(p["exact_match"] for p in t["phases"].values()) for t in subset
                )
                total_wall_s = sum(t["wall_ms"] for t in subset) / 1000.0
                row = {
                    "trajectories": len(subset),
                    "schema_valid_trajectories": sum(
                        all(p["schema_valid"] for p in t["phases"].values()) for t in subset
                    ),
                    "exact_match_trajectories": exact_count,
                    "valid_tasks_per_s": round(exact_count / total_wall_s, 3) if total_wall_s > 0 else 0.0,
                    "trajectory_p50_ms": round(statistics.median(t["wall_ms"] for t in subset), 3),
                    "trajectory_p95_ms": _p95([t["wall_ms"] for t in subset]),
                }
                for phase in PHASES:
                    steps = [t["phases"][phase] for t in subset]
                    row[phase] = {}
                    for metric in ("ttft_ms", "decode_tok_s", "cached_tokens", "prompt_tokens", "completion_tokens"):
                        known = [s[metric] for s in steps if s[metric] is not None]
                        row[phase][f"{metric}_p50"] = round(statistics.median(known), 3) if known else None
                    frame_gaps = [gap for s in steps for gap in s["frame_gaps_ms"]]
                    row[phase]["sse_frame_gap_p50_ms"] = round(statistics.median(frame_gaps), 3) if frame_gaps else None
                summary[t_key][arm][group] = row
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
            for grp in ("initial_group", "subsequent_group"):
                entry: dict[str, object] = {}
                for lbl, rep in zip(labels, reports):
                    grp_data = ((rep.get("summary") or {}).get(temp, {}).get(pfx, {}).get(grp))
                    if grp_data:
                        entry[lbl] = {
                            "trajectory_p50_ms": grp_data["trajectory_p50_ms"],
                            "trajectory_p95_ms": grp_data["trajectory_p95_ms"],
                            "schema_valid": f"{grp_data['schema_valid_trajectories']}/{grp_data['trajectories']}",
                            "exact_match": f"{grp_data['exact_match_trajectories']}/{grp_data['trajectories']}",
                            "valid_tasks_per_s": grp_data.get("valid_tasks_per_s"),
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

    report = {
        "config": {
            "base_url": args.base_url,
            "model": args.model,
            "temperatures": temps,
            "pairs": args.pairs,
            "iterations": args.iterations,
            "concurrency": args.concurrency,
            "seed": args.seed,
            "run_id": args.run_id,
            "enable_thinking": args.enable_thinking,
            "catalog_bytes": len(_compact_json(TOOL_CATALOG)),
            "requests": len(trials) * len(PHASES),
            "dry_run": args.dry_run,
        },
        "arm_orders": arm_orders,
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
