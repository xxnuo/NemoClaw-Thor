#!/usr/bin/env python3
"""Repeated fixed-corpus concurrency probe for the DS4 OpenAI endpoint."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import statistics
import time
import urllib.request


def make_prompt(target_bytes: int) -> str:
    parts = [
        "Read this deterministic archive, then output every decimal integer from "
        "1000001 through 1999999 inclusive, one per line, without explanation. "
        "Begin with 1000001.\n",
    ]
    index = 0
    while sum(len(part) for part in parts) < target_bytes:
        parts.append(
            f"record_{index:06d}: state=verified dependency=runtime "
            f"checksum={hashlib.sha256(str(index).encode()).hexdigest()[:16]}\n"
        )
        index += 1
    return "".join(parts)


def valid_integer_sequence(content: str) -> bool:
    lines = content.splitlines()
    tail = ""
    if content and not content.endswith("\n"):
        *lines, tail = lines
    return len(lines) >= 2 and all(
        line == str(1000001 + index) for index, line in enumerate(lines)
    ) and (not tail or str(1000001 + len(lines)).startswith(tail))


def request(
    endpoint: str,
    model: str,
    prompt: str,
    output: int,
    timeout: int,
    wave: int,
    slot: int,
) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": output,
        "reasoning_effort": "none",
    }
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = json.load(response)
        choice = body["choices"][0]
        content = choice["message"]["content"]
        finish_reason = choice.get("finish_reason")
        completion_tokens = int(body["usage"]["completion_tokens"])
        if not isinstance(content, str) or not content or completion_tokens <= 0:
            raise ValueError("incomplete chat completion response")
        if (
            completion_tokens != output
            or finish_reason != "length"
            or not valid_integer_sequence(content)
        ):
            return {
                "ok": False,
                "wave": wave,
                "slot": slot,
                "endpoint": endpoint,
                "wall_seconds": time.monotonic() - started,
                "completion_tokens": completion_tokens,
                "finish_reason": finish_reason,
                "content": content,
                "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
                "error_type": "IncompleteCompletion",
                "error": (
                    f"expected {output} tokens, finish_reason=length, and a valid "
                    f"integer sequence; got {completion_tokens}, {finish_reason}"
                ),
            }
    except Exception as error:
        return {
            "ok": False,
            "wave": wave,
            "slot": slot,
            "endpoint": endpoint,
            "wall_seconds": time.monotonic() - started,
            "error_type": type(error).__name__,
            "error": str(error)[:512],
        }
    return {
        "ok": True,
        "wave": wave,
        "slot": slot,
        "endpoint": endpoint,
        "wall_seconds": time.monotonic() - started,
        "prompt_tokens": int(body.get("usage", {}).get("prompt_tokens", 0)),
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
        "prefill_tok_s": float(body.get("timings", {}).get("prefill_tok_s", 0)),
        "decode_tok_s": float(body.get("timings", {}).get("decode_tok_s", 0)),
        "spec_accept_rate": body.get("timings", {}).get("spec_accept_rate"),
        "tok_per_step": body.get("timings", {}).get("tok_per_step"),
        "content": content,
        "content_sha256": hashlib.sha256(
            content.encode()
        ).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", action="append")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--waves", type=int, default=3)
    parser.add_argument("--prompt-kib", type=int, default=72)
    parser.add_argument("--output", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    base_urls = args.base_url or ["http://127.0.0.1:8050/v1"]
    endpoints = [f"{base_url.rstrip('/')}/chat/completions" for base_url in base_urls]
    if args.concurrency < 1 or args.waves < 1 or args.prompt_kib < 1 or args.output < 1:
        parser.error("concurrency, waves, prompt-kib, and output must be positive")
    all_rows = []
    wave_rows = []
    failed = False

    for wave in range(1, args.waves + 1):
        prompt = make_prompt(args.prompt_kib * 1024)
        prompts = [prompt] * args.concurrency
        started = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [
                pool.submit(
                    request,
                    endpoints[slot % len(endpoints)],
                    args.model,
                    prompt,
                    args.output,
                    args.timeout,
                    wave,
                    slot,
                )
                for slot, prompt in enumerate(prompts)
            ]
            rows = [future.result() for future in futures]
        wall = time.monotonic() - started
        successes = [row for row in rows if row.get("ok", True)]
        failures = [row for row in rows if not row.get("ok", True)]
        if failures:
            failed = True
        total_output = sum(row.get("completion_tokens", 0) for row in successes)
        aggregate = total_output / wall if successes and not failures else None
        all_rows.extend(rows)
        wave_rows.append({
            "wave": wave,
            "wall_seconds": wall,
            "total_output_tokens": total_output,
            "aggregate_output_tok_s": aggregate,
            "successful_requests": len(successes),
            "failed_requests": len(failures),
            "request_wall_seconds": [row["wall_seconds"] for row in rows],
            "errors": failures,
        })
        aggregate_text = (
            f"{aggregate:.2f} tok/s" if aggregate is not None else "INVALID"
        )
        print(
            f"wave={wave} concurrency={args.concurrency} wall={wall:.2f}s "
            f"output={total_output} aggregate_output={aggregate_text} "
            f"request_decode={[row.get('decode_tok_s', 0) for row in rows]} "
            f"errors={[row.get('error_type') for row in failures]}",
            flush=True,
        )

    valid_aggregates = [
        row["aggregate_output_tok_s"]
        for row in wave_rows
        if row["aggregate_output_tok_s"] is not None
    ]
    print(json.dumps({
        "label": args.label,
        "base_urls": base_urls,
        "concurrency": args.concurrency,
        "prompt_kib": args.prompt_kib,
        "output_limit": args.output,
        "waves": wave_rows,
        "median_aggregate_output_tok_s": (
            statistics.median(valid_aggregates) if not failed else None
        ),
        "runs": all_rows,
    }, indent=2, sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
