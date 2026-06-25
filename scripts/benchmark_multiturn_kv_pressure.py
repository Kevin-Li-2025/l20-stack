#!/usr/bin/env python3
"""Benchmark OpenAI-compatible multi-turn KV-cache pressure with streaming."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def post_stream(url: str, payload: dict, timeout: float):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    first_token_time = None
    token_count = 0
    chunks = []
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue
            choice = event.get("choices", [{}])[0]
            text = choice.get("text")
            if text is None:
                delta = choice.get("delta", {})
                text = delta.get("content", "")
            if text:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                token_count += 1
                chunks.append(text)
    end = time.perf_counter()
    return {
        "ttft_ms": None if first_token_time is None else (first_token_time - start) * 1000,
        "e2e_ms": (end - start) * 1000,
        "stream_chunks": token_count,
        "text": "".join(chunks),
    }


def make_prefix(target_chars: int) -> str:
    unit = (
        "L20 KV pressure fixture. Keep this prefix resident across turns; "
        "it represents a long system prompt with repeated policy, code, and notes.\n"
    )
    return (unit * ((target_chars // len(unit)) + 1))[:target_chars]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--turns", type=int, default=8)
    parser.add_argument("--prefix-chars", type=int, default=24000)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prompt = make_prefix(args.prefix_chars)
    reports = []
    endpoint = args.base_url.rstrip("/") + "/v1/completions"
    for turn in range(args.turns):
        prompt += f"\nUser turn {turn}: summarize one L20 optimization boundary in one sentence.\nAssistant:"
        payload = {
            "model": args.model,
            "prompt": prompt,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "stream": True,
            "ignore_eos": True,
        }
        try:
            result = post_stream(endpoint, payload, args.timeout)
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise SystemExit(f"HTTP {error.code}: {body}") from error
        generated = result.pop("text")
        prompt += generated
        reports.append(
            {
                "turn": turn,
                "prompt_chars": len(prompt),
                "ttft_ms": result["ttft_ms"],
                "e2e_ms": result["e2e_ms"],
                "stream_chunks": result["stream_chunks"],
                "generated_chars": len(generated),
            }
        )

    valid_ttft = [row["ttft_ms"] for row in reports if row["ttft_ms"] is not None]
    summary = {
        "schema_version": 1,
        "model": args.model,
        "base_url": args.base_url,
        "turns": args.turns,
        "prefix_chars": args.prefix_chars,
        "max_tokens": args.max_tokens,
        "reports": reports,
        "summary": {
            "first_turn_ttft_ms": valid_ttft[0] if valid_ttft else None,
            "last_turn_ttft_ms": valid_ttft[-1] if valid_ttft else None,
            "max_ttft_ms": max(valid_ttft) if valid_ttft else None,
            "total_e2e_ms": sum(row["e2e_ms"] for row in reports),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
