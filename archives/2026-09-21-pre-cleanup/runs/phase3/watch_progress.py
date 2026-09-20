#!/usr/bin/env python3
"""Read existing Phase 3 progress; never launch or modify an experiment."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
SESSION = ROOT / "handoff-20260914.NByJvb"


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def snapshot():
    now = time.time()
    lines = ["Phase 3 实时进度（只读查看，不启动实验）",
             time.strftime("%Y-%m-%d %H:%M:%S %Z"),
             "执行 thread: 01a09bd8-a78a-7702-9e4b-57442618928c · Astra / xhigh", ""]
    logs = list(SESSION.glob("events*.jsonl"))
    log = max(logs, key=lambda p: p.stat().st_mtime) if logs else None
    messages, pending = [], {}
    if log:
        with log.open() as stream:
            for raw in stream:
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue  # A concurrently appended final line may be incomplete.
                if event.get("type") == "turn.started":
                    pending.clear()
                item = event.get("item", {})
                if item.get("type") == "agent_message":
                    messages.append(item.get("text", "").strip())
                if item.get("type") == "command_execution":
                    if event.get("type") == "item.started":
                        pending[item.get("id")] = item.get("command", "")
                    elif event.get("type") == "item.completed":
                        pending.pop(item.get("id"), None)
                if event.get("type") in ("turn.completed", "turn.failed"):
                    pending.clear()
        lines += [f"执行事件最近更新：{now - log.stat().st_mtime:.0f} 秒前",
                  "最近公开进度说明（保留原文）："]
        lines += [f"  • {message}" for message in messages[-3:]]
        if pending:
            lines += ["已启动、尚未收到完成事件的命令："]
            lines += ["  " + " ".join(command.split())[:220] for command in list(pending.values())[-3:]]
    lines += ["", "各实验最近落盘状态（不将旧状态当成当前进程存活证据）："]
    paths = sorted(ROOT.rglob("progress.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in paths[:8]:
        record = read_json(path)
        if record is None:
            continue
        pid = record.get("pid")
        visible = Path(f"/proc/{pid}").exists() if pid else False
        lines.append(f"  {path.parent.relative_to(ROOT)} · {now - path.stat().st_mtime:.0f} 秒前 · PID {pid} "
                     + ("在当前进程视图可见" if visible else "当前进程视图不可见"))
        fields = ["stage", "step", "total", "loss", "nll", "ppl", "cuda_visible_devices",
                  "allocated_gib", "peak_allocated_gib"]
        lines.append("    " + " | ".join(f"{key}={record[key]}" for key in fields if key in record))
    if not paths:
        lines.append("  尚无实验 progress.json。")
    lines += ["", "每 3 秒刷新；Ctrl+C 仅退出查看，不停止后台执行。",
              "完整记录：" + str(SESSION / "events_resource_update.jsonl")]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="Print a single read-only snapshot.")
    args = parser.parse_args()
    try:
        while True:
            if sys.stdout.isatty() and not args.once:
                print("\033[2J\033[H", end="")
            print(snapshot(), flush=True)
            if args.once:
                break
            time.sleep(3)
    except KeyboardInterrupt:
        pass
