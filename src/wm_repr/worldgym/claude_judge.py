"""Stateless Claude CLI transport for the unchanged WorldGym scorer."""

import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

CLAUDE = os.environ.get("CLAUDE_BIN", "claude")
ATTEMPTS = 3  # per sample; a failed attempt is retried after 30 s


class ClaudeCLI:
    """One scorer request: `create` runs n `claude -p` processes in parallel."""

    def __init__(self, model, effort, workdir, raw_dir, tag):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        self.model, self.effort, self.tag = model, effort, tag
        self.workdir, self.raw_dir = workdir, raw_dir
        self.request, self.calls = None, []

    def cmd(self):
        return [CLAUDE, "-p",
                "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
                "--safe-mode", "--tools", "", "--no-session-persistence",
                "--system-prompt", "",  # the OpenAI request had no system message
                "--model", self.model, "--effort", self.effort]

    def create(self, model, messages, n):
        prompt, *images = messages[0]["content"]
        content = [{"type": "text", "text": prompt}] + [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                          "data": im["image"]}} for im in images]
        self.request = {"type": "user", "message": {"role": "user", "content": content}}
        with ThreadPoolExecutor(n) as pool:
            self.calls = list(pool.map(self._call, range(n)))
        choices = [SimpleNamespace(message=SimpleNamespace(content=c["text"]),
                                   finish_reason=c["stop_reason"]) for c in self.calls]
        return SimpleNamespace(choices=choices, model=self.calls[0]["model"], usage=None, id=None)

    def _call(self, sample):
        stdin = json.dumps(self.request) + "\n"
        for attempt in range(1, ATTEMPTS + 1):
            t0 = time.time()
            proc = subprocess.run(self.cmd(), input=stdin, capture_output=True, text=True,
                                  cwd=self.workdir, timeout=900)
            raw_path = self.raw_dir / f"{self.tag}_sample{sample}_attempt{attempt}.jsonl"
            raw_path.write_text(proc.stdout)
            out = parse_stream(proc.stdout)
            out.update(wall_s=round(time.time() - t0, 2), attempt=attempt, raw=str(raw_path),
                       returncode=proc.returncode, stderr_tail=proc.stderr[-500:])
            if proc.returncode == 0 and not out["is_error"] and out["text"]:
                return out
            if attempt < ATTEMPTS:
                time.sleep(30)
        raise RuntimeError(f"{self.tag} sample {sample}: no reply after {ATTEMPTS} attempts: "
                           f"{out['error_text'] or out['stderr_tail']}")


def parse_stream(stdout):
    out = {"text": "", "thinking": [], "model": None, "stop_reason": None, "usage": None,
           "duration_ms": None, "is_error": True, "error_text": ""}
    for line in stdout.splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = d.get("type")
        if t == "system" and d.get("subtype") == "init":
            out["model"] = d.get("model")
        elif t == "assistant":
            msg = d["message"]
            out["stop_reason"] = msg.get("stop_reason") or out["stop_reason"]
            for b in msg.get("content", []):
                if b.get("type") == "text":
                    out["text"] += b.get("text", "")
                elif b.get("type") in ("thinking", "redacted_thinking"):
                    out["thinking"].append(b.get("thinking", "<redacted>"))
        elif t == "result":
            out.update(usage=d.get("usage"), duration_ms=d.get("duration_ms"),
                       is_error=bool(d.get("is_error")))
            if d.get("is_error"):
                out["error_text"] = str(d.get("result", ""))[:500]
    return out


def cli_version():
    return subprocess.run([CLAUDE, "--version"], capture_output=True, text=True).stdout.strip()

