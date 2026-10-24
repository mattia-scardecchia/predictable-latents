"""Download and verify the pretrained assets used by the paper recipes."""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import requests

MANIFEST = Path(__file__).parent / "configs" / "assets.json"


def entries(component="all"):
    records = json.loads(MANIFEST.read_text())
    return [row for row in records if component == "all" or row["component"] == component]


def verify(path, record):
    if not path.is_file() or path.stat().st_size != record["bytes"]:
        return False
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return digest == record["sha256"]


def fetch(root, record, check_only=False):
    """Reuse verified files; replace a destination only after full hash verification."""
    destination = Path(root) / record["path"]
    if verify(destination, record):
        return "verified"
    if check_only:
        raise ValueError(f"Missing or invalid asset: {record['path']}")
    headers = {"User-Agent": "predictable-latents/0.1"}
    if record.get("requires_hf_auth"):
        from huggingface_hub import get_token
        token = get_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with requests.get(record["source"], headers=headers, stream=True, timeout=120) as response:
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=destination.name + ".",
                                             suffix=".part", delete=False) as handle:
                temporary = Path(handle.name)
                for block in response.iter_content(chunk_size=4 * 1024 * 1024):
                    handle.write(block)
        if not verify(temporary, record):
            raise ValueError(f"Checksum mismatch for {record['path']}; destination was preserved")
        temporary.replace(destination)
        return "downloaded"
    except requests.HTTPError as exc:
        code = exc.response.status_code
        hint = (" Authenticate with Hugging Face and accept the FLUX model terms."
                if record.get("requires_hf_auth") and code in (401, 403) else "")
        raise RuntimeError(f"Download failed for {record['path']} (HTTP {code}).{hint}") from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("assets"))
    parser.add_argument("--component", choices=["flux", "dino", "metrics", "all"], default="all")
    parser.add_argument("--check-only", action="store_true", help="Verify existing files without network access")
    args = parser.parse_args()
    records = entries(args.component)
    statuses = []
    for row in records:
        status = fetch(args.output, row, args.check_only)
        statuses.append(dict(path=row["path"], status=status, bytes=row["bytes"], sha256=row["sha256"]))
        print(f"{status}: {row['path']}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"verified-{args.component}.json").write_text(json.dumps(statuses, indent=2) + "\n")
    print(f"Verified {len(records)} files, {sum(row['bytes'] for row in records):,} bytes")


if __name__ == "__main__":
    main()
