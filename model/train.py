"""Train MicroGPT on Grafix JSONL (text → triples).

Preserves previous checkpoint by default (timestamped .bak) and can resume
when vocab/config still match.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow running as script: python -m model.train or python model/train.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.microgpt import MicroGPT
from model.tokenizer import CharTokenizer
from model.triples import Triple, encode_example


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def examples_to_docs(rows: list[dict]) -> list[str]:
    docs = []
    for row in rows:
        triples = [
            Triple(t["subject"], t["relation"], t["object"]) for t in row.get("triples", [])
        ]
        docs.append(encode_example(row["text"], triples))
    return docs


def save_checkpoint(path: Path, model: MicroGPT, tokenizer: CharTokenizer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = model.export_weights()
    payload["tokenizer"] = tokenizer.to_dict()
    path.write_text(json.dumps(payload), encoding="utf-8")


def backup_checkpoint(path: Path) -> Path | None:
    if not path.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    bak = path.with_name(f"{path.stem}.bak-{stamp}{path.suffix}")
    shutil.copy2(path, bak)
    print(f"backed up previous checkpoint -> {bak}", flush=True)
    return bak


def try_resume(
    resume_path: Path,
    docs: list[str],
    n_embd: int,
    n_head: int,
    n_layer: int,
    block_size: int,
) -> tuple[MicroGPT, CharTokenizer, bool]:
    """Load prior weights if architecture + charset still compatible."""
    data = json.loads(resume_path.read_text(encoding="utf-8"))
    cfg = data.get("config") or {}
    old_chars = list((data.get("tokenizer") or {}).get("chars") or [])
    new_tokenizer = CharTokenizer.from_texts(docs)
    same_arch = (
        int(cfg.get("n_embd", -1)) == n_embd
        and int(cfg.get("n_head", -1)) == n_head
        and int(cfg.get("n_layer", -1)) == n_layer
        and int(cfg.get("block_size", -1)) == block_size
    )
    # Resume only if old charset covers all chars needed (unknown chars would be dropped)
    if not same_arch:
        print("resume skipped: architecture differs — training fresh", flush=True)
        model = MicroGPT(
            vocab_size=new_tokenizer.vocab_size,
            n_embd=n_embd,
            n_head=n_head,
            n_layer=n_layer,
            block_size=block_size,
        )
        return model, new_tokenizer, False

    needed = set("".join(docs))
    old_set = set(old_chars)
    if not needed.issubset(old_set):
        missing = sorted(needed - old_set)[:20]
        print(
            f"resume skipped: new characters appeared ({missing!r}…) — "
            "training fresh on full dataset (old checkpoint kept as backup)",
            flush=True,
        )
        model = MicroGPT(
            vocab_size=new_tokenizer.vocab_size,
            n_embd=n_embd,
            n_head=n_head,
            n_layer=n_layer,
            block_size=block_size,
        )
        return model, new_tokenizer, False

    tokenizer = CharTokenizer.from_dict({"chars": old_chars})
    model = MicroGPT(
        vocab_size=tokenizer.vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        block_size=block_size,
    )
    model.load_weights(data["weights"])
    print(f"resumed weights from {resume_path}", flush=True)
    return model, tokenizer, True


def train(
    data_path: Path,
    out_path: Path,
    num_steps: int = 300,
    n_embd: int = 32,
    n_head: int = 4,
    n_layer: int = 1,
    block_size: int = 128,
    lr: float = 0.01,
    seed: int = 42,
    resume: Path | None = None,
    do_backup: bool = True,
) -> None:
    random.seed(seed)
    rows = load_jsonl(data_path)
    docs = examples_to_docs(rows)
    if not docs:
        raise SystemExit(f"No examples in {data_path}")

    if do_backup:
        backup_checkpoint(out_path)

    resumed = False
    resume_path = resume if resume is not None else (out_path if out_path.exists() else None)
    # After backup, out_path still exists — for resume use the backup we just made? 
    # Better: resume from explicit path or from pre-backup copy.
    # If we backed up, resume from bak if resume is None and we want continue...
    # Simpler rule: --resume PATH or --resume with default=out_path before overwrite.
    # We already copied to bak; out_path still has old weights until we overwrite at end.
    if resume_path and resume_path.exists():
        model, tokenizer, resumed = try_resume(
            resume_path, docs, n_embd, n_head, n_layer, block_size
        )
    else:
        tokenizer = CharTokenizer.from_texts(docs)
        model = MicroGPT(
            vocab_size=tokenizer.vocab_size,
            n_embd=n_embd,
            n_head=n_head,
            n_layer=n_layer,
            block_size=block_size,
            seed=seed,
        )

    print(
        f"docs={len(docs)} vocab={tokenizer.vocab_size} params={len(model.params)} "
        f"mode={'resume' if resumed else 'fresh'}",
        flush=True,
    )

    m = [0.0] * len(model.params)
    v = [0.0] * len(model.params)
    random.shuffle(docs)

    for step in range(num_steps):
        doc = docs[step % len(docs)]
        tokens = tokenizer.encode(doc, add_bos=True)
        # Keep sequences short — pure-Python MicroGPT is O(n²) per step
        max_len = block_size + 1
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        loss = model.loss_on_tokens(tokens)
        loss.backward()
        model.adam_step(m, v, step, lr=lr, num_steps=num_steps)
        if (step + 1) % 10 == 0 or step == 0:
            print(f"step {step + 1:4d}/{num_steps} | loss {loss.data:.4f}", flush=True)

    save_checkpoint(out_path, model, tokenizer)
    print(f"saved checkpoint -> {out_path}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Train Grafix MicroGPT extractor")
    p.add_argument("--data", default=str(ROOT / "data" / "sample.jsonl"))
    p.add_argument("--out", default=str(ROOT / "model" / "checkpoints" / "extractor.json"))
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument(
        "--resume",
        nargs="?",
        const="__OUT__",
        default=None,
        help="Continue from checkpoint (default: --out). Skips if vocab grew.",
    )
    p.add_argument("--no-backup", action="store_true", help="Do not copy previous --out aside")
    args = p.parse_args()
    out = Path(args.out)
    resume: Path | None
    if args.resume is None:
        resume = None
    elif args.resume == "__OUT__":
        resume = out
    else:
        resume = Path(args.resume)
    train(
        Path(args.data),
        out,
        num_steps=args.steps,
        block_size=args.block_size,
        lr=args.lr,
        resume=resume,
        do_backup=not args.no_backup,
    )


if __name__ == "__main__":
    main()
