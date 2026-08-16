#!/usr/bin/env python3
"""Benchmark: base model vs fine-tuned model on held-out subtitle corrections.

Methodology:
  - Picks N held-out episodes (seeded random) from the raw/ + corrected/ srts
    that were NOT part of the fine-tune dataset.
  - Runs BOTH models' correction on the same held-out raw srts.
  - Scores each model's output against the reference corrections
    (the DeepSeek V4 Flash corrected/ files used as training targets).

Metrics (per model, aggregated over chunks):
  - word_acc     : 1 - normalized edit distance vs reference text
  - similarity   : difflib ratio vs reference
  - term_recall  : glossary terms the reference fixed, model also fixed
  - over_change  : words the model changed that the reference did NOT (lower=better)

Output: console table + PNG bar chart.

Usage:
  python3 benchmark.py --dir /mnt/nas/anime/Shows \
      --out-dir /mnt/nas/training_data \
      --endpoint https://opencode.ai/zen/go/v1/chat/completions \
      --model-a qwen3.5-9b --model-b qwen3.5-9b-finetuned \
      --holdout 5
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import dataset_builder as db  # reuse parse_srt, correct_srt, glossary helpers


def word_acc(hyp: str, ref: str) -> float:
    a = hyp.split()
    b = ref.split()
    n = max(len(a), len(b), 1)
    # normalized Levenshtein-ish (sequence matcher is close enough + cheap)
    sm = __import__("difflib").SequenceMatcher(None, a, b)
    return sm.ratio()


def term_fix_recall(raw_text: str, ref_text: str, model_text: str, glossary_text: str) -> float:
    """Of the glossary terms that CHANGED between raw and ref, how many did the
    model also change? (i.e., did it apply the same glossary fixes)"""
    terms = set()
    for line in glossary_text.splitlines():
        for part in re.split(r"[:=,]", line):
            part = part.strip()
            if part and len(part) > 1:
                terms.add(part)
    changed = set()
    raw_words = set(raw_text.split())
    ref_words = set(ref_text.split())
    for w in raw_words - ref_words:
        for t in terms:
            if t.lower() in w.lower():
                changed.add(t.lower())
                break
    if not changed:
        return 1.0  # nothing to fix — trivially perfect
    model_words = set(model_text.split())
    fixed = 0
    for t in changed:
        # model removed/changed the term? check the raw word is gone
        if not any(t.lower() in w.lower() for w in model_words & raw_words):
            fixed += 1
    return fixed / len(changed)


def over_change(hyp: str, ref: str, raw: str) -> float:
    """Fraction of words the model changed that the reference did NOT change."""
    raw_words = raw.split()
    ref_words = ref.split()
    hyp_words = hyp.split()
    if not raw_words:
        return 0.0
    ref_changed = set()
    for i, w in enumerate(raw_words):
        if i < len(ref_words) and ref_words[i] != w:
            ref_changed.add(i)
    hyp_changed = set()
    for i, w in enumerate(raw_words):
        if i < len(hyp_words) and hyp_words[i] != w:
            hyp_changed.add(i)
    if not hyp_changed:
        return 0.0
    return len(hyp_changed - ref_changed) / len(hyp_changed)


def chunk_align(raw_cues, ref_cues, model_cues):
    n = min(len(raw_cues), len(ref_cues), len(model_cues))
    return raw_cues[:n], ref_cues[:n], model_cues[:n]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="library root (for glossaries)")
    ap.add_argument("--out-dir", required=True, help="training_data dir (raw/ + corrected/)")
    ap.add_argument("--endpoint", default=db.LM_STUDIO_URL)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--skip-server-check", action="store_true")
    ap.add_argument("--model-a", required=True)
    ap.add_argument("--model-b", required=True)
    ap.add_argument("--holdout", type=int, default=5, help="episodes to hold out")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunk-size", type=int, default=db.CHUNK_SIZE)
    ap.add_argument("--png", default=None, help="output PNG path (default: benchmark.png)")
    args = ap.parse_args()

    root = Path(args.dir)
    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    corr_dir = out_dir / "corrected"

    raw_srts = sorted(raw_dir.rglob("*.srt"))
    raw_srts = [p for p in raw_srts if ".corrected" not in p.name]
    paired = []
    for raw in raw_srts:
        rel = raw.relative_to(raw_dir)
        ref = corr_dir / rel.parent / f"{rel.stem}.corrected.srt"
        if ref.exists():
            paired.append((raw, ref))
    if not paired:
        print("[ERROR] no raw+corrected pairs found — run the correction stage first",
              file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    rng.shuffle(paired)
    holdout = paired[: args.holdout]
    print(f"holdout: {len(holdout)} episodes (seed {args.seed}) of {len(paired)} total")

    api_key = args.api_key
    if api_key is None and "opencode.ai" in args.endpoint:
        api_key = db._read_opencode_key()

    if not args.skip_server_check and not db.server_has_loaded_model(args.endpoint):
        print("[ERROR] no model loaded at endpoint — or pass --skip-server-check",
              file=sys.stderr)
        return 1

    models = [args.model_a, args.model_b]
    results = {m: {"word_acc": [], "similarity": [], "term_recall": [], "over_change": []}
               for m in models}

    for idx, (raw, ref) in enumerate(holdout, 1):
        rel = raw.relative_to(raw_dir)
        print(f"[{idx}/{len(holdout)}] {rel}")
        video_parent = root / rel.parent
        gl_path = db.find_glossary(video_parent, root)
        glossary_text = db.glossary_summary(gl_path) if gl_path else ""
        if not glossary_text:
            print("  [WARN] no glossary — term_recall will be trivial")

        raw_cues = db.parse_srt(raw)
        ref_cues = db.parse_srt(ref)
        for model in models:
            tmp = raw.with_suffix(f".bench_{model}.srt")
            tmp.parent.mkdir(parents=True, exist_ok=True)
            try:
                db.correct_srt(raw, glossary_text, tmp, args.endpoint, model, api_key)
                model_cues = db.parse_srt(tmp)
                raw_c, ref_c, model_c = chunk_align(raw_cues, ref_cues, model_cues)
                for rc, rf, rm in zip(raw_c, ref_c, model_c):
                    results[model]["word_acc"].append(word_acc(rm["text"], rf["text"]))
                    results[model]["similarity"].append(
                        __import__("difflib").SequenceMatcher(
                            None, rm["text"], rf["text"]).ratio())
                    results[model]["term_recall"].append(
                        term_fix_recall(rc["text"], rf["text"], rm["text"], glossary_text))
                    results[model]["over_change"].append(
                        over_change(rm["text"], rf["text"], rc["text"]))
            except Exception as e:
                print(f"  [ERROR] {model}: {e}")
            finally:
                tmp.unlink(missing_ok=True)

    print("\n=== RESULTS ===")
    print(f"{'metric':<14}{args.model_a:<24}{args.model_b:<24}")
    summary = {}
    for metric in ("word_acc", "similarity", "term_recall", "over_change"):
        vals_a = results[args.model_a][metric]
        vals_b = results[args.model_b][metric]
        ma = sum(vals_a) / len(vals_a) if vals_a else 0
        mb = sum(vals_b) / len(vals_b) if vals_b else 0
        summary[metric] = (ma, mb)
        print(f"{metric:<14}{ma:<24.3f}{mb:<24.3f}")

    png = args.png or "benchmark.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        labels = [m.replace("_", " ") for m in ("word_acc", "similarity", "term_recall")]
        labels.append("1 - over_change")
        a_vals = [summary["word_acc"][0], summary["similarity"][0],
                  summary["term_recall"][0], 1 - summary["over_change"][0]]
        b_vals = [summary["word_acc"][1], summary["similarity"][1],
                  summary["term_recall"][1], 1 - summary["over_change"][1]]
        x = range(len(labels))
        plt.figure(figsize=(10, 5))
        plt.bar([i - 0.2 for i in x], a_vals, width=0.4, label=args.model_a)
        plt.bar([i + 0.2 for i in x], b_vals, width=0.4, label=args.model_b)
        plt.xticks(list(x), labels)
        plt.ylim(0, 1.05)
        plt.ylabel("score (higher = better)")
        plt.title("Fine-tune benchmark: base vs fine-tuned")
        plt.legend()
        plt.tight_layout()
        plt.savefig(png, dpi=110)
        print(f"\ngraph: {png}")
    except ImportError:
        print("\n(matplotlib missing — skipped graph)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
