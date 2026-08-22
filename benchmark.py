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
import difflib
import hashlib
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
    return difflib.SequenceMatcher(None, a, b).ratio()


def _tokens(text: str) -> list[str]:
    return re.findall(r"[\w]+(?:['’\-][\w]+)*", text.casefold())


def _glossary_terms(glossary_text: str) -> set[tuple[str, ...]]:
    """Extract likely canonical/alias terms from the rendered glossary."""
    terms: set[tuple[str, ...]] = set()
    ignored = {"characters", "aliases", "mishearings", "terms", "locations",
               "location", "term", "also", "seen", "as", "none"}
    for line in glossary_text.splitlines():
        line = line.strip().lstrip("- ")
        line = re.sub(r"\(also seen as:", ",", line, flags=re.IGNORECASE)
        if ":" in line:
            line = line.split(":", 1)[1]
        line = re.sub(r"\((?:location|term)\)", "", line, flags=re.IGNORECASE)
        for part in re.split(r"[,=]", line):
            tokens = tuple(t for t in _tokens(part) if t not in ignored)
            if tokens:
                terms.add(tokens)
    return terms


def _contains_term(tokens: list[str], terms: set[tuple[str, ...]]) -> bool:
    for term in terms:
        width = len(term)
        if width == 0:
            continue
        if any(tuple(tokens[i : i + width]) == term for i in range(len(tokens) - width + 1)):
            return True
    return False


def term_fix_recall(raw_text: str, ref_text: str, model_text: str, glossary_text: str) -> float:
    """Recall of glossary-related raw->reference replacements.

    The reference side identifies the canonical term. A model gets credit only
    when that canonical term appears in its output, so a raw misspelling such
    as ``Taika`` -> ``Taiki`` is measured correctly.
    """
    raw_tokens = _tokens(raw_text)
    ref_tokens = _tokens(ref_text)
    model_tokens = _tokens(model_text)
    terms = _glossary_terms(glossary_text)
    expected: set[tuple[str, ...]] = set()
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
        None, raw_tokens, ref_tokens
    ).get_opcodes():
        if tag == "equal":
            continue
        ref_span = ref_tokens[j1:j2]
        if _contains_term(ref_span, terms):
            for term in terms:
                width = len(term)
                if any(tuple(ref_span[i : i + width]) == term
                       for i in range(len(ref_span) - width + 1)):
                    expected.add(term)
    if not expected:
        return 1.0  # no glossary-linked reference change in this cue
    fixed = sum(
        1 for term in expected
        if any(tuple(model_tokens[i : i + len(term)]) == term
               for i in range(len(model_tokens) - len(term) + 1))
    )
    return fixed / len(expected)


def over_change(hyp: str, ref: str, raw: str) -> float:
    """Fraction of words the model changed that the reference did NOT change."""
    raw_words = raw.split()
    if not raw_words and not hyp.split():
        return 0.0
    def changed_positions(target: str) -> set[int]:
        changed = set()
        for tag, i1, i2, _j1, _j2 in difflib.SequenceMatcher(
            None, raw_words, target.split()
        ).get_opcodes():
            if tag != "equal":
                changed.update(range(i1, i2))
                if tag == "insert":
                    changed.add(i1)
        return changed

    ref_changed = changed_positions(ref)
    hyp_changed = changed_positions(hyp)
    if not hyp_changed:
        return 0.0
    return len(hyp_changed - ref_changed) / len(hyp_changed)


def chunk_align(raw_cues, ref_cues, model_cues):
    """Align by cue id and retain missing/extra cues for scoring."""
    def by_id(cues):
        return {cue["id"]: cue for cue in cues}

    raw_by_id, ref_by_id, model_by_id = map(by_id, (raw_cues, ref_cues, model_cues))
    ids = list(dict.fromkeys(
        cue["id"]
        for cues in (raw_cues, ref_cues, model_cues)
        for cue in cues
    ))

    def cue_or_empty(mapping, cue_id):
        return mapping.get(cue_id, {"id": cue_id, "text": ""})

    return (
        [cue_or_empty(raw_by_id, cue_id) for cue_id in ids],
        [cue_or_empty(ref_by_id, cue_id) for cue_id in ids],
        [cue_or_empty(model_by_id, cue_id) for cue_id in ids],
    )


def _normalise_source(value: str) -> str:
    value = value.replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    return value


def _training_sources(path: Path) -> set[str]:
    sources = set()
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"invalid JSON in {path} line {line_no}: {e}") from e
            source = item.get("source")
            if source:
                sources.add(_normalise_source(str(source)))
    if not sources:
        raise ValueError(f"training manifest {path} contains no source fields")
    return sources


def _safe_model_slug(model: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", model).strip("._") or "model"
    digest = hashlib.sha1(model.encode("utf-8")).hexdigest()[:8]
    return f"{slug[:80]}_{digest}"


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
    ap.add_argument("--train-jsonl", default=None,
                    help="training manifest to exclude (default: OUT_DIR/training.jsonl)")
    ap.add_argument("--allow-overlap", action="store_true",
                    help="allow benchmarking examples that are in the training manifest")
    ap.add_argument("--png", default=None, help="output PNG path (default: benchmark.png)")
    args = ap.parse_args()

    if args.holdout <= 0:
        ap.error("--holdout must be positive")
    if args.chunk_size <= 0:
        ap.error("--chunk-size must be positive")
    if args.model_a == args.model_b:
        ap.error("--model-a and --model-b must be different")

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

    train_jsonl = Path(args.train_jsonl) if args.train_jsonl else out_dir / "training.jsonl"
    if not args.allow_overlap:
        if not train_jsonl.is_file():
            print(f"[ERROR] training manifest not found: {train_jsonl}. "
                  "Create a train/holdout split or pass --allow-overlap.",
                  file=sys.stderr)
            return 1
        try:
            train_sources = _training_sources(train_jsonl)
        except ValueError as e:
            print(f"[ERROR] {e}", file=sys.stderr)
            return 1
        paired = [
            (raw, ref) for raw, ref in paired
            if _normalise_source(str(raw.relative_to(raw_dir))) not in train_sources
        ]
        if not paired:
            print("[ERROR] no held-out pairs remain after excluding training sources. "
                  "Create a true holdout split or pass --allow-overlap.",
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
    model_slugs = {model: _safe_model_slug(model) for model in models}
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
            tmp = raw.with_suffix(f".bench_{model_slugs[model]}.srt")
            tmp.parent.mkdir(parents=True, exist_ok=True)
            try:
                db.correct_srt(raw, glossary_text, tmp, args.endpoint, model, api_key,
                               args.chunk_size)
                model_cues = db.parse_srt(tmp)
                if len(model_cues) != len(raw_cues):
                    print(f"  [WARN] {model}: cue count {len(model_cues)} vs "
                          f"raw {len(raw_cues)}; missing/extra cues will be scored")
                raw_c, ref_c, model_c = chunk_align(raw_cues, ref_cues, model_cues)
                for rc, rf, rm in zip(raw_c, ref_c, model_c):
                    results[model]["word_acc"].append(word_acc(rm["text"], rf["text"]))
                    results[model]["similarity"].append(
                        difflib.SequenceMatcher(
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
