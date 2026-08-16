#!/usr/bin/env python3
"""Build fine-tuning data from video files — fully standalone.

Scans a video library, runs ASR + LLM correction for every video, and writes
ALL output (raw srt, corrected srt, extracted embedded subs, training.jsonl)
into a separate output folder. Nothing is written next to the videos.

Usage:
    python3 dataset_builder.py --dir /path/to/anime --out-dir ./training_data

Flow per video:
    1. Probe (ffprobe): English audio? embedded subtitle streams?
    2. Extract audio (ffmpeg) -> 16kHz mono WAV (in out-dir/tmp)
    3. ASR -> raw .srt (macOS: mlx-qwen3-asr; Linux+NVIDIA: CUDA torch)
    4. LLM correction (OpenAI-compatible, whatever LM Studio has loaded)
       -> corrected .srt
    5. Pair raw+corrected into chunks -> training.jsonl
    6. Cleanup temp files

The LLM correction NEVER loads a model — it uses whatever the OpenAI-
compatible server already has loaded, and errors clearly if nothing is loaded.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib import request as urlrequest

IS_MAC = sys.platform == "darwin"

VIDEO_EXTS = {".mkv", ".mp4", ".mov"}
CHUNK_SIZE = 20
MAX_WORKERS = 3  # concurrent LLM correction chunks (gentle on LM Studio)

LM_STUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"
ASR_MODEL_MAC = "mlx-community/Qwen3-ASR-1.7B-8bit"
ASR_MODEL_CUDA = "Qwen/Qwen3-ASR-1.7B-hf"

CORRECTION_PROMPT = """You are a subtitle proofreader. Fix only the following in the subtitle text:
- Character names and glossary terms (spelled correctly per the glossary below)
- Clear ASR mishearings of names/terms
Do NOT rewrite, rephrase, or change anything else. Preserve line count and meaning.
Return ONLY a JSON array of corrections, each {{"cue_id": <number>, "wrong": "...", "correct": "..."}}.
If no corrections are needed, return [].

Glossary:
{glossary}

Subtitle cues:
{cues}
"""

SRT_BLOCK = re.compile(
    r"(?:(\d+)\s*\n)?(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*\n(.*?)(?=\n\s*\n|\Z)",
    re.DOTALL,
)


# --------------------------------------------------------------------------- #
# SRT helpers
# --------------------------------------------------------------------------- #
def parse_srt(path: Path) -> list[dict]:
    """Return cues as [{"id": int, "start": str, "end": str, "text": str}]."""
    text = path.read_text(encoding="utf-8-sig")
    cues = []
    for m in SRT_BLOCK.finditer(text):
        cue_text = " ".join(m.group(4).split()).strip()
        if not cue_text:
            continue
        cues.append({
            "id": int(m.group(1)) if m.group(1) else len(cues) + 1,
            "start": m.group(2),
            "end": m.group(3),
            "text": cue_text,
        })
    return cues


def write_srt(cues: list[dict], path: Path) -> None:
    lines = []
    for i, c in enumerate(cues, 1):
        lines.append(str(i))
        lines.append(f"{c['start']} --> {c['end']}")
        lines.append(c["text"])
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def find_glossary(start: Path, root: Path) -> Path | None:
    for parent in [start] + list(start.parents):
        cand = parent / "glossary.json"
        if cand.is_file():
            return cand
        if parent == root:
            break
    return None


def glossary_summary(path: Path) -> str:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    parts = []
    if isinstance(data, dict):
        for key in ("characters", "aliases", "mishearings", "terms"):
            v = data.get(key)
            if isinstance(v, list):
                parts.append(f"{key}: {', '.join(str(x) for x in v[:100])}")
            elif isinstance(v, dict):
                pairs = [f"{k}={v[k]}" for k in list(v)[:100] if isinstance(v[k], (str, int))]
                parts.append(f"{key}: {', '.join(pairs)}")
    return "\n".join(parts)


def chunked(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


# --------------------------------------------------------------------------- #
# Video probing
# --------------------------------------------------------------------------- #
def probe_video(video: Path) -> dict:
    """ffprobe: audio languages, subtitle streams, duration."""
    info = {"audio": [], "subs": [], "duration": 0.0}
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", "-show_format", str(video)],
            capture_output=True, text=True, timeout=120,
        ).stdout
        data = json.loads(out)
        for st in data.get("streams", []):
            lang = (st.get("tags") or {}).get("language", "")
            if st.get("codec_type") == "audio":
                info["audio"].append(lang)
            elif st.get("codec_type") == "subtitle":
                info["subs"].append({"index": st.get("index"), "lang": lang,
                                     "codec": st.get("codec_name")})
        info["duration"] = float(data.get("format", {}).get("duration", 0) or 0)
    except Exception as e:
        print(f"  [WARN] probe failed for {video.name}: {e}")
    return info


def has_english_audio(info: dict) -> bool:
    if not info["audio"]:
        return False
    return any(lang == "eng" for lang in info["audio"]) or all(lang == "" for lang in info["audio"])


def extract_embedded_subs(video: Path, info: dict, out_dir: Path) -> Path | None:
    """Extract the first embedded subtitle stream to out_dir (best effort)."""
    if not info["subs"]:
        return None
    target = None
    for s in info["subs"]:
        if s.get("lang") == "eng" and s.get("codec") in ("subrip", "srt", "ass", "ssa"):
            target = s
            break
    if target is None:
        target = info["subs"][0]
    dest = out_dir / f"{video.stem}.embedded.srt"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
             "-map", f"0:{target['index']}", str(dest)],
            capture_output=True, timeout=120,
        )
        if dest.exists() and dest.stat().st_size > 0:
            return dest
    except Exception:
        pass
    return None


# --------------------------------------------------------------------------- #
# Audio extraction + ASR
# --------------------------------------------------------------------------- #
def extract_audio(video: Path, wav: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
         "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)],
        check=True, capture_output=True, timeout=600,
    )


def transcribe_mac(wav: Path, json_out: Path, model: str) -> None:
    cmd = ["mlx-qwen3-asr", str(wav), "--output-dir", str(json_out.parent),
           "--model", model]
    subprocess.run(cmd, check=True, capture_output=True, timeout=1800)


_CUDA_ASR = None  # (processor, model, device) — cached across videos


def transcribe_cuda(wav: Path, json_out: Path, model: str) -> None:
    global _CUDA_ASR
    import torch  # lazy — CUDA machines only

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA ASR requires torch with CUDA (see --asr-model); "
                           "on macOS use mlx-qwen3-asr instead")
    if _CUDA_ASR is None:
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        device = "cuda"
        processor = AutoProcessor.from_pretrained(model)
        asr_model = AutoModelForMultimodalLM.from_pretrained(model).to(device)
        _CUDA_ASR = (processor, asr_model, device)
        print(f"  [INFO] CUDA ASR model loaded (cached for the whole run)")
    else:
        processor, asr_model, device = _CUDA_ASR
    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(str(wav))
    if sr != 16000:
        raise RuntimeError(f"expected 16kHz WAV, got {sr}")
    # Chunk the audio so the mel feature tensor stays small on 8GB GPUs
    # (a full episode's features alone can exceed VRAM). 30s segments keep
    # input_features ~200MB; ASR_BATCH of them per generate() call keeps the
    # GPU busy (batch=1 on a 1.7B model underutilizes the card).
    seg_samples = 16000 * 30
    ASR_BATCH = 4
    starts = list(range(0, len(audio), seg_samples))
    n_batches = (len(starts) + ASR_BATCH - 1) // ASR_BATCH
    chunks = []

    # Double-buffer: a prep thread builds the NEXT batch's input features on
    # the CPU while the GPU generates the current one — the GPU never idles
    # waiting for feature extraction.
    import queue as _queue

    batch_q: "_queue.Queue" = _queue.Queue(maxsize=2)

    def _prep():
        try:
            for b in range(0, len(starts), ASR_BATCH):
                batch_starts = starts[b : b + ASR_BATCH]
                segs = [audio[s : s + seg_samples] for s in batch_starts]
                # Official Qwen3-ASR entry point (handles text/audio token
                # alignment); accepts a list of audios -> batched inference.
                inputs = processor.apply_transcription_request(audio=segs).to(device, asr_model.dtype)
                batch_q.put((batch_starts, segs, inputs))
        except Exception as e:  # noqa: BLE001
            batch_q.put(e)

    prep_thread = threading.Thread(target=_prep, daemon=True)
    prep_thread.start()

    with torch.no_grad():
        for _ in range(n_batches):
            item = batch_q.get()
            if isinstance(item, Exception):
                raise item
            batch_starts, segs, inputs = item
            outputs = asr_model.generate(**inputs, max_new_tokens=512)
            seq_len = inputs["input_ids"].shape[1]
            for j, out in enumerate(outputs):
                gen_ids = out[None, seq_len:]
                text = processor.decode(gen_ids, return_format="transcription_only")[0].strip()
                if text:
                    s = batch_starts[j]
                    chunks.append({"text": text,
                                   "start": s / sr,
                                   "end": min((s + len(segs[j])) / sr, len(audio) / sr)})
    prep_thread.join()
    # NOTE: the model is intentionally NOT freed here — it stays cached in
    # _CUDA_ASR for the next video (the big per-video reload win).
    json_out.write_text(json.dumps({"segments": [], "chunks": chunks}), encoding="utf-8")


def asr_to_srt(json_path: Path, srt_path: Path) -> None:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    chunks = data.get("chunks") or []
    cues = []
    for ch in chunks:
        if not isinstance(ch, dict) or not ch.get("text", "").strip():
            continue
        start = float(ch.get("start", 0))
        end = float(ch.get("end", start + 1))
        cues.append({
            "start": fmt_ts(start),
            "end": fmt_ts(end),
            "text": " ".join(str(ch.get("text", "")).split()),
        })
    write_srt(cues, srt_path)


def fmt_ts(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    h = total_ms // 3_600_000
    m = (total_ms % 3_600_000) // 60_000
    s = (total_ms % 60_000) // 1000
    ms = total_ms % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def transcribe(wav: Path, json_out: Path, model: str) -> None:
    if IS_MAC:
        transcribe_mac(wav, json_out, model)
    else:
        transcribe_cuda(wav, json_out, model)


# --------------------------------------------------------------------------- #
# LLM correction (OpenAI-compatible)
# --------------------------------------------------------------------------- #
def http_json(method: str, url: str, payload=None, timeout: int = 600) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urlrequest.Request(url, data=data, method=method,
                             headers={"Content-Type": "application/json"})
    with urlrequest.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def server_has_loaded_model(chat_url: str) -> bool:
    base = chat_url.rsplit("/chat/completions", 1)[0]
    try:
        data = http_json("GET", f"{base}/models", timeout=8)
    except Exception:
        return False
    items = data.get("data") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return False
    for item in items:
        if not isinstance(item, dict):
            continue
        if isinstance(item.get("loaded_instances"), list) and item["loaded_instances"]:
            return True
        if item.get("state") == "loaded" or item.get("status") == "loaded":
            return True
    return False


def call_llm(prompt: str, url: str) -> str:
    payload = {"model": "current", "messages": [{"role": "user", "content": prompt}],
               "temperature": 0.0, "max_tokens": 4000}
    data = http_json("POST", url, payload)
    choice = data["choices"][0]
    if choice.get("finish_reason") == "length":
        print("  [WARN] response truncated (hit max_tokens) — chunk may be incomplete")
    return choice["message"]["content"]


def extract_json_array(text: str) -> list[dict]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE)
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []


def apply_corrections(cues: list[dict], corrections: list[dict]) -> int:
    cue_map = {c["id"]: c for c in cues}
    applied = 0
    for corr in corrections:
        cue = cue_map.get(corr.get("cue_id"))
        wrong, correct = corr.get("wrong"), corr.get("correct")
        if not cue or not wrong or not correct:
            continue
        pattern = re.compile(rf"\b{re.escape(wrong)}\b", re.IGNORECASE)
        if pattern.search(cue["text"]):
            cue["text"] = pattern.sub(correct, cue["text"])
            applied += 1
    return applied


def correct_srt(srt_path: Path, glossary_text: str, out_path: Path, url: str) -> int:
    cues = parse_srt(srt_path)
    if not cues:
        raise ValueError("no cues parsed")
    total = 0
    chunks = list(chunked(cues, CHUNK_SIZE))
    workers = min(len(chunks), MAX_WORKERS)
    prompts = []
    for idx, chunk in enumerate(chunks, 1):
        cue_lines = "\n".join(f"[{c['id']}] {c['text']}" for c in chunk)
        prompts.append(CORRECTION_PROMPT.format(glossary=glossary_text, cues=cue_lines))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(call_llm, p, url) for p in prompts]
        for fut in as_completed(futures):
            raw = fut.result()
            corrections = extract_json_array(raw)
            total += apply_corrections(cues, corrections)
    write_srt(cues, out_path)
    return total


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="video library root (recursive scan)")
    ap.add_argument("--out-dir", default="./training_data", help="ALL output goes here (use the NAS mount so both machines see it)")
    ap.add_argument("--tmp-dir", default="/tmp/dataset_asr",
                    help="LOCAL temp dir for WAV/ASR intermediates (must be local disk — "
                         "SMB/FUSE mounts can't seek, which breaks ffmpeg WAV writes)")
    ap.add_argument("--stage", choices=["all", "asr", "correct"], default="all",
                    help="asr: run ASR only (2070S, writes raw/). correct: run correction "
                         "only (Mac, reads raw/ -> corrected/ + training.jsonl). all: both.")
    ap.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    ap.add_argument("--lm-studio-url", default=LM_STUDIO_URL)
    ap.add_argument("--asr-model", default=None,
                    help=f"ASR model (default: {ASR_MODEL_MAC if IS_MAC else ASR_MODEL_CUDA})")
    ap.add_argument("--min-diff", action="store_true",
                    help="skip no-op chunks (raw == corrected) in training.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="process only first N videos (test)")
    ap.add_argument("--depth", type=int, default=6,
                    help="how many videos ahead to extract audio (pipeline overlap)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip videos that already have a raw srt in out-dir (resumable runs)")
    ap.add_argument("--dry-run", action="store_true", help="scan + plan only, no processing")
    args = ap.parse_args()

    root = Path(args.dir)
    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    corr_dir = out_dir / "corrected"
    ref_dir = out_dir / "reference"
    tmp_dir = out_dir / "tmp"
    asr_model = args.asr_model or (ASR_MODEL_MAC if IS_MAC else ASR_MODEL_CUDA)

    if args.stage in ("all", "asr"):
        return run_asr_stage(root, raw_dir, ref_dir, Path(args.tmp_dir), asr_model,
                             args.limit, args.dry_run, args.skip_existing, args.depth)
    return run_correct_stage(root, raw_dir, corr_dir, out_dir, args.chunk_size,
                             args.lm_studio_url, args.min_diff, args.limit, args.dry_run)


def run_asr_stage(root: Path, raw_dir: Path, ref_dir: Path, tmp_dir: Path,
                  asr_model: str, limit: int, dry_run: bool,
                  skip_existing: bool = False, depth: int = 3) -> int:
    videos = sorted(p for p in root.rglob("*")
                    if p.suffix.lower() in VIDEO_EXTS and not p.name.startswith("._"))
    if not videos:
        print(f"[ERROR] no video files found under {root}", file=sys.stderr)
        return 1
    if limit:
        videos = videos[:limit]
    for d in (raw_dir, ref_dir, tmp_dir):
        d.mkdir(parents=True, exist_ok=True)
    print(f"ASR stage — videos: {len(videos)}  asr: {asr_model}")

    if dry_run:
        for v in videos:
            info = probe_video(v)
            eng_audio = has_english_audio(info)
            subs = ", ".join(f"#{s['index']}:{s['lang']}" for s in info["subs"]) or "none"
            print(f"  {v.relative_to(root)}  [audio: {eng_audio}] [subs: {subs}]")
        print("dry-run complete — nothing processed")
        return 0

    done = skipped = failed = 0
    extract_pool = ThreadPoolExecutor(max_workers=depth,
                                      thread_name_prefix="extract")
    futures: dict[int, tuple[Path, "concurrent.futures.Future"]] = {}
    for idx, video in enumerate(videos, 1):
        rel = video.relative_to(root)
        rel_out = rel.parent / f"{rel.stem}.srt"
        raw_srt = raw_dir / rel_out
        if skip_existing and raw_srt.exists():
            print(f"[{idx}/{len(videos)}] {rel} — already done, skipping")
            skipped += 1
            continue
        print(f"[{idx}/{len(videos)}] {rel}")
        info = probe_video(video)
        if not has_english_audio(info):
            print("  [SKIP] no English audio track")
            skipped += 1
            continue

        # Fire-and-forget: embedded-subs extraction is informational (reference
        # copy only) — don't let it block the GPU between videos.
        if info["subs"]:
            threading.Thread(
                target=extract_embedded_subs, args=(video, info, ref_dir), daemon=True
            ).start()

        # Mirror the library tree under raw_dir so the correction stage can
        # find glossaries via the original video's parent folder.
        # idx-prefixed temp names avoid collisions when two videos share a stem.
        wav = tmp_dir / f"{idx:04d}_{video.stem}.wav"
        asr_json = tmp_dir / f"{idx:04d}_{video.stem}.json"
        raw_srt.parent.mkdir(parents=True, exist_ok=True)

        # Use the prefetched wav if a worker extracted it, else extract now.
        entry = futures.pop(idx, None)
        if entry is not None:
            _, fut = entry
            try:
                fut.result()  # raises if extraction failed
            except Exception as e:
                print(f"  [ERROR] prefetched audio extract failed: {e} — retrying sync")
                try:
                    extract_audio(video, wav)
                except Exception as e2:
                    print(f"  [ERROR] audio extract failed: {e2}")
                    failed += 1
                    continue
        else:
            try:
                extract_audio(video, wav)
            except subprocess.CalledProcessError as e:
                print(f"  [ERROR] audio extract failed: {str(e)[:160]}")
                failed += 1
                continue
            except Exception as e:
                print(f"  [ERROR] audio extract failed: {e}")
                failed += 1
                continue

        try:
            transcribe(wav, asr_json, asr_model)
            asr_to_srt(asr_json, raw_srt)
            print(f"  [OK] raw srt -> {raw_srt.relative_to(raw_dir.parent)}")
            done += 1
        except subprocess.CalledProcessError as e:
            print(f"  [ERROR] ASR failed: {str(e)[:160]}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] ASR failed: {e}")
            failed += 1
        finally:
            for p in (wav, asr_json):
                p.unlink(missing_ok=True)

        # Pipeline overlap: keep up to `depth` extractions running ahead so the
        # GPU never waits on the SMB read + ffmpeg extract.
        for ahead in range(idx + 1, min(idx + depth, len(videos)) + 1):
            if ahead in futures:
                continue
            nv = videos[ahead - 1]
            nwav = tmp_dir / f"{ahead:04d}_{nv.stem}.wav"
            futures[ahead] = (nwav, extract_pool.submit(extract_audio, nv, nwav))

    extract_pool.shutdown(wait=False)
    print(f"\nASR done: {done} ok, {skipped} skipped, {failed} failed")
    return 0


def run_correct_stage(root: Path, raw_dir: Path, corr_dir: Path, out_dir: Path,
                      chunk_size: int, lm_url: str, min_diff: bool,
                      limit: int, dry_run: bool) -> int:
    if not raw_dir.is_dir():
        print(f"[ERROR] no raw/ dir at {raw_dir} — run the ASR stage first", file=sys.stderr)
        return 1
    raw_srts = sorted(raw_dir.rglob("*.srt"))
    raw_srts = [p for p in raw_srts if ".corrected" not in p.name]
    if not raw_srts:
        print(f"[ERROR] no *.srt files under {raw_dir}", file=sys.stderr)
        return 1
    if limit:
        raw_srts = raw_srts[:limit]
    corr_dir.mkdir(parents=True, exist_ok=True)
    print(f"Correction stage — raw srts: {len(raw_srts)}  llm: {lm_url}")

    if dry_run:
        for raw in raw_srts:
            rel = raw.relative_to(raw_dir)
            video_parent = root / rel.parent
            gl = find_glossary(video_parent, root)
            print(f"  {rel}  [glossary: {gl.relative_to(root) if gl else 'none'}]")
        print("dry-run complete — nothing processed")
        return 0

    if not server_has_loaded_model(lm_url):
        print("[ERROR] No model loaded at the OpenAI-compatible endpoint "
              f"({lm_url}). Load one in LM Studio first — this script will NOT "
              "load one for you.")
        return 1

    n_total = n_changed = n_noop = 0
    out_jsonl = open(out_dir / "training.jsonl", "w", encoding="utf-8")
    try:
        for idx, raw in enumerate(raw_srts, 1):
            rel = raw.relative_to(raw_dir)
            print(f"[{idx}/{len(raw_srts)}] {rel}")
            video_parent = root / rel.parent
            glossary_path = find_glossary(video_parent, root)
            glossary_text = glossary_summary(glossary_path) if glossary_path else ""
            if not glossary_text:
                print("  [WARN] no glossary found — correcting without one")

            corr = corr_dir / rel.parent / f"{rel.stem}.corrected.srt"
            corr.parent.mkdir(parents=True, exist_ok=True)
            try:
                applied = correct_srt(raw, glossary_text, corr, lm_url)
                print(f"  [OK] corrected ({applied} fixes)")
            except Exception as e:
                print(f"  [ERROR] correction failed: {e}")
                continue

            raw_cues = parse_srt(raw)
            corr_cues = parse_srt(corr)
            n = min(len(raw_cues), len(corr_cues))
            for r_chunk, c_chunk in zip(chunked(raw_cues[:n], chunk_size),
                                        chunked(corr_cues[:n], chunk_size)):
                raw_text = "\n".join(c["text"] for c in r_chunk)
                corr_text = "\n".join(c["text"] for c in c_chunk)
                changed = raw_text != corr_text
                if min_diff and not changed:
                    continue
                n_total += 1
                n_changed += 1 if changed else 0
                n_noop += 0 if changed else 1
                out_jsonl.write(json.dumps({
                    "raw": raw_text,
                    "corrected": corr_text,
                    "changed": changed,
                    "source": str(rel),
                }, ensure_ascii=False) + "\n")
    finally:
        out_jsonl.close()

    print(f"\nchunks: {n_total}  (changed: {n_changed}, no-op: {n_noop})")
    print(f"training data: {out_dir / 'training.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
