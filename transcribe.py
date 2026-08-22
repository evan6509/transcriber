"""Transcribe anime videos into .srt subtitles, correct them against a glossary via
LM Studio, and embed them back into the video containers.

Cross-platform ASR backend (single script):
- macOS (Apple Silicon): runs the `mlx-qwen3-asr` CLI (MLX). Torch/transformers are
  never imported on macOS, so they are not required to be installed there.
- Linux + NVIDIA (CUDA): runs Qwen3-ASR directly with `torch` + `transformers` on a
  CUDA GPU (lazy-imported inside the CUDA branch). Uses a forced aligner for
  word-level timestamps, chunking long audio at low-energy boundaries.
Both backends emit the same JSON shape ({segments, chunks}) consumed by _json_to_srt.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

IS_MAC = sys.platform == "darwin"

VIDEO_EXTS = {".mkv", ".mp4", ".mov"}
MODEL = "mlx-community/Qwen3-ASR-1.7B-8bit"
# transformers-native checkpoints for the Linux/CUDA path (verified on the HF hub):
CUDA_MODEL = "Qwen/Qwen3-ASR-1.7B-hf"
CUDA_ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B-hf"

LM_STUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"
LM_STUDIO_MODEL = "current"
LLM_API_KEY: str | None = None
CHUNK_SIZE = 20
TIMEOUT = 600
CORRECTED_HASHES_FILE = "corrected_hashes.json"
EXCLUDE_DIRS = {"Wishlist", "dump"}
DRY_RUN = False
YES = False

# CUDA ASR tuning (mirrors the qwen-asr package's reference limits).
SAMPLE_RATE = 16000
CUDA_CHUNK_SEC = 180.0
CUDA_MAX_NEW_TOKENS = 2048


def _in_excluded(path: Path, root: Path) -> bool:
    try:
        rel = path.relative_to(root)
        for part in rel.parts:
            if part in EXCLUDE_DIRS:
                return True
    except ValueError:
        return False
    return False

SRT_BLOCK_RE = re.compile(
    r"(\d+)\s*\n"
    r"(\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3})\s*\n"
    r"(.*?)(?=\n\d+\s*\n\d{2}:\d{2}:\d{2},\d{3}|\Z)",
    re.DOTALL,
)

def _check_tool(name: str) -> None:
    if subprocess.run(["which", name], capture_output=True).returncode != 0:
        print(f"[ERROR] '{name}' not found on PATH. Install it and try again.")
        sys.exit(1)


def _check_tools() -> None:
    _check_tool("ffmpeg")
    # ffprobe is used for audio/subtitle detection on every platform and for
    # validating embedded output.
    _check_tool("ffprobe")
    if IS_MAC:
        _check_tool("mlx-qwen3-asr")
        return
    try:
        import torch
    except ImportError:
        torch = None
    if torch is None or not torch.cuda.is_available():
        print("[ERROR] Qwen3-ASR CUDA path requires torch with CUDA; on macOS use mlx-qwen3-asr instead")
        sys.exit(1)


def _is_appledouble(path: Path) -> bool:
    return path.name.startswith("._")


def _is_english_audio(video: Path) -> bool:
    """Check if the video has an English audio track. Returns True for eng-tagged tracks.
    Untagged 'und' audio is treated as unknown — returns False (skip, since we can't
    confirm it's English and ASR with --language english would produce garbage on JP audio)."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-select_streams", "a",
        "-show_entries", "stream=codec_type:stream_tags=language",
        "-of", "json",
        str(video),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        streams = json.loads(result.stdout).get("streams", [])
        for s in streams:
            lang = s.get("tags", {}).get("language", "")
            if lang in ("eng", "en", "English"):
                return True
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        pass
    return False


def _scan_videos(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTS and not _in_excluded(p, root) and not _is_appledouble(p)]


_TEXT_SUB_CODECS = {"subrip", "srt", "ass", "ssa", "webvtt", "mov_text"}


def _has_embedded_english_subs(video: Path) -> bool:
    cmd = [
        "ffprobe", "-v", "quiet",
        "-select_streams", "s",
        "-show_entries", "stream=codec_name:stream_tags=language",
        "-of", "json",
        str(video),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        streams = json.loads(result.stdout).get("streams", [])
        for s in streams:
            if s.get("codec_name") in _TEXT_SUB_CODECS:
                lang = s.get("tags", {}).get("language", "")
                if lang in ("eng", "en", "English"):
                    return True
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        pass
    return False


def _has_existing_srt(video: Path) -> bool:
    if video.with_suffix(".srt").exists():
        return True
    if list(video.parent.glob(f"{video.stem}.*.srt")):
        return True
    return _has_embedded_english_subs(video)


def _classify(files: list[Path], force: bool = False) -> tuple[list[Path], list[Path]]:
    pending: list[Path] = []
    skipped: list[Path] = []
    for f in sorted(files):
        if not _is_english_audio(f):
            skipped.append(f)
            continue
        (skipped if not force and _has_existing_srt(f) else pending).append(f)
    return pending, skipped


def _extract_audio(video: Path, dst: Path) -> None:
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video),
        "-ar", "16000", "-ac", "1",
        str(dst),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def _transcribe(audio_path: Path, model_id: str = MODEL) -> Path:
    if IS_MAC:
        return _transcribe_mlx(audio_path, model_id)
    return _transcribe_cuda(audio_path, model_id)


def _transcribe_mlx(audio_path: Path, model_id: str) -> Path:
    output_path = audio_path.with_suffix(".json")
    cmd = [
        "mlx-qwen3-asr",
        "--model", model_id,
        "--language", "english",
        "--output-format", "json",
        "--output-dir", str(audio_path.parent),
        "--quiet",
        "--timestamps",
        str(audio_path),
    ]
    subprocess.run(cmd, check=True)
    return output_path


def _read_wav_float32(audio_path: Path) -> "np.ndarray":
    """Read a 16 kHz mono WAV (as written by _extract_audio) into a float32 array."""
    import wave
    import numpy as np

    with wave.open(str(audio_path), "rb") as wav:
        n_channels = wav.getnchannels()
        sampwidth = wav.getsampwidth()
        raw = wav.readframes(wav.getnframes())
    if sampwidth == 2:
        pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sampwidth == 4:
        pcm = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif sampwidth == 1:
        pcm = (np.frombuffer(raw, dtype="<u1").astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth} byte(s)")
    if n_channels > 1:
        pcm = pcm.reshape(-1, n_channels)[:, 0]
    return np.asarray(pcm, dtype=np.float32)


def _split_long_audio(wav, sr: int, max_chunk_sec: float):
    """Split long audio into <=max_chunk_sec chunks at low-energy boundaries.

    Mirrors qwen-asr's split_audio_into_chunks: concatenating the returned
    chunks reproduces the input exactly (no overlaps, no gaps). Returns a list
    of (chunk_wav, offset_sec) pairs.
    """
    import numpy as np

    wav = np.asarray(wav, dtype=np.float32)
    total_len = int(wav.shape[0])
    total_sec = total_len / float(sr)
    if total_sec <= max_chunk_sec:
        return [(wav, 0.0)]

    max_len = int(max_chunk_sec * sr)
    expand = int(5.0 * sr)
    win = max(4, int(0.1 * sr))

    chunks = []
    start = 0
    offset = 0.0
    while (total_len - start) > max_len:
        cut = start + max_len
        left = max(start, cut - expand)
        right = min(total_len, cut + expand)
        if right - left <= win:
            boundary = cut
        else:
            seg = wav[left:right]
            window_sums = np.convolve(np.abs(seg), np.ones(win, dtype=np.float32), mode="valid")
            min_pos = int(np.argmin(window_sums))
            boundary = left + min_pos + int(np.argmin(np.abs(seg[min_pos : min_pos + win])))
        boundary = int(max(boundary, start + 1))
        boundary = int(min(boundary, total_len))
        chunks.append((wav[start:boundary], offset))
        offset += (boundary - start) / float(sr)
        start = boundary
    chunks.append((wav[start:total_len], offset))
    return chunks


def _join_words(texts: list[str]) -> str:
    text = " ".join(t for t in texts if t)
    return re.sub(r"\s+([,.;:!?，。！？；：])", r"\1", text).strip()


def _words_to_chunks(word_items: list[dict]) -> list[dict]:
    """Group word-level timestamps into sentence-sized chunks for _json_to_srt.

    Emits {"text", "start", "end"} dicts; start/end are floats in seconds. A
    chunk is flushed at sentence-ending punctuation (or every 80 words as a
    safety cap), skipping single-word abbreviations like "Mr."/"Dr.".
    """
    chunks = []
    cur = []
    for item in word_items:
        cur.append(item)
        text = item["text"]
        is_end = text.endswith(("。", "！", "？", ".", "!", "?"))
        if text.endswith(".") and len(cur) == 1 and re.match(r"^[A-Za-z]{1,4}\.$", text):
            is_end = False  # abbreviation, not a sentence end
        if (is_end and len(cur) >= 2) or len(cur) >= 80:
            joined = _join_words([w["text"] for w in cur])
            if joined:
                chunks.append({"text": joined, "start": float(cur[0]["start"]), "end": float(cur[-1]["end"])})
            cur = []
    if cur:
        joined = _join_words([w["text"] for w in cur])
        if joined:
            chunks.append({"text": joined, "start": float(cur[0]["start"]), "end": float(cur[-1]["end"])})
    return chunks


def _transcribe_cuda(audio_path: Path, model_id: str) -> Path:
    output_path = audio_path.with_suffix(".json")
    import torch
    from transformers import (
        AutoModelForMultimodalLM,
        AutoModelForTokenClassification,
        AutoProcessor,
    )

    device = "cuda"
    dtype = torch.bfloat16

    audio = _read_wav_float32(audio_path)
    pieces = _split_long_audio(audio, SAMPLE_RATE, CUDA_CHUNK_SEC)

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForMultimodalLM.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()

    aligner_processor = AutoProcessor.from_pretrained(CUDA_ALIGNER_MODEL)
    aligner = AutoModelForTokenClassification.from_pretrained(
        CUDA_ALIGNER_MODEL, torch_dtype=dtype
    ).to(device).eval()

    word_items = []
    try:
        with torch.no_grad():
            for chunk_wav, offset in pieces:
                inputs = processor.apply_transcription_request(
                    audio=chunk_wav, language="English"
                ).to(device, dtype)
                output_ids = model.generate(**inputs, max_new_tokens=CUDA_MAX_NEW_TOKENS)
                generated = output_ids[:, inputs["input_ids"].shape[1]:]
                parsed = processor.decode(generated, return_format="parsed")[0]
                text = (parsed.get("transcription") or "").strip()
                language = parsed.get("language") or "English"
                if not text:
                    continue
                align_inputs, word_lists = aligner_processor.prepare_forced_aligner_inputs(
                    audio=chunk_wav, transcript=text, language=language,
                )
                align_inputs = align_inputs.to(device, dtype)
                align_out = aligner(**align_inputs)
                timestamps = aligner_processor.decode_forced_alignment(
                    logits=align_out.logits,
                    input_ids=align_inputs["input_ids"],
                    word_lists=word_lists,
                    timestamp_token_id=aligner.config.timestamp_token_id,
                )[0]
                for item in timestamps:
                    word_items.append({
                        "text": item["text"],
                        "start": round(float(item["start_time"]) + offset, 3),
                        "end": round(float(item["end_time"]) + offset, 3),
                    })
    finally:
        del model, aligner
        torch.cuda.empty_cache()

    chunks = _words_to_chunks(word_items)
    output_path.write_text(json.dumps({"segments": [], "chunks": chunks}, ensure_ascii=False, indent=2))
    return output_path


def _fmt_ts(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    h = total_ms // 3_600_000
    m = (total_ms % 3_600_000) // 60_000
    s = (total_ms % 60_000) // 1000
    ms = total_ms % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r'(?<=[。！？])(?!\s*[。！？])|(?<=[.!?])\s+(?!\.)', text.strip())
    return [s for s in parts if s]


def _json_to_srt(json_path: Path, srt_path: Path) -> None:
    with open(json_path) as f:
        data = json.load(f)

    segments = data.get("segments")
    chunks = data.get("chunks", [])

    with open(srt_path, "w") as f:
        seq = 0

        if segments:
            seg_idx = 0
            for chunk in chunks:
                text = chunk.get("text", "").strip()
                if not text:
                    continue

                chunk_start = chunk["start"]
                chunk_end = chunk["end"]

                chunk_segs: list[dict] = []
                while seg_idx < len(segments) and segments[seg_idx]["start"] < chunk_end:
                    if segments[seg_idx]["end"] > chunk_start:
                        chunk_segs.append(segments[seg_idx])
                    seg_idx += 1

                if not chunk_segs:
                    continue

                sentences = _split_sentences(text)

                entries: list[tuple[float, float, str]] = []
                word_pos = 0
                for sentence in sentences:
                    wc = max(1, len(re.findall(r'\S+', sentence)))
                    end_pos = min(word_pos + wc, len(chunk_segs))
                    seg_slice = chunk_segs[word_pos:end_pos]
                    if not seg_slice:
                        break
                    word_pos = end_pos
                    entries.append((seg_slice[0]["start"], seg_slice[-1]["end"], sentence))

                for i, (s_start, s_end, text) in enumerate(entries):
                    padded_end = min(s_end + 0.1, entries[i + 1][0] if i + 1 < len(entries) else chunk_end)
                    seq += 1
                    f.write(f"{seq}\n")
                    f.write(f"{_fmt_ts(s_start)} --> {_fmt_ts(padded_end)}\n")
                    f.write(f"{text}\n\n")
        else:
            for chunk in chunks:
                text = chunk.get("text", "").strip()
                if not text:
                    continue
                chunk_start = chunk["start"]
                chunk_end = chunk["end"]
                chunk_dur = chunk_end - chunk_start

                sentences = _split_sentences(text)
                if len(sentences) == 1:
                    seq += 1
                    f.write(f"{seq}\n")
                    f.write(f"{_fmt_ts(chunk_start)} --> {_fmt_ts(chunk_end)}\n")
                    f.write(f"{sentences[0]}\n\n")
                    continue

                total_chars = sum(len(s) for s in sentences)
                cursor = 0
                for sentence in sentences:
                    seq += 1
                    cursor += len(sentence)
                    s_start = chunk_start + ((cursor - len(sentence)) / total_chars) * chunk_dur
                    s_end = chunk_start + (cursor / total_chars) * chunk_dur
                    f.write(f"{seq}\n")
                    f.write(f"{_fmt_ts(s_start)} --> {_fmt_ts(s_end)}\n")
                    f.write(f"{sentence}\n\n")


def _parse_srt(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    cues = []
    for m in SRT_BLOCK_RE.finditer(text):
        seq, timing, body = m.groups()
        cues.append({"id": int(seq), "timing": timing.strip(), "text": body.strip()})
    return cues


def _write_srt(cues: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for cue in cues:
            f.write(f"{cue['id']}\n{cue['timing']}\n{cue['text']}\n\n")


def _load_hashes(path: Path) -> set[str]:
    if path.exists():
        return set(json.loads(path.read_text()))
    return set()


def _save_hashes(path: Path, hashes: set[str]) -> None:
    path.write_text(json.dumps(sorted(hashes), indent=2) + "\n")


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _swap_corrected_srts(root: Path, tracking_path: Path) -> int:
    corrected_files = sorted(p for p in root.rglob("*.corrected.srt") if not _in_excluded(p, root) and not _is_appledouble(p))
    if not corrected_files:
        return 0
    hashes = _load_hashes(tracking_path)
    swapped = 0
    for cf in corrected_files:
        raw_path = cf.parent / cf.name.replace(".corrected", "")
        content = cf.read_bytes()
        raw_path.write_bytes(content)
        cf.unlink()
        hashes.add(_hash_file(raw_path))
        swapped += 1
    _save_hashes(tracking_path, hashes)
    return swapped


def _load_glossary(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _glossary_summary(glossary: dict) -> str:
    lines = []
    for name, aliases in glossary.get("characters", {}).items():
        alias_str = ", ".join(aliases) if aliases else "none"
        lines.append(f"- {name} (also seen as: {alias_str})")
    for loc in glossary.get("locations", []):
        lines.append(f"- {loc} (location)")
    for term in glossary.get("terms", []):
        lines.append(f"- {term} (term)")
    return "\n".join(lines) if lines else "(empty glossary)"


def _build_correction_prompt(chunk: list[dict], glossary_text: str) -> str:
    cue_lines = "\n".join(f"[{c['id']}] {c['text']}" for c in chunk)
    return f"""You are correcting names and terminology in anime subtitle text against a known glossary.

GLOSSARY (canonical spellings):
{glossary_text}

SUBTITLE CUES:
{cue_lines}

TASK:
Compare each cue's text against the glossary. If a cue contains a likely
misspelling or near-miss of a name/place/term from the glossary, record a
correction. Make each determination once. Do not re-verify the same cue
multiple times. Do NOT correct grammar, punctuation, wording, or anything
not related to names/terminology. Do NOT invent corrections not grounded
in the glossary. If a cue needs no correction, omit it from the output.

Return ONLY a JSON array, no other text, in this exact format:
[{{"cue_id": 12, "wrong": "Taika", "correct": "Taiki"}}]

If no corrections are needed, return an empty array: []
"""


def _http_json(method: str, url: str, payload: dict | None = None,
               timeout: int | None = None, api_key: str | None = None) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method=method,
    )
    with urllib.request.urlopen(req, timeout=timeout if timeout is not None else TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _server_has_loaded_model(chat_url: str, api_key: str | None = None) -> bool:
    """True if the OpenAI-compatible server reports a loaded model.

    The script NEVER triggers a model load — it only uses what the server
    already has loaded. Returns False if the server is unreachable, errors,
    or has no loaded model (caller treats that as 'load a model yourself').
    """
    base = chat_url.rsplit("/chat/completions", 1)[0]  # .../v1
    try:
        data = _http_json("GET", f"{base}/models", timeout=8, api_key=api_key)
    except Exception:
        return False
    items = data.get("data") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return False
    for item in items:
        if not isinstance(item, dict):
            continue
        instances = item.get("loaded_instances")
        if isinstance(instances, list) and instances:
            return True
        if item.get("state") == "loaded" or item.get("status") == "loaded":
            return True
    return False


def _call_llm(prompt: str, endpoint: dict) -> str:
    payload = {
        "model": endpoint["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 4000,
    }
    data = _http_json("POST", endpoint["base"], payload,
                      api_key=endpoint.get("api_key"))
    choice = data["choices"][0]
    if choice.get("finish_reason") == "length":
        print("  [WARN] Response truncated (hit max_tokens) — chunk likely incomplete")
    return choice["message"]["content"]


def _run_chunk(prompt: str, endpoints: list[dict], idx: int) -> str:
    last = None
    for attempt in range(3):
        try:
            return _call_llm(prompt, endpoints[idx % len(endpoints)])
        except Exception as e:
            last = e
            if attempt < 2:
                time.sleep(2 + attempt)
    raise last


def _extract_json_array(text: str) -> list[dict]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []


def _apply_corrections(cues: list[dict], corrections: list[dict]) -> int:
    cue_map = {c["id"]: c for c in cues}
    applied = 0
    for corr in corrections:
        cue = cue_map.get(corr.get("cue_id"))
        wrong = corr.get("wrong")
        correct = corr.get("correct")
        if not cue or not wrong or not correct:
            continue
        pattern = re.compile(rf"\b{re.escape(wrong)}\b", re.IGNORECASE)
        if pattern.search(cue["text"]):
            cue["text"] = pattern.sub(correct, cue["text"])
            applied += 1
    return applied


def _chunked(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _correct_srt_file(srt_path: Path, glossary_text: str, endpoints: list[dict]) -> None:
    cues = _parse_srt(srt_path)
    if not cues:
        raise ValueError("No cues parsed from SRT")

    total_applied = 0
    chunks = list(_chunked(cues, CHUNK_SIZE))
    workers = min(len(chunks), 3)
    prompts = [_build_correction_prompt(c, glossary_text) for c in chunks]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_run_chunk, prompt, endpoints, idx)
            for idx, prompt in enumerate(prompts)
        ]
        done = 0
        for future in as_completed(futures):
            done += 1
            print(f"    [{done}/{len(chunks)}]")
            raw = future.result()
            corrections = _extract_json_array(raw)
            total_applied += _apply_corrections(cues, corrections)

    out_path = srt_path.with_suffix(".corrected.srt")
    _write_srt(cues, out_path)
    print(f"  [OK] {srt_path.name}: {total_applied} correction(s) -> {out_path.name}")


def _find_glossary(start: Path, root: Path) -> Path | None:
    for parent in [start] + list(start.parents):
        candidate = parent / "glossary.json"
        if candidate.is_file():
            return candidate
        if parent == root:
            break
    return None


_SUB_CODEC = {".mp4": "mov_text", ".mkv": "srt", ".mov": "mov_text"}


_FMT_MAP = {".mp4": "mp4", ".mkv": "matroska", ".mov": "mov"}


def _embed_srt(video: Path, srt: Path) -> None:
    codec = _SUB_CODEC.get(video.suffix.lower())
    fmt = _FMT_MAP.get(video.suffix.lower())
    if codec is None or fmt is None:
        return
    subtitle_probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "s",
         "-show_entries", "stream=index", "-of", "csv=p=0", str(video)],
        check=True, capture_output=True, text=True,
    )
    existing_subtitle_count = sum(
        1 for line in subtitle_probe.stdout.splitlines() if line.strip()
    )
    new_subtitle_index = existing_subtitle_count
    # Write temp file in the SAME directory with a valid .mkv/.mp4/.mov extension
    # (CIFS rejects .tmp extensions; ffmpeg can't guess the format)
    tmp = video.with_suffix(".embed_tmp" + video.suffix)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video), "-i", str(srt),
        # Map ALL streams from the original (video, ALL audio, subtitles, attachments)
        # so dual-audio tracks, chapter data, etc. are preserved
        "-map", "0", "-map", "1",
        "-c", "copy", "-c:s", codec,
        f"-metadata:s:s:{new_subtitle_index}", "language=eng",
        "-f", fmt,
        str(tmp),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        # Verify the temp file is valid and contains the newly added subtitle
        # before replacing the original.
        verify = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "s",
             "-show_entries", "stream=codec_type:stream_tags=language",
             "-of", "json", str(tmp)],
            check=True, capture_output=True, text=True,
        )
        streams = json.loads(verify.stdout).get("streams", [])
        if len(streams) <= existing_subtitle_count:
            raise RuntimeError(
                f"Verification failed for {tmp.name}: new subtitle missing "
                "— original preserved"
            )
        if streams[new_subtitle_index].get("tags", {}).get("language") not in (
            "eng", "en", "English"
        ):
            raise RuntimeError(
                f"Verification failed for {tmp.name}: new subtitle is not tagged "
                "English — original preserved"
            )
        os.replace(tmp, video)
    finally:
        tmp.unlink(missing_ok=True)


def _models_url(chat_url: str) -> str:
    base = chat_url.rsplit("/chat/completions", 1)[0].rstrip("/")
    return f"{base}/models"


def _list_models(chat_url: str, api_key: str | None = None) -> list[str]:
    """Return model IDs from an OpenAI-compatible provider's /models endpoint."""
    data = _http_json("GET", _models_url(chat_url), timeout=20, api_key=api_key)
    items = data.get("data") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return []
    model_ids = []
    for item in items:
        if isinstance(item, dict) and item.get("id"):
            model_ids.append(str(item["id"]))
    return list(dict.fromkeys(model_ids))


def _is_local_endpoint(chat_url: str) -> bool:
    host = (urlparse(chat_url).hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "::1"}


def _prompt_text(label: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{label}{suffix}: ").strip()
    return value or default


def _prompt_bool(label: str, default: bool = True) -> bool:
    suffix = "Y/n" if default else "y/N"
    while True:
        value = input(f"{label} [{suffix}]: ").strip().lower()
        if not value:
            return default
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        print("Please answer y or n.")


def _prompt_int(label: str, default: int, minimum: int = 1) -> int:
    while True:
        value = _prompt_text(label, str(default))
        try:
            parsed = int(value)
        except ValueError:
            print("Please enter a whole number.")
            continue
        if parsed >= minimum:
            return parsed
        print(f"Please enter a number greater than or equal to {minimum}.")


def _prompt_secret(label: str) -> str:
    import getpass

    try:
        return getpass.getpass(f"{label}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        raise SystemExit(1)


def _choose_provider_model(endpoint: str, api_key: str | None,
                           fallback: str) -> str:
    print("\nFetching available models from the provider...")
    try:
        models = _list_models(endpoint, api_key)
    except Exception as e:
        print(f"Could not fetch models: {e}")
        models = []
    if not models:
        print("The provider did not return any models.")
        return _prompt_text("Model id", fallback)

    print(f"Found {len(models)} model(s):")
    for index, model in enumerate(models, 1):
        print(f"  {index}. {model}")
    while True:
        choice = _prompt_text("Choose model number", "1")
        try:
            index = int(choice)
        except ValueError:
            index = 0
        if 1 <= index <= len(models):
            return models[index - 1]
        print(f"Please choose a number from 1 to {len(models)}.")


def _interactive_args(parser: argparse.ArgumentParser,
                      existing: argparse.Namespace | None = None) -> argparse.Namespace:
    """Collect a complete run configuration for users launching without flags."""
    args = existing if existing is not None else parser.parse_args([])
    default_model = MODEL if IS_MAC else CUDA_MODEL
    default_root = str(Path.cwd())

    print("\nTranscriber setup")
    print("=" * 48)
    while True:
        input_dir = Path(_prompt_text("Video library", default_root)).expanduser()
        if input_dir.is_dir():
            break
        print(f"Not a directory: {input_dir}")
    args.input_dir = str(input_dir)

    print("\nASR model")
    print(f"  1. Default ({default_model})")
    print("  2. Enter a custom model id")
    while True:
        choice = _prompt_text("Choose", "1")
        if choice == "1":
            args.asr_model = default_model
            break
        if choice == "2":
            args.asr_model = _prompt_text("ASR model id", default_model)
            break
        print("Please choose 1 or 2.")

    args.chunk_size = _prompt_int("Subtitle cues per correction request", args.chunk_size)
    args.timeout = _prompt_int("LLM request timeout in seconds", args.timeout)

    args.no_correction = not _prompt_bool("Correct names and terms with the LLM", True)
    if not args.no_correction:
        args.lm_studio_url = _prompt_text("LLM endpoint", args.lm_studio_url)
        if args.api_key:
            print("An API key is already configured through TRANSCRIBER_API_KEY or --api-key.")
            use_existing = _prompt_bool("Use the configured API key", True)
            if not use_existing:
                args.api_key = _prompt_secret("API key (leave blank for local/no auth)")
        else:
            args.api_key = _prompt_secret("API key (leave blank for local/no auth)")
        args.lm_studio_model = _choose_provider_model(
            args.lm_studio_url, args.api_key, args.lm_studio_model
        )
    args.no_embedding = not _prompt_bool("Embed subtitles back into the videos", True)
    args.keep_srt = _prompt_bool("Keep generated .srt files after embedding", False)
    args.force = _prompt_bool("Reprocess videos that already have subtitle output", False)
    args.dry_run = _prompt_bool("Perform a dry run only", False)
    args.yes = False
    args.interactive = False

    print("\nRun summary")
    print(f"  Library:  {args.input_dir}")
    print(f"  ASR:      {args.asr_model}")
    print(f"  Correct:  {'no' if args.no_correction else f'{args.lm_studio_model} @ {args.lm_studio_url}'}")
    print(f"  API key:  {'configured' if args.api_key else 'none'}")
    print(f"  Embed:    {'no' if args.no_embedding else 'yes'}")
    print(f"  Keep SRT: {'yes' if args.keep_srt else 'no'}")
    print(f"  Reprocess:{'yes' if args.force else 'no'}")
    print(f"  Dry run:  {'yes' if args.dry_run else 'no'}")
    if not _prompt_bool("Start this run", True):
        print("Cancelled.")
        raise SystemExit(0)
    return args


def main() -> int:
    global DRY_RUN, YES, CHUNK_SIZE, TIMEOUT, LM_STUDIO_URL, LM_STUDIO_MODEL, LLM_API_KEY
    parser = argparse.ArgumentParser(
        prog="transcribe.py",
        description="Transcribe anime videos (macOS MLX / Linux CUDA), correct subtitles "
                    "against a glossary via LM Studio, and embed them.",
    )
    parser.add_argument("--interactive", action="store_true",
                        help="Launch the guided setup wizard.")
    parser.add_argument("--input-dir", default=None,
                        help="Video library root (default: current directory).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Scan and report what would be done without changing any files.")
    parser.add_argument("-y", "--yes", action="store_true",
                        help="Non-interactive: auto-accept the default answer for every prompt.")
    parser.add_argument("--asr-model", default=None,
                        help=f"ASR model id (default: {MODEL if IS_MAC else CUDA_MODEL}).")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE,
                        help=f"Subtitle cues per correction chunk (default: {CHUNK_SIZE}).")
    parser.add_argument("--timeout", type=int, default=TIMEOUT,
                        help=f"HTTP timeout in seconds for LLM calls (default: {TIMEOUT}).")
    parser.add_argument("--lm-studio-url", default=LM_STUDIO_URL,
                        help=f"OpenAI-compatible chat completions endpoint (default: {LM_STUDIO_URL}).")
    parser.add_argument("--lm-studio-model", default=LM_STUDIO_MODEL,
                        help=f"LM Studio model id (default: {LM_STUDIO_MODEL}).")
    parser.add_argument("--api-key", default=os.environ.get("TRANSCRIBER_API_KEY"),
                        help="Bearer token for cloud LLM APIs (prefer TRANSCRIBER_API_KEY).")
    parser.add_argument("--no-correction", action="store_true",
                        help="Skip glossary/LLM correction.")
    parser.add_argument("--no-embedding", action="store_true",
                        help="Keep subtitles as files; do not modify video containers.")
    parser.add_argument("--keep-srt", action="store_true",
                        help="Keep generated .srt files after embedding.")
    parser.add_argument("--force", action="store_true",
                        help="Reprocess files even when subtitle output already exists.")
    args = parser.parse_args()

    if args.interactive or len(sys.argv) == 1:
        args = _interactive_args(parser, args)

    if args.asr_model is None:
        args.asr_model = MODEL if IS_MAC else CUDA_MODEL
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    DRY_RUN = args.dry_run
    YES = args.yes
    CHUNK_SIZE = args.chunk_size
    TIMEOUT = args.timeout
    LM_STUDIO_URL = args.lm_studio_url
    LM_STUDIO_MODEL = args.lm_studio_model
    LLM_API_KEY = args.api_key

    _check_tools()

    root = Path(args.input_dir or os.getcwd()).expanduser()
    if not root.is_dir():
        parser.error(f"input directory does not exist: {root}")
    root = root.resolve()
    all_files = _scan_videos(root)
    pending, skipped = _classify(all_files, force=args.force)
    total = len(all_files)
    pending_count = len(pending)
    skipped_count = len(skipped)

    print(f"Found {total} video file(s) — {skipped_count} skipped (no English audio or already transcribed), {pending_count} to process")
    if not pending:
        print("Nothing to do.")
    else:
        print()

    succeeded: list[str] = []
    failed: list[str] = []
    created_srts: list[Path] = []
    correction_failed = 0

    for i, video in enumerate(pending, 1):
        name = video.name
        if DRY_RUN:
            print(f"[{i}/{pending_count}] Would transcribe {name} -> {video.with_suffix('.en.srt').name}")
            succeeded.append(name)
            created_srts.append(video.with_suffix(".en.srt"))
            continue

        print(f"[{i}/{pending_count}] Transcribing {name}...")

        tmp_wav = None
        tmp_json = None
        try:
            # Use a temp directory that's NOT /tmp tmpfs (can fill up on Pi with batch runs).
            # Write WAV beside the video on the NAS — same volume, cleaned up after.
            tmp_wav = video.with_suffix(".extract_tmp.wav")
            _extract_audio(video, tmp_wav)
            tmp_json = _transcribe(tmp_wav, args.asr_model)
            target_srt = video.with_suffix(".en.srt")
            _json_to_srt(tmp_json, target_srt)
            succeeded.append(name)
            created_srts.append(target_srt)
        except subprocess.CalledProcessError:
            print(f"  [ERROR] {name}: subprocess failed")
            failed.append(name)
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
            failed.append(name)
        finally:
            for p in (tmp_wav, tmp_json):
                if p is not None and p.exists():
                    try:
                        p.unlink()
                    except OSError:
                        pass

    print()
    print("=" * 48)
    print(f"  Total:     {total}")
    print(f"  Skipped:   {skipped_count}  (already had .srt)")
    print(f"  Succeeded: {len(succeeded)}")
    print(f"  Failed:    {len(failed)}")
    if failed:
        for name in failed:
            print(f"    - {name}")
    print("=" * 48)

    if not args.no_correction and not created_srts:
        for p in sorted(root.rglob("*.srt")):
            if ".corrected" not in p.stem and not _in_excluded(p, root) and not _is_appledouble(p):
                created_srts.append(p)

    if not args.no_correction and created_srts:
        print()
        print("=" * 48)
        print("  Correction Phase")
        print("=" * 48)
        tracking_path = root / CORRECTED_HASHES_FILE

        existing = [p for p in sorted(root.rglob("*.corrected.srt")) if not _in_excluded(p, root) and not _is_appledouble(p)]
        if existing:
            print(f"  Found {len(existing)} existing corrected subtitle file(s).")
            if DRY_RUN:
                print(f"  Would swap {len(existing)} file(s).")
            else:
                choice = "" if YES else input(
                    "  Swap them into raw .srts now? [1] Yes (default) / [2] No: "
                ).strip()
                if choice in ("", "1"):
                    swapped = _swap_corrected_srts(root, tracking_path)
                    print(f"  Swapped {swapped} file(s).")
            print("=" * 48)
            print()

        _glossary_cache: dict[Path, str] = {}
        corrected_hashes = _load_hashes(tracking_path)
        to_correct: list[tuple[Path, str]] = []
        skipped_no_glossary = 0
        skipped_already = 0
        for srt_path in sorted(created_srts):
            if srt_path.with_suffix(".corrected.srt").exists():
                skipped_already += 1
                continue
            parent = srt_path.parent
            if parent not in _glossary_cache:
                glossary_path = _find_glossary(parent, root)
                if glossary_path is not None:
                    _glossary_cache[parent] = _glossary_summary(_load_glossary(glossary_path))
                    print(f"  Glossary: {glossary_path.relative_to(root)}")
                else:
                    _glossary_cache[parent] = ""
            glossary_text = _glossary_cache[parent]
            if not glossary_text:
                skipped_no_glossary += 1
                continue
            if corrected_hashes and _hash_file(srt_path) in corrected_hashes:
                skipped_already += 1
                continue
            to_correct.append((srt_path, glossary_text))
        endpoints = [{"base": LM_STUDIO_URL, "model": LM_STUDIO_MODEL,
                      "api_key": LLM_API_KEY}]
        if (to_correct and not DRY_RUN and _is_local_endpoint(endpoints[0]["base"])
                and not _server_has_loaded_model(endpoints[0]["base"], LLM_API_KEY)):
            print("  [ERROR] No model is loaded in LM Studio (nothing served at "
                  f"{endpoints[0]['base']}). Load a model first, then re-run — "
                  "the script will NOT load one for you.")
            print("  Skipping the correction phase.")
            corrected_ok = 0
            corrected_fail = 1
            correction_failed = 1
        else:
            corrected_ok = 0
            corrected_fail = 0
            for idx, (srt_path, glossary_text) in enumerate(to_correct, 1):
                total = len(to_correct)
                if DRY_RUN:
                    print(f"  [{idx}/{total}] Would correct {srt_path.name}")
                    corrected_ok += 1
                    continue
                print(f"  [{idx}/{total}] Correcting {srt_path.name}...")
                try:
                    _correct_srt_file(srt_path, glossary_text, endpoints)
                    corrected_ok += 1
                except Exception as e:
                    print(f"  [ERROR] {srt_path.name}: {e}")
                    corrected_fail += 1
            correction_failed = corrected_fail
        print()
        parts = []
        if corrected_ok:
            parts.append(f"Corrected: {corrected_ok}")
        if corrected_fail:
            parts.append(f"Failed: {corrected_fail}")
        if skipped_no_glossary:
            parts.append(f"Skipped (no glossary): {skipped_no_glossary}")
        if skipped_already:
            parts.append(f"Skipped (already corrected): {skipped_already}")
        print(f"  {', '.join(parts)}" if parts else "  No glossaries found.")
        print("=" * 48)

        if corrected_ok and not DRY_RUN:
            print()
            choice = "" if YES else input(
                "  Replace raw .srts with corrected versions? [1] Yes (default) / [2] No: "
            ).strip()
            if choice in ("", "1"):
                swapped = _swap_corrected_srts(root, root / CORRECTED_HASHES_FILE)
                print(f"  Swapped {swapped} file(s). Done.")
            else:
                print("  Skipping swap — both .srt and .corrected.srt kept.")
            print("=" * 48)
        elif corrected_ok and DRY_RUN:
            print()
            print("  Would prompt to replace raw .srts with corrected versions.")
            print("=" * 48)

    if args.no_embedding:
        print("\nEmbed phase skipped (--no-embedding).")
        return 1 if failed or correction_failed else 0

    print()
    print("=" * 48)
    print("  Embed Phase")
    print("=" * 48)
    embedded = 0
    embedding_failed = 0
    for video in sorted(_scan_videos(root)):
        if _has_embedded_english_subs(video):
            continue
        srt = video.with_suffix(".en.srt")
        if not srt.exists():
            continue
        if DRY_RUN:
            action = "delete" if not args.keep_srt else "keep"
            print(f"  Would embed {srt.name} into {video.name}, then {action} the SRT")
            embedded += 1
            continue
        print(f"  Embedding {srt.name} into {video.name}...")
        try:
            _embed_srt(video, srt)
            if not args.keep_srt:
                srt.unlink()
            embedded += 1
        except (subprocess.CalledProcessError, RuntimeError, json.JSONDecodeError) as e:
            stderr = getattr(e, "stderr", None)
            msg = stderr.decode() if isinstance(stderr, bytes) else str(stderr or e)
            print(f"  [ERROR] Failed to embed {srt.name}: {msg[:200]}")
            embedding_failed += 1
    print(f"  Embedded {embedded} subtitle file(s).")
    print("=" * 48)
    return 1 if failed or correction_failed or embedding_failed else 0


if __name__ == "__main__":
    sys.exit(main())
