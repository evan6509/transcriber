import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


VIDEO_EXTS = {".mkv", ".mp4", ".avi", ".mov"}
MODEL = "mlx-community/Qwen3-ASR-1.7B-8bit"

LM_STUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"
LM_STUDIO_MODEL = "qwen3.6-27b"
LM_STUDIO_NATIVE = "http://127.0.0.1:1234"
CHUNK_SIZE = 20
TIMEOUT = 600
CORRECTED_HASHES_FILE = "corrected_hashes.json"
EXCLUDE_DIRS = {"Wishlist", "dump"}
DRY_RUN = "--dry-run" in sys.argv


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
    _check_tool("mlx-qwen3-asr")


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


def _classify(files: list[Path]) -> tuple[list[Path], list[Path]]:
    pending: list[Path] = []
    skipped: list[Path] = []
    for f in sorted(files):
        if not _is_english_audio(f):
            skipped.append(f)
            continue
        (skipped if _has_existing_srt(f) else pending).append(f)
    return pending, skipped


def _extract_audio(video: Path, dst: Path) -> None:
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video),
        "-ar", "16000", "-ac", "1",
        str(dst),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def _transcribe(audio_path: Path) -> Path:
    output_path = audio_path.with_suffix(".json")
    cmd = [
        "mlx-qwen3-asr",
        "--model", MODEL,
        "--language", "english",
        "--output-format", "json",
        "--output-dir", str(audio_path.parent),
        "--quiet",
        "--timestamps",
        str(audio_path),
    ]
    subprocess.run(cmd, check=True)
    return output_path


def _fmt_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms == 1000:
        s += 1
        ms = 0
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
                    padded_end = min(s_end + 0.25, entries[i + 1][0] if i + 1 < len(entries) else chunk_end)
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


def _http_json(method: str, url: str, payload: dict | None = None, timeout: int = TIMEOUT) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _normalize_host(raw: str) -> str:
    s = raw.strip().rstrip("/")
    if not s:
        return ""
    if s.startswith("http://") or s.startswith("https://"):
        return s
    if ":" in s:
        return f"http://{s}"
    return f"http://{s}:1234"


def _probe_models(base: str, label: str = "") -> list[dict] | None:
    try:
        data = _http_json("GET", f"{base}/api/v1/models", timeout=8)
    except Exception as e:
        if label:
            tail = str(e).split(":")[-1].strip()[:100] if str(e) else str(e)
            print(f"    {label}: error — {tail}")
        return None
    raw_items: list | None = None
    if isinstance(data, list):
        raw_items = data
    elif isinstance(data, dict):
        for key in ("data", "models", "results"):
            v = data.get(key)
            if isinstance(v, list):
                raw_items = v
                break
        if raw_items is None:
            raw_items = []
    models = []
    for item in raw_items or []:
        if not isinstance(item, dict):
            continue
        mid = item.get("key") or item.get("id") or item.get("identifier") or item.get("name")
        if not mid:
            continue
        instances = item.get("loaded_instances")
        loaded = bool(instances) if isinstance(instances, list) else (
            item.get("state") == "loaded" or item.get("status") == "loaded"
        )
        models.append({"id": str(mid), "loaded": loaded, "type": item.get("type") or "llm"})
    return models


def _select_model(context: str, models: list[dict], default_model: str,
                  *, shared: set[str] = (), local_ids: set[str] = ()) -> str:
    llm = [m for m in models if m.get("type") == "llm"]
    seen: dict[str, dict] = {}
    for m in llm:
        mid = m["id"]
        if mid not in seen:
            seen[mid] = dict(m)
        elif m.get("loaded"):
            seen[mid]["loaded"] = True
    entries = list(seen.values())
    if not entries:
        return ""
    def sort_key(e):
        return (0 if e["id"] in local_ids else 1, e["id"])
    entries.sort(key=sort_key)
    loaded_ids = {e["id"] for e in entries if e.get("loaded")}
    ids_set = {e["id"] for e in entries}
    default = default_model if default_model in ids_set else entries[0]["id"]
    default_num = next(i for i, e in enumerate(entries, 1) if e["id"] == default)
    print(f"    {context} — available LLMs:")
    for i, e in enumerate(entries, 1):
        m = f"      [{i}] {e['id']}"
        if e["id"] in loaded_ids:
            m += "  *"
        if e["id"] in shared:
            m += "  (all servers)"
        print(m)
    while True:
        raw = input(f"    Select model number [default {default_num} ({default})]: ").strip()
        if not raw:
            return default
        if raw in ids_set:
            return raw
        try:
            n = int(raw)
            if 1 <= n <= len(entries):
                return entries[n - 1]["id"]
        except ValueError:
            pass
        print(f"    Invalid. Enter a number 1–{len(entries)} or the model id.")


def _prompt_yes_no(prompt: str) -> bool:
    return input(f"{prompt} [Y/n]: ").strip().lower() not in ("n", "no")


def _ensure_model(base: str, model: str, allow_download: bool) -> bool:
    models = _probe_models(base)
    if models is None:
        print(f"  [WARN] {base}: server unreachable — skipping")
        return False
    ids = [m["id"] for m in models]
    if model not in ids:
        if not allow_download or not _prompt_yes_no(
            f"  {base}: model '{model}' is not installed. Download it (can be large)?"
        ):
            print(f"  [WARN] {base}: model '{model}' not installed — skipping")
            return False
        try:
            job = _http_json("POST", f"{base}/api/v1/models/download", {"model": model}, timeout=30)
        except Exception as e:
            print(f"  [ERROR] {base}: download start failed: {e}")
            return False
        job_id = job.get("job_id") or job.get("id")
        if not job_id:
            print(f"  [ERROR] {base}: no job_id in download response: {job}")
            return False
        for _ in range(300):
            try:
                st = _http_json("GET", f"{base}/api/v1/models/download/status/{job_id}", timeout=30)
            except Exception as e:
                print(f"  [ERROR] {base}: download status error: {e}")
                return False
            status = st.get("status")
            print(f"  {base}: download status = {status}")
            if status in ("completed", "done"):
                break
            if status == "failed":
                print(f"  [ERROR] {base}: model download failed")
                return False
            time.sleep(3)
        else:
            print(f"  [ERROR] {base}: download timed out")
            return False
        models = _probe_models(base) or []
    if not any(m.get("loaded") for m in models if m["id"] == model):
        try:
            _http_json("POST", f"{base}/api/v1/models/load", {"model": model}, timeout=120)
            print(f"  {base}: loaded {model}")
        except Exception as e:
            print(f"  [ERROR] {base}: model load failed: {e}")
            return False
    return True


def _pick_model(host: str, models: list[dict]) -> str:
    ids = {m["id"] for m in models if m.get("type") == "llm"}
    if not ids:
        return ""
    preferred = next((i for i in ids if LM_STUDIO_MODEL in i.lower()),
                     next((i for i in ids if "qwen" in i.lower()),
                          next(iter(ids))))
    return _select_model(host, models, preferred)


def _choose_local_native() -> list[dict]:
    base = LM_STUDIO_NATIVE
    models = _probe_models(base)
    if not models:
        print("  [WARN] Local native API unreachable/empty — falling back to OpenAI-compatible endpoint.")
        return [{"base": LM_STUDIO_URL, "protocol": "openai", "model": LM_STUDIO_MODEL}]
    model = _pick_model("127.0.0.1", models)
    if _ensure_model(base, model, allow_download=True):
        return [{"base": base, "protocol": "native", "model": model}]
    return [{"base": LM_STUDIO_URL, "protocol": "openai", "model": LM_STUDIO_MODEL}]


def _choose_remote_servers() -> list[dict]:
    raw = input(
        "    Enter server IPs (space/comma separated, e.g. 192.168.1.10 192.168.1.11:1235)\n    -> "
    ).strip()
    hosts = [h for h in re.split(r"[,\s]+", raw) if h]
    if not hosts:
        print("    No servers entered — falling back to local LM Studio.")
        return _choose_local_native()
    bases = [_normalize_host(h) for h in hosts]
    probed = []
    for base in bases:
        models = None
        for attempt in range(2):
            models = _probe_models(base, label=base)
            if models is not None:
                break
            if attempt == 0:
                print(f"    {base}: retrying...")
        probed.append((base, models))
    for base, models in probed:
        if models is None:
            print(f"    {base}: unreachable")
        else:
            detail = ", ".join(f"{m['id']}{'*' if m.get('loaded') else ''}" for m in models) or "(no models)"
            print(f"    {base}: {detail}")
    live = [(base, m) for base, m in probed if m is not None]
    if not live:
        print("  Remote servers unreachable — falling back to local.")
        return _choose_local_native()
    live_llm = [(base, [m for m in models if m.get("type") == "llm"]) for base, models in live]
    all_ids = sorted({m["id"] for _, models in live_llm for m in models})
    if not all_ids:
        print("  Reachable servers but no LLM models installed — falling back to local.")
        return _choose_local_native()
    shared = {i for i in all_ids if all(i in [m["id"] for m in models] for _, models in live_llm)}
    candidates = list(shared) if shared else all_ids
    default_model = next((i for i in candidates if LM_STUDIO_MODEL in i.lower()),
                         next((i for i in candidates if "qwen" in i.lower()),
                              candidates[0]))
    local_base = _normalize_host(LM_STUDIO_NATIVE)
    local_ids = set()
    for base, models in live_llm:
        if base == local_base:
            local_ids = {m["id"] for m in models}
            break
    all_raw: list[dict] = []
    for _, models in live:
        all_raw.extend(models)
    model = _select_model("every server", all_raw, default_model, shared=shared, local_ids=local_ids)
    endpoints = []
    for base, _ in live:
        if _ensure_model(base, model, allow_download=True):
            endpoints.append({"base": base, "protocol": "native", "model": model})
    if not endpoints:
        print("  No servers could load the model — falling back to local.")
        return _choose_local_native()
    print(f"    Using {len(endpoints)} server(s) with model '{model}'.")
    return endpoints


def _choose_backend() -> list[dict]:
    print("  Correction backend:")
    print("    [1] Local LM Studio (default) — native API (127.0.0.1:1234)")
    print("    [2] Remote LM Studio servers — sync/download/load the same model on each")
    print("    [3] Local OpenAI-compatible endpoint (current method) — 127.0.0.1:1234/v1/chat/completions")
    choice = input("    Select [1]: ").strip()
    if choice == "2":
        return _choose_remote_servers()
    if choice == "3":
        return [{"base": LM_STUDIO_URL, "protocol": "openai", "model": LM_STUDIO_MODEL}]
    return _choose_local_native()


def _call_llm_openai(prompt: str, endpoint: dict) -> str:
    payload = {
        "model": endpoint["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 4000,
    }
    data = _http_json("POST", endpoint["base"], payload)
    choice = data["choices"][0]
    if choice.get("finish_reason") == "length":
        print("  [WARN] Response truncated (hit max_tokens) — chunk likely incomplete")
    return choice["message"]["content"]


def _call_llm_native(prompt: str, endpoint: dict) -> str:
    payload = {
        "model": endpoint["model"],
        "input": prompt,
        "temperature": 0.0,
        "max_output_tokens": 4000,
        "store": False,
    }
    data = _http_json("POST", f"{endpoint['base']}/api/v1/chat", payload)
    parts = [
        item["content"]
        for item in data.get("output", [])
        if isinstance(item, dict) and item.get("type") == "message"
    ]
    stats = data.get("stats") or {}
    if stats.get("total_output_tokens", 0) >= 4000:
        print("  [WARN] Response likely truncated (hit max_output_tokens) — chunk may be incomplete")
    return "".join(parts)


def _call_llm(prompt: str, endpoint: dict) -> str:
    if endpoint.get("protocol") == "native":
        return _call_llm_native(prompt, endpoint)
    return _call_llm_openai(prompt, endpoint)


def _run_chunk(prompt: str, endpoints: list[dict], idx: int) -> str:
    attempts = min(len(endpoints), 2)
    last = None
    for offset in range(attempts):
        try:
            return _call_llm(prompt, endpoints[(idx + offset) % len(endpoints)])
        except Exception as e:
            last = e
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
        if wrong in cue["text"]:
            cue["text"] = cue["text"].replace(wrong, correct)
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
    workers = min(len(chunks), 5 * len(endpoints))
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


_SUB_CODEC = {".mp4": "mov_text", ".mkv": "srt"}


_FMT_MAP = {".mp4": "mp4", ".mkv": "matroska", ".avi": "avi", ".mov": "mov"}


def _embed_srt(video: Path, srt: Path) -> None:
    codec = _SUB_CODEC.get(video.suffix.lower())
    fmt = _FMT_MAP.get(video.suffix.lower())
    if codec is None or fmt is None:
        return
    # Write temp file in the SAME directory with a valid .mkv/.mp4 extension
    # (CIFS rejects .tmp extensions; ffmpeg can't guess the format)
    tmp = video.with_suffix(".embed_tmp" + video.suffix)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video), "-i", str(srt),
        # Map ALL streams from the original (video, ALL audio, subtitles, attachments)
        # so dual-audio tracks, chapter data, etc. are preserved
        "-map", "0", "-map", "1",
        "-c", "copy", "-c:s", codec,
        "-metadata:s:s:0", "language=eng",
        "-f", fmt,
        str(tmp),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    # Verify the temp file is valid before replacing the original
    verify = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "stream=codec_type", "-of", "csv", str(tmp)],
        capture_output=True, text=True
    )
    if verify.returncode != 0 or not verify.stdout.strip():
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Verification failed for {tmp.name} — original preserved")
    os.replace(tmp, video)


def main() -> None:
    _check_tools()

    root = Path(os.getcwd())
    all_files = _scan_videos(root)
    pending, skipped = _classify(all_files)
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
            tmp_json = _transcribe(tmp_wav)
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

    if not created_srts:
        for p in sorted(root.rglob("*.srt")):
            if ".corrected" not in p.stem and not _in_excluded(p, root) and not _is_appledouble(p):
                created_srts.append(p)

    if created_srts:
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
                choice = input(
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
        endpoints = []
        if to_correct and not DRY_RUN:
            endpoints = _choose_backend()
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
            choice = input(
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

    print()
    print("=" * 48)
    print("  Embed Phase")
    print("=" * 48)
    embedded = 0
    for video in sorted(_scan_videos(root)):
        if _has_embedded_english_subs(video):
            continue
        srt = video.with_suffix(".en.srt")
        if not srt.exists():
            continue
        if DRY_RUN:
            print(f"  Would embed {srt.name} into {video.name}, then delete {srt.name}")
            embedded += 1
            continue
        print(f"  Embedding {srt.name} into {video.name}...")
        try:
            _embed_srt(video, srt)
            srt.unlink()
            embedded += 1
        except subprocess.CalledProcessError as e:
            msg = e.stderr.decode() if isinstance(e.stderr, bytes) else str(e.stderr)
            print(f"  [ERROR] Failed to embed {srt.name}: {msg[:200]}")
    print(f"  Embedded {embedded} subtitle file(s).")
    print("=" * 48)


if __name__ == "__main__":
    main()
