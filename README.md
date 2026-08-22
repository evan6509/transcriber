# Video Transcriber

`transcribe.py` turns videos with English audio into timed English subtitles. It can:

1. Find supported videos recursively.
2. Transcribe audio with Qwen3 ASR.
3. Correct names, places, and domain-specific terminology against a local glossary using an OpenAI-compatible LLM endpoint.
4. Embed the resulting subtitles back into the original video container.

It supports macOS Apple Silicon through MLX and Linux/NVIDIA through CUDA. The program is interactive when launched without arguments, but every operation is also available through command-line flags for scripts and batch jobs.

## Quick start

From the repository directory:

```bash
python3 transcribe.py
```

The setup wizard asks for the video library, ASR model, correction settings, embedding behavior, and confirmation before processing.

To see all noninteractive options:

```bash
python3 transcribe.py --help
```

## Requirements

Common requirements:

- Python 3.10 or newer
- `ffmpeg` and `ffprobe` on `PATH`
- Video files in `.mkv`, `.mp4`, or `.mov` format

Install FFmpeg with one of the following examples:

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

### macOS Apple Silicon

Install and configure the `mlx-qwen3-asr` command from the MLX Qwen3 ASR project. The default model is:

```text
mlx-community/Qwen3-ASR-1.7B-8bit
```

### Linux with NVIDIA CUDA

Install a CUDA-enabled PyTorch build, plus the model dependencies:

```bash
pip install torch transformers numpy
```

The default ASR model is:

```text
Qwen/Qwen3-ASR-1.7B-hf
```

The forced aligner used for word-level timestamps is:

```text
Qwen/Qwen3-ForcedAligner-0.6B-hf
```

The model files are downloaded by Hugging Face the first time they are used.

## Interactive mode

Launching with no arguments starts the guided wizard:

```bash
python3 transcribe.py
```

The wizard lets you choose:

- Video library folder
- Default or custom ASR model
- Subtitle correction and LLM endpoint/model
- Whether to embed subtitles into videos
- Whether to retain `.srt` files
- Whether to reprocess existing subtitle output
- Dry-run mode

When correction is enabled, the wizard asks for an API key using a hidden prompt, queries the provider's OpenAI-compatible `/models` endpoint, and lets you select the model from a numbered list. Leave the key blank for a local LM Studio server.

Use `--interactive` if you want to explicitly request the wizard:

```bash
python3 transcribe.py --interactive
```

## Command-line mode

Process a specific library and keep the generated SRT files:

```bash
python3 transcribe.py \
  --input-dir "/path/to/media" \
  --keep-srt
```

Transcribe only, without correction or video modification:

```bash
python3 transcribe.py \
  --input-dir "/path/to/media" \
  --no-correction \
  --no-embedding \
  --keep-srt
```

Preview the work without modifying files:

```bash
python3 transcribe.py \
  --input-dir "/path/to/media" \
  --dry-run
```

Reprocess videos even when subtitle output already exists:

```bash
python3 transcribe.py \
  --input-dir "/path/to/media" \
  --force \
  --keep-srt
```

Use `-y`/`--yes` for unattended runs. This accepts the prompt to replace corrected sidecar subtitles with their corrected versions.

## LLM correction

Correction is optional. By default, the program expects an already-loaded model at:

```text
http://127.0.0.1:1234/v1/chat/completions
```

LM Studio must be running with an OpenAI-compatible model loaded. The program never loads or unloads the model itself.

For cloud providers, enter the provider's chat-completions endpoint and API key in the wizard. The key is not displayed. You can also provide it noninteractively through an environment variable:

```bash
export TRANSCRIBER_API_KEY="your-token"
python3 transcribe.py --input-dir "/path/to/anime"
```

Or pass it explicitly with `--api-key`:

```bash
python3 transcribe.py \
  --input-dir "/path/to/anime" \
  --lm-studio-url "https://api.example.com/v1/chat/completions" \
  --api-key "$TRANSCRIBER_API_KEY"
```

Interactive mode uses the provider's model list automatically. If the provider does not implement `/models`, the wizard falls back to allowing a model ID to be entered manually.

Override the endpoint and model with:

```bash
python3 transcribe.py \
  --input-dir "/path/to/media" \
  --lm-studio-url "http://127.0.0.1:1234/v1/chat/completions" \
  --lm-studio-model "your-model-id"
```

## Glossaries

Place a `glossary.json` file in a media folder or any parent folder up to the selected library root. The glossary is used for names and terminology only; the correction prompt intentionally avoids grammar and style rewriting.

Example:

```json
{
  "characters": {
    "Dr. Rivera": ["Dr Rivera", "Doctor Rivera"],
    "OpenAI": ["Open A.I."]
  },
  "locations": ["Main office"],
  "terms": ["spectral analysis"]
}
```

The `characters` and `locations` keys can be used for people, places, speakers, or other named entities in your material.

## Output and safety

For a video named `recording.mkv`, the normal workflow creates:

```text
recording.en.srt
recording.en.corrected.srt
```

After confirmation, corrected sidecar subtitles may be swapped into the raw `.srt` name, and the English subtitle can be embedded into the video. Generated SRT files are removed after successful embedding unless `--keep-srt` is selected.

The program writes a validated temporary container before replacing the original video, but users should still maintain backups of important media. Use `--dry-run` before a large batch.

The scanner currently processes explicitly English-tagged audio tracks and skips excluded directories named `Wishlist` and `dump`.

## Dataset and benchmark tools

`dataset_builder.py` is a separate pipeline for generating fine-tuning data without writing outputs next to the source videos:

```bash
python3 dataset_builder.py \
  --dir "/path/to/media" \
  --out-dir "/path/to/training_data" \
  --stage all
```

It supports separate `asr` and `correct` stages, a local `--tmp-dir`, resumable raw subtitle generation, and JSONL training output.

`benchmark.py` compares two correction models on held-out subtitle pairs:

```bash
python3 benchmark.py \
  --dir "/path/to/media" \
  --out-dir "/path/to/training_data" \
  --model-a "base-model" \
  --model-b "fine-tuned-model" \
  --holdout 5
```

The benchmark excludes sources listed in `training.jsonl` unless `--allow-overlap` is explicitly supplied.

## Troubleshooting

### `ffmpeg` or `ffprobe` not found

Install FFmpeg and confirm both commands work:

```bash
ffmpeg -version
ffprobe -version
```

### No videos are found

Check that the selected folder contains `.mkv`, `.mp4`, or `.mov` files and that the files are not inside `Wishlist` or `dump`.

### Videos are skipped

The transcriber requires an audio stream tagged `eng`, `en`, or `English`. Inspect a file with:

```bash
ffprobe -v quiet -show_streams -of json "recording.mkv"
```

### Correction is skipped

Make sure the LLM endpoint is running and a model is loaded. To generate subtitles without correction, select that option in the wizard or pass `--no-correction`.

### CUDA errors

Confirm that the installed PyTorch build can see the GPU:

```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```
