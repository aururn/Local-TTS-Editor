# Long-form inference

`long_infer.py` is a wrapper for generating audio longer than the model's
per-request 30 second window. It splits long text into natural chunks, loads the
Irodori runtime once, synthesizes each chunk, and merges the chunks into one WAV.

Preview the split without loading the model:

```bash
python long_infer.py \
  --text-file script.txt \
  --dry-run
```

Generate with a reference voice:

```bash
python long_infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-500M-v3 \
  --text-file script.txt \
  --ref-wav path/to/reference.wav \
  --output-wav outputs/long.wav \
  --model-device cuda \
  --codec-device cuda
```

Generate without a reference:

```bash
python long_infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-500M-v3 \
  --text-file script.txt \
  --no-ref \
  --output-wav outputs/long.wav
```

VoiceDesign example:

```bash
python long_infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-500M-v2-VoiceDesign \
  --text-file script.txt \
  --caption "calm, soft, close-mic female voice" \
  --no-ref \
  --output-wav outputs/long_voice_design.wav
```

Useful controls:

- `--chunk-max-seconds`: per-chunk duration cap. The default is `25`; keep this
  at or below `30`.
- `--chars-per-second`: rough text splitting estimate. Lower values make shorter
  chunks.
- `--max-chars`: explicit character limit per chunk, overriding the estimate.
- `--pause-ms`: silence between chunks.
- `--edge-fade-ms`: tiny fade applied to chunk edges to reduce clicks.
- `--normalize-chunks-db`: optional RMS normalization target for each chunk.
- `--seed` and `--seed-mode`: reproducible chunk seeds.
- `--save-chunks-dir` / `--no-save-chunks`: controls intermediate chunk WAVs.

When `--ref-wav` is used, the script encodes it to a cached latent once and
reuses that latent for every chunk, avoiding repeated reference encoding.

## Web UI

Launch the long-form Gradio app:

```bash
python gradio_app_long.py --server-name 127.0.0.1 --server-port 7862
```

The UI exposes split preview, reference audio / reference latent / speaker
embedding inputs, per-chunk generation, and final WAV merging.
