#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Any


SENTENCE_ENDERS = set("。．.!?！？")
SOFT_BREAKS = set("、，,;；:：")
CLOSERS = set("」』”’）)]｝】》〉")
JAPANESE_TEXT_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")


def _parse_optional_float(value: str) -> float | None:
    raw = str(value).strip().lower()
    if raw in {"none", "null", "off", "disable", "disabled"}:
        return None
    try:
        out = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Expected float or one of [none, null, off, disable, disabled]."
        ) from exc
    if not math.isfinite(out):
        raise argparse.ArgumentTypeError(f"Expected finite float for value={value!r}.")
    return out


def _visible_len(text: str) -> int:
    return len(re.sub(r"\s+", "", text))


def _flush_buffer(buffer: list[str], out: list[str]) -> None:
    piece = "".join(buffer).strip()
    buffer.clear()
    if piece:
        out.append(piece)


def split_sentences(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    sentences: list[str] = []
    buffer: list[str] = []
    pending_sentence_end = False

    for ch in normalized:
        if pending_sentence_end and ch not in CLOSERS:
            _flush_buffer(buffer, sentences)
            pending_sentence_end = False

        if ch == "\n":
            _flush_buffer(buffer, sentences)
            pending_sentence_end = False
            continue

        buffer.append(ch)
        if ch in SENTENCE_ENDERS:
            pending_sentence_end = True

    _flush_buffer(buffer, sentences)
    return sentences


def _join_text(left: str, right: str) -> str:
    if not left:
        return right
    if not right:
        return left
    if left[-1].isspace() or right[0].isspace():
        return left + right
    if left[-1].isascii() and right[0].isascii() and left[-1].isalnum() and right[0].isalnum():
        return left + " " + right
    if left[-1] in ".!?;:" and right[0].isascii():
        return left + " " + right
    return left + right


def _hard_split(text: str, max_chars: int) -> list[str]:
    parts: list[str] = []
    remaining = text.strip()
    while _visible_len(remaining) > max_chars:
        visible_count = 0
        cut_index = len(remaining)
        preferred_index: int | None = None
        preferred_visible_count = 0

        for i, ch in enumerate(remaining):
            if not ch.isspace():
                visible_count += 1
            if ch.isspace() or ch in SOFT_BREAKS:
                preferred_index = i + 1
                preferred_visible_count = visible_count
            if visible_count >= max_chars:
                cut_index = i + 1
                break

        if (
            preferred_index is not None
            and preferred_visible_count >= max(1, int(max_chars * 0.65))
        ):
            cut_index = preferred_index

        part = remaining[:cut_index].strip()
        if not part:
            part = remaining[: max(1, cut_index)].strip()
            cut_index = max(1, cut_index)
        parts.append(part)
        remaining = remaining[cut_index:].strip()

    if remaining:
        parts.append(remaining)
    return parts


def _pack_units(units: list[str], max_chars: int) -> list[str]:
    chunks: list[str] = []
    current = ""

    for raw_unit in units:
        unit = raw_unit.strip()
        if not unit:
            continue
        if _visible_len(unit) > max_chars:
            for part in _hard_split(unit, max_chars):
                if current:
                    chunks.append(current)
                    current = ""
                chunks.append(part)
            continue

        candidate = _join_text(current, unit)
        if current and _visible_len(candidate) > max_chars:
            chunks.append(current)
            current = unit
        else:
            current = candidate

    if current:
        chunks.append(current)
    return chunks


def _split_long_sentence(sentence: str, max_chars: int) -> list[str]:
    pieces: list[str] = []
    buffer: list[str] = []
    for ch in sentence.strip():
        buffer.append(ch)
        if ch in SOFT_BREAKS:
            _flush_buffer(buffer, pieces)
    _flush_buffer(buffer, pieces)
    if len(pieces) <= 1:
        return _hard_split(sentence, max_chars)
    return _pack_units(pieces, max_chars)


def chunk_text(text: str, *, max_chars: int) -> list[str]:
    if max_chars <= 0:
        raise ValueError(f"max_chars must be > 0, got {max_chars}")
    units: list[str] = []
    for sentence in split_sentences(text):
        if _visible_len(sentence) > max_chars:
            units.extend(_split_long_sentence(sentence, max_chars))
        else:
            units.append(sentence)
    return _pack_units(units, max_chars)


def _strip_trailing_closers(text: str) -> tuple[str, str]:
    body = text.rstrip()
    closers = ""
    while body and body[-1] in CLOSERS:
        closers = body[-1] + closers
        body = body[:-1].rstrip()
    return body, closers


def _has_sentence_end(text: str) -> bool:
    body, _closers = _strip_trailing_closers(text)
    return bool(body) and body[-1] in SENTENCE_ENDERS


def _boundary_punctuation(text: str) -> str:
    return "。" if JAPANESE_TEXT_RE.search(text) else "."


def _punctuate_boundary(text: str) -> str:
    trailing_ws_len = len(text) - len(text.rstrip())
    trailing_ws = text[len(text) - trailing_ws_len :] if trailing_ws_len else ""
    body, closers = _strip_trailing_closers(text)
    if not body or body[-1] in SENTENCE_ENDERS:
        return text
    if body[-1] in SOFT_BREAKS:
        body = body[:-1].rstrip()
    if not body or body[-1] in SENTENCE_ENDERS:
        return body + closers + trailing_ws
    return body + _boundary_punctuation(body) + closers + trailing_ws


def stabilize_chunk_boundaries(chunks: list[str], *, mode: str) -> list[str]:
    if mode == "off":
        return list(chunks)
    if mode != "punctuate":
        raise ValueError(f"Unsupported boundary mode: {mode}")
    out: list[str] = []
    last_index = len(chunks) - 1
    for i, chunk in enumerate(chunks):
        out.append(chunk if i == last_index or _has_sentence_end(chunk) else _punctuate_boundary(chunk))
    return out


def _preview_text(text: str, limit: int = 72) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _read_long_text(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str:
    if args.text is not None and args.text_file is not None:
        parser.error("Use either --text or --text-file, not both.")
    if args.text_file is not None:
        if str(args.text_file) == "-":
            return sys.stdin.read()
        text_path = Path(str(args.text_file)).expanduser()
        if not text_path.is_file():
            parser.error(
                f"Text file not found: {text_path}. "
                "Create the file first, pass an existing path, or use --text directly."
            )
        return text_path.read_text(encoding="utf-8")
    if args.text is not None:
        return str(args.text)
    parser.error("Specify --text or --text-file.")
    raise AssertionError("unreachable")


def _validate_optional_file_arg(
    parser: argparse.ArgumentParser,
    *,
    value: str | None,
    label: str,
) -> str | None:
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_file():
        parser.error(f"{label} not found: {path}. Pass an existing file path.")
    return str(path)


def _resolve_checkpoint_path(args: argparse.Namespace) -> str:
    if args.checkpoint is not None:
        checkpoint_path = Path(str(args.checkpoint)).expanduser()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        print(f"[checkpoint] using local file: {checkpoint_path}", flush=True)
        return str(checkpoint_path)

    from huggingface_hub import hf_hub_download

    repo_id = str(args.hf_checkpoint).strip()
    if repo_id == "":
        raise ValueError("hf_checkpoint must be non-empty.")
    checkpoint_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
    print(
        f"[checkpoint] downloaded model.safetensors from hf://{repo_id} -> {checkpoint_path}",
        flush=True,
    )
    return str(checkpoint_path)


def _chunk_seed(base_seed: int | None, chunk_index: int, mode: str) -> int | None:
    if base_seed is None or mode == "random":
        return None
    if mode == "same":
        return int(base_seed)
    return int(base_seed) + int(chunk_index) - 1


def _fade_edges(audio: Any, sample_rate: int, fade_ms: float) -> Any:
    if fade_ms <= 0:
        return audio
    import torch

    fade_samples = int(float(sample_rate) * float(fade_ms) / 1000.0)
    fade_samples = min(fade_samples, int(audio.shape[-1]) // 2)
    if fade_samples <= 0:
        return audio
    out = audio.clone()
    ramp = torch.linspace(
        0.0,
        1.0,
        fade_samples,
        dtype=out.dtype,
        device=out.device,
    ).unsqueeze(0)
    out[..., :fade_samples] = out[..., :fade_samples] * ramp
    out[..., -fade_samples:] = out[..., -fade_samples:] * ramp.flip(dims=[-1])
    return out


def _normalize_rms(audio: Any, target_db: float | None, peak_limit: float = 0.98) -> Any:
    if target_db is None:
        return audio
    import torch

    out = audio.clone()
    rms = torch.sqrt(torch.mean(out.float() ** 2)).clamp_min(1e-8)
    current_db = 20.0 * torch.log10(rms).item()
    gain = 10.0 ** ((float(target_db) - current_db) / 20.0)
    out = out * gain
    peak = torch.max(torch.abs(out)).item()
    if peak > peak_limit:
        out = out * (float(peak_limit) / peak)
    return out


def merge_audios(
    audios: list[Any],
    *,
    sample_rate: int,
    pause_ms: float,
    edge_fade_ms: float,
    normalize_chunks_db: float | None,
) -> Any:
    if not audios:
        raise ValueError("No audio chunks to merge.")

    import torch

    processed = [
        _fade_edges(
            _normalize_rms(audio, normalize_chunks_db),
            sample_rate=sample_rate,
            fade_ms=edge_fade_ms,
        )
        for audio in audios
    ]
    pause_samples = max(0, int(float(sample_rate) * float(pause_ms) / 1000.0))
    if pause_samples <= 0:
        return torch.cat(processed, dim=-1)

    channels = int(processed[0].shape[0])
    silence = torch.zeros(
        (channels, pause_samples),
        dtype=processed[0].dtype,
        device=processed[0].device,
    )
    merged: list[Any] = []
    for i, audio in enumerate(processed):
        if i > 0:
            merged.append(silence)
        merged.append(audio)
    return torch.cat(merged, dim=-1)


def _encode_reference_latent_once(
    *,
    runtime: Any,
    args: argparse.Namespace,
    chunk_dir: Path,
) -> str | None:
    if args.ref_wav is None:
        return None

    import torch
    from irodori_tts.inference_runtime import _load_audio

    chunk_dir.mkdir(parents=True, exist_ok=True)
    latent_path = chunk_dir / "reference.latent.pt"
    print(f"[reference] encoding reference once: {args.ref_wav}", flush=True)
    wav, sr = _load_audio(args.ref_wav)
    if args.max_ref_seconds is not None and args.max_ref_seconds > 0:
        max_ref_samples = max(1, int(float(args.max_ref_seconds) * float(sr)))
        if wav.shape[1] > max_ref_samples:
            print(
                "[reference] trimming reference "
                f"from {float(wav.shape[1]) / float(sr):.2f}s "
                f"to {float(max_ref_samples) / float(sr):.2f}s",
                flush=True,
            )
            wav = wav[:, :max_ref_samples]
    ref_latent = runtime.codec.encode_waveform(
        wav.unsqueeze(0),
        sample_rate=int(sr),
        normalize_db=args.ref_normalize_db,
        ensure_max=bool(args.ref_ensure_max),
    ).cpu()
    torch.save(ref_latent[0], latent_path)
    print(f"[reference] cached latent: {latent_path}", flush=True)
    return str(latent_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Long-form inference for Irodori-TTS. Text is split into <=30s chunks, "
            "generated with one loaded runtime, then merged into one WAV."
        )
    )
    checkpoint_group = parser.add_mutually_exclusive_group(required=False)
    checkpoint_group.add_argument(
        "--checkpoint",
        default=None,
        help="Local model checkpoint path (.pt or .safetensors).",
    )
    checkpoint_group.add_argument(
        "--hf-checkpoint",
        default=None,
        help="Hugging Face model repo id to download model.safetensors from.",
    )

    text_group = parser.add_argument_group("long text")
    text_group.add_argument("--text", default=None, help="Long text to synthesize.")
    text_group.add_argument(
        "--text-file",
        default=None,
        help="UTF-8 text file to synthesize. Use '-' to read stdin.",
    )
    text_group.add_argument("--output-wav", default="outputs/long.wav")
    text_group.add_argument(
        "--chunk-max-seconds",
        type=float,
        default=25.0,
        help="Per-request duration cap. Keep <=30 for released checkpoints.",
    )
    text_group.add_argument(
        "--chars-per-second",
        type=float,
        default=6.0,
        help="Rough chunking estimate. max_chars = chunk_max_seconds * chars_per_second.",
    )
    text_group.add_argument(
        "--max-chars",
        type=int,
        default=None,
        help="Override automatic chunk character limit.",
    )
    text_group.add_argument(
        "--pause-ms",
        type=float,
        default=220.0,
        help="Silence inserted between generated chunks.",
    )
    text_group.add_argument(
        "--edge-fade-ms",
        type=float,
        default=5.0,
        help="Tiny fade-in/out per chunk to avoid boundary clicks.",
    )
    text_group.add_argument(
        "--normalize-chunks-db",
        type=_parse_optional_float,
        default=None,
        help="Optional RMS normalization target dBFS for each chunk. Use 'none' to disable.",
    )
    text_group.add_argument(
        "--save-chunks-dir",
        default=None,
        help="Directory for chunk WAVs and cached reference latent.",
    )
    text_group.add_argument(
        "--save-chunks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save generated chunk WAV files (default: enabled).",
    )
    text_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show the text split; do not load the model.",
    )
    text_group.add_argument(
        "--boundary-mode",
        choices=["punctuate", "off"],
        default="punctuate",
        help=(
            "How to stabilize chunk endings for synthesis. 'punctuate' adds a sentence "
            "ending to non-final chunks that do not already end like a sentence."
        ),
    )

    parser.add_argument(
        "--lora-adapter",
        default=None,
        help="Optional PEFT LoRA adapter directory to load dynamically.",
    )
    parser.add_argument(
        "--caption",
        default=None,
        help="Optional caption/style-control text for caption-enabled checkpoints.",
    )
    parser.add_argument(
        "--model-device",
        default=None,
        help="Model inference device (default: Irodori runtime auto-detect).",
    )
    parser.add_argument(
        "--model-precision",
        choices=["fp32", "bf16"],
        default="fp32",
        help="Model precision for weights/compute.",
    )
    parser.add_argument(
        "--codec-device",
        default=None,
        help="Codec device (default: same auto-detect as Irodori infer.py).",
    )
    parser.add_argument(
        "--codec-precision",
        choices=["fp32", "bf16"],
        default="fp32",
        help="Codec precision for weights/compute.",
    )
    parser.add_argument(
        "--codec-deterministic-encode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic DACVAE encode path.",
    )
    parser.add_argument(
        "--codec-deterministic-decode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic DACVAE decode path.",
    )
    parser.add_argument("--codec-repo", default="Aratako/Semantic-DACVAE-Japanese-32dim")
    parser.add_argument("--max-ref-seconds", type=float, default=30.0)
    parser.add_argument("--ref-normalize-db", type=_parse_optional_float, default=-16.0)
    parser.add_argument("--ref-ensure-max", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-text-len", type=int, default=None)
    parser.add_argument("--max-caption-len", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=40)
    parser.add_argument(
        "--t-schedule-mode",
        choices=["linear", "sway"],
        default="linear",
        help="Timestep schedule for RF Euler sampling.",
    )
    parser.add_argument("--sway-coeff", type=float, default=-1.0)
    parser.add_argument("--duration-scale", type=float, default=1.0)
    parser.add_argument(
        "--decode-mode",
        choices=["sequential", "batch"],
        default="sequential",
    )
    parser.add_argument("--compile-model", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compile-dynamic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cfg-scale-text", type=float, default=3.0)
    parser.add_argument("--cfg-scale-caption", type=float, default=3.0)
    parser.add_argument("--cfg-scale-speaker", type=float, default=5.0)
    parser.add_argument(
        "--cfg-guidance-mode",
        choices=["independent", "joint", "alternating"],
        default="independent",
    )
    parser.add_argument("--cfg-scale", type=float, default=None)
    parser.add_argument("--cfg-min-t", type=float, default=0.5)
    parser.add_argument("--cfg-max-t", type=float, default=1.0)
    parser.add_argument("--truncation-factor", type=float, default=None)
    parser.add_argument("--rescale-k", type=float, default=None)
    parser.add_argument("--rescale-sigma", type=float, default=None)
    parser.add_argument("--context-kv-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--speaker-kv-scale", type=float, default=None)
    parser.add_argument("--speaker-kv-min-t", type=float, default=0.9)
    parser.add_argument("--speaker-kv-max-layers", type=int, default=None)
    parser.add_argument(
        "--speaker-uncond-mode",
        choices=["mask", "noise"],
        default="mask",
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--seed-mode",
        choices=["offset", "same", "random"],
        default="same",
        help="How to derive per-chunk seeds when --seed is set.",
    )
    parser.add_argument("--trim-tail", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tail-window-size", type=int, default=20)
    parser.add_argument("--tail-std-threshold", type=float, default=0.05)
    parser.add_argument("--tail-mean-threshold", type=float, default=0.1)
    parser.add_argument("--show-timings", action=argparse.BooleanOptionalAction, default=False)

    ref_group = parser.add_mutually_exclusive_group(required=False)
    ref_group.add_argument("--ref-wav", default=None)
    ref_group.add_argument("--ref-latent", default=None)
    ref_group.add_argument("--ref-embed", default=None)
    ref_group.add_argument("--no-ref", action="store_true")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.chunk_max_seconds <= 0:
        parser.error("--chunk-max-seconds must be > 0.")
    if args.chunk_max_seconds > 30.0:
        parser.error("--chunk-max-seconds should be <= 30.0 for released Irodori checkpoints.")
    if args.chars_per_second <= 0:
        parser.error("--chars-per-second must be > 0.")

    args.ref_wav = _validate_optional_file_arg(parser, value=args.ref_wav, label="Reference WAV")
    args.ref_latent = _validate_optional_file_arg(
        parser,
        value=args.ref_latent,
        label="Reference latent",
    )
    args.ref_embed = _validate_optional_file_arg(
        parser,
        value=args.ref_embed,
        label="Speaker embedding",
    )

    long_text = _read_long_text(args, parser)
    max_chars = (
        int(args.max_chars)
        if args.max_chars is not None
        else max(1, int(args.chunk_max_seconds * args.chars_per_second))
    )
    chunks = chunk_text(long_text, max_chars=max_chars)
    if not chunks:
        parser.error("Text produced no chunks.")
    synthesis_chunks = stabilize_chunk_boundaries(chunks, mode=str(args.boundary_mode))

    total_chars = sum(_visible_len(chunk) for chunk in chunks)
    print(
        f"[chunking] chunks={len(chunks)} max_chars={max_chars} "
        f"visible_chars={total_chars} chunk_max_seconds={args.chunk_max_seconds:.2f}",
        flush=True,
    )
    for i, chunk in enumerate(chunks, start=1):
        est_seconds = _visible_len(chunk) / float(args.chars_per_second)
        print(
            f"[chunk {i:03d}/{len(chunks):03d}] chars={_visible_len(chunk)} "
            f"est={est_seconds:.1f}s text={_preview_text(chunk)}",
            flush=True,
        )
        synth_chunk = synthesis_chunks[i - 1]
        if synth_chunk != chunk:
            print(f"[boundary {i:03d}] synthesis_text={_preview_text(synth_chunk)}", flush=True)

    if args.dry_run:
        return
    if args.checkpoint is None and args.hf_checkpoint is None:
        parser.error("Specify --checkpoint or --hf-checkpoint unless --dry-run is set.")

    from irodori_tts.inference_runtime import (
        InferenceRuntime,
        RuntimeKey,
        SamplingRequest,
        default_runtime_device,
        resolve_cfg_scales,
        save_wav,
    )

    model_device = args.model_device or default_runtime_device()
    codec_device = args.codec_device or default_runtime_device()
    output_path = Path(str(args.output_wav))
    chunk_dir = (
        Path(str(args.save_chunks_dir))
        if args.save_chunks_dir is not None
        else output_path.with_name(f"{output_path.stem}_chunks")
    )

    checkpoint_path = _resolve_checkpoint_path(args)
    runtime = InferenceRuntime.from_key(
        RuntimeKey(
            checkpoint=checkpoint_path,
            model_device=str(model_device),
            codec_repo=str(args.codec_repo),
            model_precision=str(args.model_precision),
            codec_device=str(codec_device),
            codec_precision=str(args.codec_precision),
            codec_deterministic_encode=bool(args.codec_deterministic_encode),
            codec_deterministic_decode=bool(args.codec_deterministic_decode),
            compile_model=bool(args.compile_model),
            compile_dynamic=bool(args.compile_dynamic),
        )
    )

    if runtime.model_cfg.use_speaker_condition and not (
        args.no_ref
        or args.ref_wav is not None
        or args.ref_latent is not None
        or args.ref_embed is not None
    ):
        parser.error(
            "speaker-conditioned checkpoints require one of --ref-wav, --ref-latent, "
            "--ref-embed, or --no-ref."
        )

    cached_ref_latent = _encode_reference_latent_once(
        runtime=runtime,
        args=args,
        chunk_dir=chunk_dir,
    )
    ref_wav = None if cached_ref_latent is not None else args.ref_wav
    ref_latent = cached_ref_latent if cached_ref_latent is not None else args.ref_latent

    cfg_scale_text, cfg_scale_caption, cfg_scale_speaker, scale_messages = resolve_cfg_scales(
        cfg_guidance_mode=str(args.cfg_guidance_mode),
        cfg_scale_text=float(args.cfg_scale_text),
        cfg_scale_caption=float(args.cfg_scale_caption),
        cfg_scale_speaker=float(args.cfg_scale_speaker),
        cfg_scale=float(args.cfg_scale) if args.cfg_scale is not None else None,
        use_caption_condition=bool(
            runtime.model_cfg.use_caption_condition
            and args.caption is not None
            and str(args.caption).strip() != ""
        ),
        use_speaker_condition=bool(runtime.model_cfg.use_speaker_condition),
    )
    for msg in scale_messages:
        print(msg, flush=True)

    audios: list[Any] = []
    sample_rate: int | None = None
    manual_chunk_seconds = (
        None if runtime.model_cfg.use_duration_predictor else float(args.chunk_max_seconds)
    )
    if manual_chunk_seconds is not None:
        print(
            "[duration] checkpoint has no duration predictor; "
            f"using manual chunk seconds={manual_chunk_seconds:.3f}",
            flush=True,
        )
    for i, chunk in enumerate(synthesis_chunks, start=1):
        seed = _chunk_seed(args.seed, i, str(args.seed_mode))
        print(
            f"[synthesize] chunk {i}/{len(chunks)} seed={'random' if seed is None else seed}",
            flush=True,
        )
        result = runtime.synthesize(
            SamplingRequest(
                text=chunk,
                caption=None if args.caption is None else str(args.caption),
                ref_wav=ref_wav,
                ref_latent=ref_latent,
                ref_embed=args.ref_embed,
                no_ref=bool(args.no_ref),
                ref_normalize_db=args.ref_normalize_db,
                ref_ensure_max=bool(args.ref_ensure_max),
                num_candidates=1,
                decode_mode=str(args.decode_mode),
                seconds=manual_chunk_seconds,
                duration_scale=float(args.duration_scale),
                min_seconds=0.5,
                max_seconds=float(args.chunk_max_seconds),
                max_ref_seconds=float(args.max_ref_seconds)
                if args.max_ref_seconds is not None
                else None,
                max_text_len=None if args.max_text_len is None else int(args.max_text_len),
                max_caption_len=None if args.max_caption_len is None else int(args.max_caption_len),
                num_steps=int(args.num_steps),
                cfg_scale_text=cfg_scale_text,
                cfg_scale_caption=cfg_scale_caption,
                cfg_scale_speaker=cfg_scale_speaker,
                cfg_guidance_mode=str(args.cfg_guidance_mode),
                cfg_scale=None,
                cfg_min_t=float(args.cfg_min_t),
                cfg_max_t=float(args.cfg_max_t),
                truncation_factor=None
                if args.truncation_factor is None
                else float(args.truncation_factor),
                rescale_k=None if args.rescale_k is None else float(args.rescale_k),
                rescale_sigma=None if args.rescale_sigma is None else float(args.rescale_sigma),
                context_kv_cache=bool(args.context_kv_cache),
                speaker_kv_scale=None
                if args.speaker_kv_scale is None
                else float(args.speaker_kv_scale),
                speaker_kv_min_t=None
                if args.speaker_kv_scale is None
                else float(args.speaker_kv_min_t),
                speaker_kv_max_layers=None
                if args.speaker_kv_max_layers is None
                else int(args.speaker_kv_max_layers),
                speaker_uncond_mode=str(args.speaker_uncond_mode),
                seed=seed,
                t_schedule_mode=str(args.t_schedule_mode),
                sway_coeff=float(args.sway_coeff),
                trim_tail=bool(args.trim_tail),
                tail_window_size=int(args.tail_window_size),
                tail_std_threshold=float(args.tail_std_threshold),
                tail_mean_threshold=float(args.tail_mean_threshold),
                lora_adapter=None if args.lora_adapter is None else str(args.lora_adapter),
            ),
            log_fn=print if args.show_timings else None,
        )
        print(f"[seed] chunk {i:03d} used_seed={result.used_seed}", flush=True)
        audios.append(result.audio)
        sample_rate = int(result.sample_rate)

        if args.save_chunks:
            chunk_dir.mkdir(parents=True, exist_ok=True)
            chunk_path = chunk_dir / f"chunk_{i:03d}.wav"
            save_wav(chunk_path, result.audio, int(result.sample_rate))
            print(f"[chunk] saved: {chunk_path}", flush=True)

    if sample_rate is None:
        raise RuntimeError("No chunks were synthesized.")

    final_audio = merge_audios(
        audios,
        sample_rate=int(sample_rate),
        pause_ms=float(args.pause_ms),
        edge_fade_ms=float(args.edge_fade_ms),
        normalize_chunks_db=args.normalize_chunks_db,
    )
    saved = save_wav(output_path, final_audio, int(sample_rate))
    duration = float(final_audio.shape[-1]) / float(sample_rate)
    print(f"[done] saved: {saved}", flush=True)
    print(f"[done] duration: {duration:.2f}s", flush=True)


if __name__ == "__main__":
    main()
