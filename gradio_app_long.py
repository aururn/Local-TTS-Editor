#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr
from huggingface_hub import hf_hub_download

from irodori_tts.inference_runtime import (
    RuntimeKey,
    SamplingRequest,
    clear_cached_runtime,
    default_runtime_device,
    get_cached_runtime,
    list_available_runtime_devices,
    list_available_runtime_precisions,
    resolve_cfg_scales,
    save_wav,
)
from irodori_tts.speaker_inversion import is_speaker_inversion_safetensors_path
from long_infer import _chunk_seed, _preview_text, _visible_len, chunk_text, merge_audios


CHECKPOINT_PRESETS = {
    "v3 Base": "Aratako/Irodori-TTS-500M-v3",
    "v2 VoiceDesign": "Aratako/Irodori-TTS-500M-v2-VoiceDesign",
    "Custom": "",
}

APP_CSS = """
.gradio-container {
    max-width: 1480px !important;
    color: #251f1a;
    background:
        linear-gradient(180deg, rgba(246, 242, 234, 0.96), rgba(241, 236, 226, 0.96)),
        repeating-linear-gradient(90deg, rgba(69, 120, 116, 0.05) 0 1px, transparent 1px 28px);
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
#app-hero {
    border: 1px solid #d9d0c1;
    border-radius: 8px;
    padding: 22px 24px;
    margin-bottom: 14px;
    background: #fffdf8;
    box-shadow: 0 14px 36px rgba(50, 43, 34, 0.08);
    display: flex;
    align-items: end;
    justify-content: space-between;
    gap: 20px;
}
#app-hero .kicker {
    margin: 0 0 4px;
    color: #55736f;
    font-size: 12px;
    font-weight: 760;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}
#app-hero h1 {
    margin: 0;
    color: #211d19;
    font-size: 34px;
    line-height: 1.05;
    letter-spacing: 0;
}
#app-hero .signal {
    color: #6f5f49;
    font-size: 13px;
    text-align: right;
}
.panel {
    border: 1px solid #d9d0c1;
    border-radius: 8px;
    background: rgba(255, 253, 248, 0.94);
    padding: 12px;
    box-shadow: 0 12px 28px rgba(50, 43, 34, 0.06);
}
.compact-panel {
    border: 1px solid #d9d0c1;
    border-radius: 8px;
    background: rgba(255, 253, 248, 0.86);
    padding: 10px 12px;
}
.gradio-container label,
.gradio-container .label-wrap span {
    color: #4d443b !important;
    font-size: 12px !important;
    font-weight: 700 !important;
}
.gradio-container button.primary {
    background: #2f6f6a !important;
    border: 1px solid #255b57 !important;
    color: #fffaf2 !important;
}
.gradio-container button.secondary {
    border-color: #c9bda8 !important;
}
textarea,
input,
.gradio-container .wrap {
    border-radius: 7px !important;
}
@media (max-width: 820px) {
    #app-hero {
        align-items: start;
        flex-direction: column;
    }
    #app-hero .signal {
        text-align: left;
    }
}
"""


def _default_checkpoint() -> str:
    candidates = sorted(
        [
            *Path(".").glob("**/checkpoint_*.pt"),
            *(
                path
                for path in Path(".").glob("**/checkpoint_*.safetensors")
                if not is_speaker_inversion_safetensors_path(path)
            ),
        ]
    )
    if not candidates:
        return "Aratako/Irodori-TTS-500M-v3"
    return str(candidates[-1])


def _on_checkpoint_preset_change(preset: str, current_checkpoint: str) -> gr.Textbox:
    value = CHECKPOINT_PRESETS.get(str(preset), "")
    if value == "":
        value = str(current_checkpoint or "").strip()
    return gr.Textbox(value=value)


def _precision_choices_for_device(device: str) -> list[str]:
    return list_available_runtime_precisions(device)


def _on_device_change(device: str) -> gr.Dropdown:
    choices = _precision_choices_for_device(device)
    return gr.Dropdown(choices=choices, value=choices[0])


def _on_t_schedule_mode_change(mode: str) -> object:
    return gr.update(interactive=str(mode).strip().lower() == "sway")


def _parse_optional_float(raw: str | None, label: str) -> float | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "" or text.lower() == "none":
        return None
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"{label} must be a float or blank.") from exc


def _parse_optional_int(raw: str | None, label: str) -> int | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "" or text.lower() == "none":
        return None
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(f"{label} must be an int or blank.") from exc


def _parse_optional_str(raw: str | None) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "" or text.lower() in {"none", "null", "off", "disable", "disabled", "base"}:
        return None
    return text


def _coerce_gradio_file_path(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, dict):
        for key in ("path", "name"):
            candidate = value.get(key)
            if candidate is not None and str(candidate).strip():
                return str(candidate)
        return None
    candidate = getattr(value, "name", None)
    if candidate is not None and str(candidate).strip():
        return str(candidate)
    text = str(value).strip()
    return text or None


def _read_text_file(path: str) -> str:
    file_path = Path(path)
    for encoding in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return file_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return file_path.read_text(encoding="utf-8", errors="replace")


def _resolve_long_text(text: str | None, uploaded_text_file: object) -> str:
    if text is not None and str(text).strip():
        return str(text)
    text_file = _coerce_gradio_file_path(uploaded_text_file)
    if text_file is None:
        raise ValueError("Text is required.")
    return _read_text_file(text_file)


def _chunk_limit(
    *,
    chunk_max_seconds: float,
    chars_per_second: float,
    max_chars_raw: str | None,
) -> int:
    if chunk_max_seconds <= 0:
        raise ValueError("chunk_max_seconds must be > 0.")
    if chunk_max_seconds > 30.0:
        raise ValueError("chunk_max_seconds must be <= 30.0 for released checkpoints.")
    if chars_per_second <= 0:
        raise ValueError("chars_per_second must be > 0.")
    max_chars = _parse_optional_int(max_chars_raw, "max_chars")
    if max_chars is not None:
        if max_chars <= 0:
            raise ValueError("max_chars must be > 0.")
        return int(max_chars)
    return max(1, int(float(chunk_max_seconds) * float(chars_per_second)))


def _make_chunk_table(chunks: list[str], chars_per_second: float) -> list[list[object]]:
    rows: list[list[object]] = []
    for i, chunk in enumerate(chunks, start=1):
        chars = _visible_len(chunk)
        rows.append([i, chars, round(chars / float(chars_per_second), 1), _preview_text(chunk)])
    return rows


def _prepare_chunks(
    text: str | None,
    uploaded_text_file: object,
    chunk_max_seconds: float,
    chars_per_second: float,
    max_chars_raw: str | None,
) -> tuple[list[str], int, list[list[object]], str]:
    long_text = _resolve_long_text(text, uploaded_text_file)
    max_chars = _chunk_limit(
        chunk_max_seconds=float(chunk_max_seconds),
        chars_per_second=float(chars_per_second),
        max_chars_raw=max_chars_raw,
    )
    chunks = chunk_text(long_text, max_chars=max_chars)
    if not chunks:
        raise ValueError("Text produced no chunks.")
    table = _make_chunk_table(chunks, float(chars_per_second))
    total_chars = sum(_visible_len(chunk) for chunk in chunks)
    summary = (
        f"chunks: {len(chunks)}\n"
        f"max_chars: {max_chars}\n"
        f"visible_chars: {total_chars}\n"
        f"chunk_max_seconds: {float(chunk_max_seconds):.2f}"
    )
    return chunks, max_chars, table, summary


def _preview_chunks(
    text: str | None,
    uploaded_text_file: object,
    chunk_max_seconds: float,
    chars_per_second: float,
    max_chars_raw: str | None,
) -> tuple[list[list[object]], str]:
    _chunks, _max_chars, table, summary = _prepare_chunks(
        text,
        uploaded_text_file,
        chunk_max_seconds,
        chars_per_second,
        max_chars_raw,
    )
    return table, summary


def _resolve_checkpoint_path(raw_checkpoint: str) -> str:
    checkpoint = str(raw_checkpoint).strip()
    if checkpoint == "":
        raise ValueError("checkpoint is required.")
    suffix = Path(checkpoint).suffix.lower()
    if suffix in {".pt", ".safetensors"}:
        path = Path(checkpoint).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return str(path)
    resolved = hf_hub_download(repo_id=checkpoint, filename="model.safetensors")
    print(f"[gradio-long] checkpoint: hf://{checkpoint} -> {resolved}", flush=True)
    return str(resolved)


def _build_runtime_key(
    checkpoint: str,
    model_device: str,
    model_precision: str,
    codec_device: str,
    codec_precision: str,
    codec_repo: str,
) -> RuntimeKey:
    checkpoint_path = _resolve_checkpoint_path(checkpoint)
    return RuntimeKey(
        checkpoint=checkpoint_path,
        model_device=str(model_device),
        codec_repo=str(codec_repo),
        model_precision=str(model_precision),
        codec_device=str(codec_device),
        codec_precision=str(codec_precision),
        compile_model=False,
        compile_dynamic=False,
    )


def _load_model(
    checkpoint: str,
    model_device: str,
    model_precision: str,
    codec_device: str,
    codec_precision: str,
    codec_repo: str,
) -> str:
    runtime_key = _build_runtime_key(
        checkpoint=checkpoint,
        model_device=model_device,
        model_precision=model_precision,
        codec_device=codec_device,
        codec_precision=codec_precision,
        codec_repo=codec_repo,
    )
    _, reloaded = get_cached_runtime(runtime_key)
    status = "loaded model into memory" if reloaded else "model already loaded"
    return (
        f"{status}\n"
        f"checkpoint: {runtime_key.checkpoint}\n"
        f"model_device: {runtime_key.model_device}\n"
        f"model_precision: {runtime_key.model_precision}\n"
        f"codec_device: {runtime_key.codec_device}\n"
        f"codec_precision: {runtime_key.codec_precision}"
    )


def _clear_runtime_cache() -> str:
    clear_cached_runtime()
    return "cleared loaded model from memory"


def _encode_reference_latent_once(
    *,
    runtime: Any,
    ref_wav: str,
    out_dir: Path,
    max_ref_seconds: float | None,
    ref_normalize_db: float | None,
    ref_ensure_max: bool,
) -> tuple[str, list[str]]:
    import torch

    from irodori_tts.inference_runtime import _load_audio

    messages: list[str] = []
    latent_path = out_dir / "reference.latent.pt"
    wav, sr = _load_audio(ref_wav)
    if max_ref_seconds is not None and max_ref_seconds > 0:
        max_ref_samples = max(1, int(float(max_ref_seconds) * float(sr)))
        if wav.shape[1] > max_ref_samples:
            messages.append(
                "reference trimmed: "
                f"{float(wav.shape[1]) / float(sr):.2f}s -> "
                f"{float(max_ref_samples) / float(sr):.2f}s"
            )
            wav = wav[:, :max_ref_samples]
    ref_latent = runtime.codec.encode_waveform(
        wav.unsqueeze(0),
        sample_rate=int(sr),
        normalize_db=ref_normalize_db,
        ensure_max=bool(ref_ensure_max),
    ).cpu()
    torch.save(ref_latent[0], latent_path)
    messages.append(f"reference latent: {latent_path}")
    return str(latent_path), messages


def _resolve_conditioning(
    *,
    runtime: Any,
    uploaded_audio: str | None,
    uploaded_ref_latent: object,
    uploaded_speaker_embedding: object,
    no_reference: bool,
    out_dir: Path,
    max_ref_seconds: float | None,
    ref_normalize_db_raw: str | None,
    ref_ensure_max: bool,
) -> tuple[str | None, str | None, str | None, bool, list[str]]:
    ref_wav = _coerce_gradio_file_path(uploaded_audio)
    ref_latent = _coerce_gradio_file_path(uploaded_ref_latent)
    ref_embed = _coerce_gradio_file_path(uploaded_speaker_embedding)
    provided = [value is not None for value in (ref_wav, ref_latent, ref_embed)]
    if no_reference and any(provided):
        raise ValueError("No Reference cannot be combined with reference inputs.")
    if sum(1 for value in provided if value) > 1:
        raise ValueError("Use only one of reference audio, reference latent, or speaker embedding.")
    if no_reference or not any(provided):
        return None, None, None, True, []
    if ref_wav is not None:
        ref_normalize_db = _parse_optional_float(ref_normalize_db_raw, "ref_normalize_db")
        cached_ref_latent, messages = _encode_reference_latent_once(
            runtime=runtime,
            ref_wav=ref_wav,
            out_dir=out_dir,
            max_ref_seconds=max_ref_seconds,
            ref_normalize_db=ref_normalize_db,
            ref_ensure_max=bool(ref_ensure_max),
        )
        return None, cached_ref_latent, None, False, messages
    if ref_latent is not None:
        return None, ref_latent, None, False, []
    return None, None, ref_embed, False, []


def _format_timings(stage_timings: list[tuple[str, float]], total_to_decode: float) -> list[str]:
    return [
        *[f"timing {name}: {sec * 1000.0:.1f} ms" for name, sec in stage_timings],
        f"timing total_to_decode: {total_to_decode:.3f} s",
    ]


def _run_long_generation(
    checkpoint: str,
    model_device: str,
    model_precision: str,
    codec_device: str,
    codec_precision: str,
    codec_repo: str,
    text: str | None,
    uploaded_text_file: object,
    uploaded_audio: str | None,
    uploaded_ref_latent: object,
    uploaded_speaker_embedding: object,
    no_reference: bool,
    caption: str | None,
    chunk_max_seconds: float,
    chars_per_second: float,
    max_chars_raw: str,
    pause_ms: float,
    edge_fade_ms: float,
    normalize_chunks_db_raw: str,
    num_steps: int,
    duration_scale: float,
    t_schedule_mode: str,
    sway_coeff: float,
    seed_raw: str,
    seed_mode: str,
    decode_mode: str,
    cfg_guidance_mode: str,
    cfg_scale_text: float,
    cfg_scale_caption: float,
    cfg_scale_speaker: float,
    cfg_scale_raw: str,
    cfg_min_t: float,
    cfg_max_t: float,
    context_kv_cache: bool,
    max_ref_seconds: float,
    ref_normalize_db_raw: str,
    ref_ensure_max: bool,
    trim_tail: bool,
    tail_window_size: int,
    tail_std_threshold: float,
    tail_mean_threshold: float,
    max_text_len_raw: str,
    max_caption_len_raw: str,
    truncation_factor_raw: str,
    rescale_k_raw: str,
    rescale_sigma_raw: str,
    speaker_kv_scale_raw: str,
    speaker_kv_min_t_raw: str,
    speaker_kv_max_layers_raw: str,
    speaker_uncond_mode: str,
    lora_adapter_raw: str,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
) -> tuple[str, str, list[list[object]], str]:
    chunks, max_chars, table, summary = _prepare_chunks(
        text,
        uploaded_text_file,
        chunk_max_seconds,
        chars_per_second,
        max_chars_raw,
    )
    runtime_key = _build_runtime_key(
        checkpoint=checkpoint,
        model_device=model_device,
        model_precision=model_precision,
        codec_device=codec_device,
        codec_precision=codec_precision,
        codec_repo=codec_repo,
    )
    runtime, reloaded = get_cached_runtime(runtime_key)

    out_dir = Path("gradio_long_outputs") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_wav, ref_latent, ref_embed, no_ref, ref_messages = _resolve_conditioning(
        runtime=runtime,
        uploaded_audio=uploaded_audio,
        uploaded_ref_latent=uploaded_ref_latent,
        uploaded_speaker_embedding=uploaded_speaker_embedding,
        no_reference=bool(no_reference),
        out_dir=out_dir,
        max_ref_seconds=float(max_ref_seconds) if max_ref_seconds is not None else None,
        ref_normalize_db_raw=ref_normalize_db_raw,
        ref_ensure_max=bool(ref_ensure_max),
    )

    has_caption = bool(
        runtime.model_cfg.use_caption_condition and caption is not None and str(caption).strip()
    )
    cfg_scale = _parse_optional_float(cfg_scale_raw, "cfg_scale")
    cfg_scale_text_value, cfg_scale_caption_value, cfg_scale_speaker_value, scale_messages = (
        resolve_cfg_scales(
            cfg_guidance_mode=str(cfg_guidance_mode),
            cfg_scale_text=float(cfg_scale_text),
            cfg_scale_caption=float(cfg_scale_caption),
            cfg_scale_speaker=float(cfg_scale_speaker),
            cfg_scale=cfg_scale,
            use_caption_condition=has_caption,
            use_speaker_condition=bool(runtime.model_cfg.use_speaker_condition),
        )
    )

    seed = _parse_optional_int(seed_raw, "seed")
    normalize_chunks_db = _parse_optional_float(normalize_chunks_db_raw, "normalize_chunks_db")
    max_text_len = _parse_optional_int(max_text_len_raw, "max_text_len")
    max_caption_len = _parse_optional_int(max_caption_len_raw, "max_caption_len")
    truncation_factor = _parse_optional_float(truncation_factor_raw, "truncation_factor")
    rescale_k = _parse_optional_float(rescale_k_raw, "rescale_k")
    rescale_sigma = _parse_optional_float(rescale_sigma_raw, "rescale_sigma")
    speaker_kv_scale = _parse_optional_float(speaker_kv_scale_raw, "speaker_kv_scale")
    speaker_kv_min_t = _parse_optional_float(speaker_kv_min_t_raw, "speaker_kv_min_t")
    speaker_kv_max_layers = _parse_optional_int(speaker_kv_max_layers_raw, "speaker_kv_max_layers")
    lora_adapter = _parse_optional_str(lora_adapter_raw)

    logs: list[str] = [
        "runtime: reloaded" if reloaded else "runtime: reused",
        f"checkpoint: {runtime_key.checkpoint}",
        f"chunks: {len(chunks)}",
        f"max_chars: {max_chars}",
        *scale_messages,
        *ref_messages,
    ]
    audios: list[Any] = []
    sample_rate: int | None = None
    manual_chunk_seconds = (
        None if runtime.model_cfg.use_duration_predictor else float(chunk_max_seconds)
    )
    if manual_chunk_seconds is not None:
        logs.append(
            "duration: checkpoint has no duration predictor; "
            f"using manual chunk seconds={manual_chunk_seconds:.3f}"
        )

    for i, chunk in enumerate(chunks, start=1):
        progress((i - 1) / len(chunks), desc=f"chunk {i}/{len(chunks)}")
        chunk_seed = _chunk_seed(seed, i, str(seed_mode))
        result = runtime.synthesize(
            SamplingRequest(
                text=chunk,
                caption=None if caption is None else str(caption),
                ref_wav=ref_wav,
                ref_latent=ref_latent,
                ref_embed=ref_embed,
                no_ref=bool(no_ref),
                ref_normalize_db=_parse_optional_float(ref_normalize_db_raw, "ref_normalize_db"),
                ref_ensure_max=bool(ref_ensure_max),
                num_candidates=1,
                decode_mode=str(decode_mode),
                seconds=manual_chunk_seconds,
                duration_scale=float(duration_scale),
                min_seconds=0.5,
                max_seconds=float(chunk_max_seconds),
                max_ref_seconds=float(max_ref_seconds) if max_ref_seconds is not None else None,
                max_text_len=max_text_len,
                max_caption_len=max_caption_len,
                num_steps=int(num_steps),
                cfg_scale_text=cfg_scale_text_value,
                cfg_scale_caption=cfg_scale_caption_value,
                cfg_scale_speaker=cfg_scale_speaker_value,
                cfg_guidance_mode=str(cfg_guidance_mode),
                cfg_scale=None,
                cfg_min_t=float(cfg_min_t),
                cfg_max_t=float(cfg_max_t),
                truncation_factor=truncation_factor,
                rescale_k=rescale_k,
                rescale_sigma=rescale_sigma,
                context_kv_cache=bool(context_kv_cache),
                speaker_kv_scale=speaker_kv_scale,
                speaker_kv_min_t=speaker_kv_min_t if speaker_kv_scale is not None else None,
                speaker_kv_max_layers=speaker_kv_max_layers,
                speaker_uncond_mode=str(speaker_uncond_mode),
                seed=chunk_seed,
                t_schedule_mode=str(t_schedule_mode),
                sway_coeff=float(sway_coeff),
                trim_tail=bool(trim_tail),
                tail_window_size=int(tail_window_size),
                tail_std_threshold=float(tail_std_threshold),
                tail_mean_threshold=float(tail_mean_threshold),
                lora_adapter=lora_adapter,
            ),
            log_fn=lambda msg: print(msg, flush=True),
        )
        audios.append(result.audio)
        sample_rate = int(result.sample_rate)
        chunk_path = save_wav(out_dir / f"chunk_{i:03d}.wav", result.audio, result.sample_rate)
        logs.extend(
            [
                f"chunk {i:03d}: seed={result.used_seed} chars={_visible_len(chunk)}",
                f"chunk {i:03d}: saved={chunk_path}",
                *_format_timings(result.stage_timings, result.total_to_decode),
                *result.messages,
            ]
        )

    if sample_rate is None:
        raise RuntimeError("No chunks were generated.")

    final_audio = merge_audios(
        audios,
        sample_rate=int(sample_rate),
        pause_ms=float(pause_ms),
        edge_fade_ms=float(edge_fade_ms),
        normalize_chunks_db=normalize_chunks_db,
    )
    final_path = save_wav(out_dir / "long.wav", final_audio, int(sample_rate))
    duration = float(final_audio.shape[-1]) / float(sample_rate)
    progress(1.0, desc="done")
    logs.extend(
        [
            f"final: {final_path}",
            f"duration: {duration:.2f}s",
            f"output_dir: {out_dir}",
        ]
    )
    return str(final_path), str(final_path), table, summary + "\n" + "\n".join(logs)


def build_ui() -> gr.Blocks:
    default_checkpoint = _default_checkpoint()
    default_model_device = default_runtime_device()
    default_codec_device = default_runtime_device()
    device_choices = list_available_runtime_devices()
    model_precision_choices = _precision_choices_for_device(default_model_device)
    codec_precision_choices = _precision_choices_for_device(default_codec_device)

    with gr.Blocks(title="Irodori-TTS Long Form") as demo:
        gr.HTML(
            """
            <section id="app-hero">
              <div>
                <p class="kicker">Irodori longform</p>
                <h1>Sequence Studio</h1>
              </div>
              <div class="signal">30s chunks / cached runtime / single WAV</div>
            </section>
            """
        )

        with gr.Row(elem_classes=["panel"]):
            checkpoint_preset = gr.Dropdown(
                label="Preset",
                choices=list(CHECKPOINT_PRESETS.keys()),
                value="v3 Base",
                scale=1,
            )
            checkpoint = gr.Textbox(
                label="Checkpoint",
                value=default_checkpoint,
                scale=3,
            )
            model_device = gr.Dropdown(
                label="Model Device",
                choices=device_choices,
                value=default_model_device,
                scale=1,
            )
            model_precision = gr.Dropdown(
                label="Model Precision",
                choices=model_precision_choices,
                value=model_precision_choices[0],
                scale=1,
            )
            codec_device = gr.Dropdown(
                label="Codec Device",
                choices=device_choices,
                value=default_codec_device,
                scale=1,
            )
            codec_precision = gr.Dropdown(
                label="Codec Precision",
                choices=codec_precision_choices,
                value=codec_precision_choices[0],
                scale=1,
            )

        with gr.Row(elem_classes=["compact-panel"]):
            codec_repo = gr.Textbox(
                label="Codec Repo",
                value="Aratako/Semantic-DACVAE-Japanese-32dim",
                scale=4,
            )
            load_model_btn = gr.Button("Load Model", scale=1)
            clear_cache_btn = gr.Button("Unload Model", scale=1)
        model_status = gr.Textbox(label="Model Status", lines=4, interactive=False)

        with gr.Row():
            with gr.Column(scale=3):
                text = gr.Textbox(label="Text", lines=12)
                uploaded_text_file = gr.File(
                    label="Text File",
                    type="filepath",
                    file_count="single",
                    file_types=[".txt", ".md"],
                )
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("Reference Audio"):
                        uploaded_audio = gr.Audio(
                            label="Reference Audio",
                            type="filepath",
                        )
                    with gr.Tab("Reference Latent"):
                        uploaded_ref_latent = gr.File(
                            label="Reference Latent",
                            type="filepath",
                            file_count="single",
                            file_types=[".pt"],
                        )
                    with gr.Tab("Speaker Embedding"):
                        uploaded_speaker_embedding = gr.File(
                            label="Speaker Embedding",
                            type="filepath",
                            file_count="single",
                            file_types=[".safetensors"],
                        )
                no_reference = gr.Checkbox(label="No Reference", value=False)
                caption = gr.Textbox(label="Caption", lines=3)

        with gr.Accordion("Chunking", open=True, elem_classes=["compact-panel"]):
            with gr.Row():
                chunk_max_seconds = gr.Slider(
                    label="Chunk Max Seconds",
                    minimum=2.0,
                    maximum=30.0,
                    value=25.0,
                    step=0.5,
                )
                chars_per_second = gr.Slider(
                    label="Chars Per Second",
                    minimum=2.0,
                    maximum=12.0,
                    value=6.0,
                    step=0.25,
                )
                max_chars_raw = gr.Textbox(label="Max Chars", value="")
            with gr.Row():
                pause_ms = gr.Slider(label="Pause ms", minimum=0, maximum=1200, value=220, step=10)
                edge_fade_ms = gr.Slider(
                    label="Edge Fade ms",
                    minimum=0,
                    maximum=50,
                    value=5,
                    step=1,
                )
                normalize_chunks_db_raw = gr.Textbox(label="Normalize Chunks dB", value="")

        preview_btn = gr.Button("Preview Split")
        chunk_table = gr.Dataframe(
            headers=["#", "chars", "est_sec", "text"],
            datatype=["number", "number", "number", "str"],
            interactive=False,
        )
        preview_log = gr.Textbox(label="Run Log", lines=12, interactive=False)

        with gr.Accordion("Sampling", open=True, elem_classes=["compact-panel"]):
            with gr.Row():
                num_steps = gr.Slider(label="Num Steps", minimum=1, maximum=120, value=40, step=1)
                duration_scale = gr.Slider(
                    label="Duration Scale",
                    minimum=0.5,
                    maximum=1.5,
                    value=1.0,
                    step=0.01,
                )
                seed_raw = gr.Textbox(label="Seed", value="")
                seed_mode = gr.Dropdown(
                    label="Seed Mode",
                    choices=["offset", "same", "random"],
                    value="offset",
                )
                decode_mode = gr.Dropdown(
                    label="Decode Mode",
                    choices=["sequential", "batch"],
                    value="sequential",
                )
            with gr.Row():
                t_schedule_mode = gr.Dropdown(
                    label="Time Schedule",
                    choices=["linear", "sway"],
                    value="linear",
                )
                sway_coeff = gr.Slider(
                    label="Sway Coeff",
                    minimum=-1.0,
                    maximum=1.5,
                    value=-1.0,
                    step=0.1,
                    interactive=False,
                )
            with gr.Row():
                cfg_guidance_mode = gr.Dropdown(
                    label="CFG Guidance Mode",
                    choices=["independent", "joint", "alternating"],
                    value="independent",
                )
                cfg_scale_text = gr.Slider(
                    label="CFG Text",
                    minimum=0.0,
                    maximum=10.0,
                    value=3.0,
                    step=0.1,
                )
                cfg_scale_caption = gr.Slider(
                    label="CFG Caption",
                    minimum=0.0,
                    maximum=10.0,
                    value=3.0,
                    step=0.1,
                )
                cfg_scale_speaker = gr.Slider(
                    label="CFG Speaker",
                    minimum=0.0,
                    maximum=10.0,
                    value=5.0,
                    step=0.1,
                )

        with gr.Accordion("Advanced", open=False, elem_classes=["compact-panel"]):
            with gr.Row():
                cfg_scale_raw = gr.Textbox(label="CFG Override", value="")
                cfg_min_t = gr.Number(label="CFG Min t", value=0.5)
                cfg_max_t = gr.Number(label="CFG Max t", value=1.0)
                context_kv_cache = gr.Checkbox(label="Context KV Cache", value=True)
            with gr.Row():
                max_ref_seconds = gr.Number(label="Max Ref Seconds", value=30.0)
                ref_normalize_db_raw = gr.Textbox(label="Ref Normalize dB", value="-16.0")
                ref_ensure_max = gr.Checkbox(label="Ref Ensure Max", value=True)
            with gr.Row():
                trim_tail = gr.Checkbox(label="Trim Tail", value=True)
                tail_window_size = gr.Number(label="Tail Window", value=20)
                tail_std_threshold = gr.Number(label="Tail Std", value=0.05)
                tail_mean_threshold = gr.Number(label="Tail Mean", value=0.1)
            with gr.Row():
                max_text_len_raw = gr.Textbox(label="Max Text Len", value="")
                max_caption_len_raw = gr.Textbox(label="Max Caption Len", value="")
                truncation_factor_raw = gr.Textbox(label="Truncation Factor", value="")
            with gr.Row():
                rescale_k_raw = gr.Textbox(label="Rescale k", value="")
                rescale_sigma_raw = gr.Textbox(label="Rescale sigma", value="")
                speaker_uncond_mode = gr.Dropdown(
                    label="Speaker Uncond",
                    choices=["mask", "noise"],
                    value="mask",
                )
            with gr.Row():
                speaker_kv_scale_raw = gr.Textbox(label="Speaker KV Scale", value="")
                speaker_kv_min_t_raw = gr.Textbox(label="Speaker KV Min t", value="0.9")
                speaker_kv_max_layers_raw = gr.Textbox(label="Speaker KV Max Layers", value="")
            lora_adapter_raw = gr.Textbox(label="LoRA Adapter Directory", value="")

        generate_btn = gr.Button("Generate Long Audio", variant="primary")
        with gr.Row():
            final_audio = gr.Audio(label="Final Audio", type="filepath", interactive=False)
            final_file = gr.File(label="Final WAV", interactive=False)

        preview_inputs = [
            text,
            uploaded_text_file,
            chunk_max_seconds,
            chars_per_second,
            max_chars_raw,
        ]
        preview_btn.click(
            _preview_chunks,
            inputs=preview_inputs,
            outputs=[chunk_table, preview_log],
        )

        generate_btn.click(
            _run_long_generation,
            inputs=[
                checkpoint,
                model_device,
                model_precision,
                codec_device,
                codec_precision,
                codec_repo,
                text,
                uploaded_text_file,
                uploaded_audio,
                uploaded_ref_latent,
                uploaded_speaker_embedding,
                no_reference,
                caption,
                chunk_max_seconds,
                chars_per_second,
                max_chars_raw,
                pause_ms,
                edge_fade_ms,
                normalize_chunks_db_raw,
                num_steps,
                duration_scale,
                t_schedule_mode,
                sway_coeff,
                seed_raw,
                seed_mode,
                decode_mode,
                cfg_guidance_mode,
                cfg_scale_text,
                cfg_scale_caption,
                cfg_scale_speaker,
                cfg_scale_raw,
                cfg_min_t,
                cfg_max_t,
                context_kv_cache,
                max_ref_seconds,
                ref_normalize_db_raw,
                ref_ensure_max,
                trim_tail,
                tail_window_size,
                tail_std_threshold,
                tail_mean_threshold,
                max_text_len_raw,
                max_caption_len_raw,
                truncation_factor_raw,
                rescale_k_raw,
                rescale_sigma_raw,
                speaker_kv_scale_raw,
                speaker_kv_min_t_raw,
                speaker_kv_max_layers_raw,
                speaker_uncond_mode,
                lora_adapter_raw,
            ],
            outputs=[final_audio, final_file, chunk_table, preview_log],
        )
        load_model_btn.click(
            _load_model,
            inputs=[
                checkpoint,
                model_device,
                model_precision,
                codec_device,
                codec_precision,
                codec_repo,
            ],
            outputs=[model_status],
        )
        clear_cache_btn.click(_clear_runtime_cache, outputs=[model_status])
        model_device.change(_on_device_change, inputs=[model_device], outputs=[model_precision])
        codec_device.change(_on_device_change, inputs=[codec_device], outputs=[codec_precision])
        checkpoint_preset.change(
            _on_checkpoint_preset_change,
            inputs=[checkpoint_preset, checkpoint],
            outputs=[checkpoint],
        )
        t_schedule_mode.change(
            _on_t_schedule_mode_change,
            inputs=[t_schedule_mode],
            outputs=[sway_coeff],
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Gradio app for long-form Irodori-TTS.")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7862)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    demo = build_ui()
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=bool(args.share),
        debug=bool(args.debug),
        css=APP_CSS,
    )


if __name__ == "__main__":
    main()
