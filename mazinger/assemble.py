"""Time-align TTS segments and assemble the final dubbed audio track."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess

import numpy as np
import soundfile as sf
from tqdm.auto import tqdm

from mazinger.utils import get_audio_duration

log = logging.getLogger(__name__)

TARGET_SR = 24_000


def _load_and_resample(wav_path: str, target_sr: int) -> np.ndarray:
    """Load a WAV and convert to mono at *target_sr* using ffmpeg."""
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", wav_path,
            "-ar", str(target_sr), "-ac", "1", "-f", "f32le", "-",
        ],
        capture_output=True,
        check=True,
    )
    return np.frombuffer(result.stdout, dtype=np.float32)


def _tempo_stretch(
    wav_path: str,
    factor: float,
    out_path: str,
    sr: int,
) -> np.ndarray:
    """Change playback speed by *factor* using the ffmpeg ``atempo`` filter.

    ``factor > 1`` speeds up, ``factor < 1`` slows down.
    """
    filters: list[str] = []
    remaining = factor
    while remaining > 100.0:
        filters.append("atempo=100.0")
        remaining /= 100.0
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    filters.append(f"atempo={remaining:.6f}")

    subprocess.run(
        [
            "ffmpeg", "-y", "-i", wav_path,
            "-filter:a", ",".join(filters),
            "-ar", str(sr), "-ac", "1", out_path,
        ],
        capture_output=True,
        check=True,
    )
    data, _ = sf.read(out_path, dtype="float32")
    return data


def assemble_timeline(
    segment_info: list[dict],
    original_duration: float,
    output_path: str,
    *,
    sample_rate: int = TARGET_SR,
    speed_threshold: float = 0.05,
    min_speed_ratio: float = 0.5,
    tempo_mode: str = "auto",
    fixed_tempo: float | None = None,
    max_tempo: float = 1.3,
) -> str:
    """Assemble per-segment TTS WAVs into a single time-aligned audio file.

    Each segment is placed at its SRT start time on a silence-filled
    timeline matching *original_duration*.

    Parameters:
        segment_info:      List of dicts from :func:`mazinger.tts.synthesize_segments`.
        original_duration: Duration of the original audio in seconds.
        output_path:       Where to write the final WAV.
        sample_rate:       Target sample rate.
        speed_threshold:   Fractional tolerance before tempo-stretching is applied.
        min_speed_ratio:   Lowest allowed slowdown factor (default 0.5 = max 2× slower).
                           Prevents extreme stretching that sounds unnatural.
        tempo_mode:        ``auto`` — speed up only overflowing segments (default);
                           ``off`` — no tempo adjustment;
                           ``dynamic`` — per-segment speed matching (up and down);
                           ``fixed`` — apply *fixed_tempo* to every segment.
        fixed_tempo:       Tempo rate applied to all segments when
                           ``tempo_mode="fixed"`` (e.g. 1.1).
        max_tempo:         Upper speed limit for dynamic/auto mode (default 1.3).

    Returns:
        The *output_path*.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    total_samples = int(original_duration * sample_rate)
    timeline = np.zeros(total_samples, dtype=np.float32)

    stats = {"sped_up": 0, "slowed_down": 0, "ok": 0, "skipped": 0, "truncated": 0}
    overflow_total = 0.0

    for seg in tqdm(segment_info, desc="Aligning"):
        if seg["wav_path"] is None:
            stats["skipped"] += 1
            continue

        target_dur = seg["target_dur"]
        target_samps = int(target_dur * sample_rate)
        start_samp = int(seg["start"] * sample_rate)

        raw_audio = _load_and_resample(seg["wav_path"], sample_rate)
        actual_dur = len(raw_audio) / sample_rate

        if actual_dur <= 0:
            stats["skipped"] += 1
            continue

        speed_ratio = actual_dur / target_dur

        if tempo_mode == "fixed" and fixed_tempo is not None:
            stretched_path = seg["wav_path"].replace(".wav", "_stretched.wav")
            audio = _tempo_stretch(seg["wav_path"], fixed_tempo, stretched_path, sample_rate)
            stats["sped_up"] += 1
        elif tempo_mode in ("dynamic", "auto"):
            if speed_ratio > 1.0 + speed_threshold:
                effective_ratio = min(speed_ratio, max_tempo)
                stretched_path = seg["wav_path"].replace(".wav", "_stretched.wav")
                audio = _tempo_stretch(seg["wav_path"], effective_ratio, stretched_path, sample_rate)
                stats["sped_up"] += 1
                if effective_ratio < speed_ratio:
                    overflow_secs = actual_dur / effective_ratio - target_dur
                    overflow_total += max(0, overflow_secs)
                    log.warning(
                        "Seg %s: TTS=%.2fs > target=%.2fs, capped speed-up at %.2fx "
                        "(needed %.2fx) — %.2fs will be truncated",
                        seg["idx"], actual_dur, target_dur,
                        effective_ratio, speed_ratio, max(0, overflow_secs),
                    )
            elif tempo_mode == "dynamic" and speed_ratio < 1.0 - speed_threshold:
                # "auto" mode only speeds up; never slows down
                effective_ratio = max(speed_ratio, min_speed_ratio)
                slowed_path = seg["wav_path"].replace(".wav", "_slowed.wav")
                audio = _tempo_stretch(seg["wav_path"], effective_ratio, slowed_path, sample_rate)
                stats["slowed_down"] += 1
            else:
                audio = raw_audio
                stats["ok"] += 1
        else:
            # tempo_mode == "off"
            if speed_ratio > 1.0 + speed_threshold:
                overflow_secs = actual_dur - target_dur
                overflow_total += overflow_secs
                log.warning(
                    "Seg %s: TTS=%.2fs > target=%.2fs (overflow %.2fs, will be truncated). "
                    "Consider using --dynamic-tempo or tempo_mode='auto'.",
                    seg["idx"], actual_dur, target_dur, overflow_secs,
                )
                stats["truncated"] += 1
            else:
                stats["ok"] += 1
            audio = raw_audio

        if len(audio) > target_samps:
            audio = audio[:target_samps]
        elif len(audio) < target_samps:
            audio = np.pad(audio, (0, target_samps - len(audio)))

        end_samp = min(start_samp + len(audio), total_samples)
        actual_len = end_samp - start_samp
        timeline[start_samp:end_samp] = audio[:actual_len]

    sf.write(output_path, timeline, sample_rate)

    if overflow_total > 0.1:
        log.warning(
            "Total overflow: %.2fs of TTS audio was truncated to fit the timeline. "
            "This may cause speech to be cut off. Consider reducing translation "
            "word count (--duration-budget) or increasing --max-tempo.",
            overflow_total,
        )

    log.info(
        "Timeline assembled: %.2fs (sped_up=%d, slowed_down=%d, ok=%d, skipped=%d, truncated=%d)",
        total_samples / sample_rate,
        stats["sped_up"], stats["slowed_down"], stats["ok"], stats["skipped"],
        stats["truncated"],
    )
    return output_path


def _measure_loudness(path: str) -> float:
    """Return integrated loudness (LUFS) of an audio file via ffmpeg."""
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", path,
         "-af", "loudnorm=print_format=json", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    import json as _json, re as _re
    m = _re.search(r'\{[^}]+"input_i"[^}]+\}', result.stderr, _re.DOTALL)
    if m:
        data = _json.loads(m.group())
        return float(data["input_i"])
    return -24.0


def _extract_background(audio_path: str, out_path: str, sr: int = TARGET_SR) -> str:
    """Extract non-vocal background from *audio_path*.

    Uses demucs (htdemucs model) for high-quality source separation.
    Falls back to spectral masking via librosa when demucs is unavailable.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    try:
        import torch, torchaudio
        from demucs.pretrained import get_model
        from demucs.apply import apply_model

        log.info("Extracting background with demucs")
        model = get_model("htdemucs")
        model.eval()
        wav, wav_sr = torchaudio.load(audio_path)
        if wav_sr != model.samplerate:
            wav = torchaudio.functional.resample(wav, wav_sr, model.samplerate)
        with torch.no_grad():
            sources = apply_model(model, wav.unsqueeze(0))
        # sources: (1, num_stems, channels, samples)
        vocals_idx = model.sources.index("vocals")
        bg = sources[0]
        bg[vocals_idx] = 0
        bg_np = bg.sum(dim=0).mean(dim=0).cpu().numpy()
        import librosa
        if model.samplerate != sr:
            bg_np = librosa.resample(bg_np, orig_sr=model.samplerate, target_sr=sr)
        sf.write(out_path, bg_np, sr)
    except Exception as exc:
        log.info("Demucs unavailable (%s), using spectral masking fallback", exc)
        import librosa
        y, _ = librosa.load(audio_path, sr=sr, mono=True)
        S = librosa.stft(y)
        H, P = librosa.decompose.hpss(np.abs(S), kernel_size=31, margin=4.0)
        mask = P / (H + P + 1e-10)
        bg = librosa.istft(S * mask, length=len(y))
        sf.write(out_path, bg, sr)
    return out_path


def post_process(
    dubbed_path: str,
    original_audio: str,
    output_path: str,
    *,
    loudness_match: bool = True,
    mix_background: bool = True,
    background_volume: float = 0.15,
) -> str:
    """Apply loudness normalisation and background audio mixing.

    Parameters:
        dubbed_path:       Path to the assembled TTS audio.
        original_audio:    Path to the original source audio.
        output_path:       Where to write the processed result.
        loudness_match:    Match dubbed loudness to the original.
        mix_background:    Extract and mix background from original.
        background_volume: Gain multiplier for the background layer (0.0–1.0).
    """
    if not loudness_match and not mix_background:
        if dubbed_path != output_path:
            shutil.copy2(dubbed_path, output_path)
        return output_path

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    work = dubbed_path

    # -- loudness matching ------------------------------------------------
    if loudness_match:
        target_lufs = _measure_loudness(original_audio)
        target_lufs = max(target_lufs, -30.0)  # safety floor
        norm_path = output_path + ".norm.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", work,
             "-af", f"loudnorm=I={target_lufs:.1f}:TP=-1.5:LRA=11",
             "-ar", str(TARGET_SR), "-ac", "1", norm_path],
            capture_output=True, check=True,
        )
        work = norm_path
        log.info("Loudness matched to %.1f LUFS", target_lufs)

    # -- background mixing ------------------------------------------------
    if mix_background:
        bg_path = os.path.join(os.path.dirname(output_path), "background.wav")
        _extract_background(original_audio, bg_path, sr=TARGET_SR)
        log.info("Background audio saved: %s", bg_path)

        dur_dub = get_audio_duration(work)
        mix_path = output_path + ".mix.wav"
        filt = (
            f"[1:a]atrim=0:{dur_dub:.3f},asetpts=PTS-STARTPTS,"
            f"volume={background_volume:.2f}[bg];"
            f"[0:a][bg]amix=inputs=2:duration=first:weights=1 {background_volume:.2f}[out]"
        )
        subprocess.run(
            ["ffmpeg", "-y", "-i", work, "-i", bg_path,
             "-filter_complex", filt, "-map", "[out]",
             "-ar", str(TARGET_SR), "-ac", "1", mix_path],
            capture_output=True, check=True,
        )
        work = mix_path
        log.info("Mixed background at volume %.0f%%", background_volume * 100)

    # -- move final result into place ------------------------------------
    if work != output_path:
        shutil.move(work, output_path)

    # cleanup temp files (background.wav is kept for inspection)
    for suffix in (".norm.wav", ".mix.wav"):
        tmp = output_path + suffix
        if os.path.exists(tmp) and tmp != output_path:
            os.remove(tmp)

    return output_path


def mux_video(video_path: str, audio_path: str, output_path: str) -> str | None:
    """Replace the audio track of *video_path* with *audio_path*.

    Uses ffmpeg to copy the video stream and encode the new audio.
    Returns *output_path*, or ``None`` if ffmpeg is not installed.
    """
    if shutil.which("ffmpeg") is None:
        log.warning(
            "ffmpeg not found — cannot produce dubbed video. "
            "Install ffmpeg (e.g. 'apt install ffmpeg' or 'brew install ffmpeg') "
            "and re-run with --output-type video."
        )
        return None
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    log.info("Muxed video saved: %s", output_path)
    return output_path
