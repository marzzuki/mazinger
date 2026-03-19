"""mazinger transcribe — audio to SRT."""

from __future__ import annotations

import argparse

from mazinger.cli._groups import add_common, add_openai


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("transcribe", help="Transcribe audio to SRT.")
    p.add_argument("audio", help="Path to audio file.")
    p.add_argument("-o", "--output", required=True, help="Output SRT path.")
    p.add_argument("--method", default="openai", choices=["openai", "faster-whisper", "whisperx"],
                   help="Transcription backend.")
    p.add_argument("--model", default=None,
                   help="Model name. Defaults to 'whisper-1' for OpenAI, 'large-v3' for local.")
    p.add_argument("--device", default="cuda", help="Device for local methods (cuda/cpu).")
    p.add_argument("--batch-size", type=int, default=16, help="Batch size for local methods.")
    p.add_argument("--compute-type", default="float16", help="Compute type: float16, int8, int8_float16.")
    p.add_argument("--language", default=None, help="Force language code (e.g., en, ar).")
    p.add_argument("--max-chars", type=int, default=120, help="Max chars per subtitle.")
    p.add_argument("--max-duration", type=float, default=10.0, help="Max seconds per subtitle.")
    p.add_argument("--no-resegment", action="store_true", help="Skip resegmentation.")
    add_openai(p)
    add_common(p)


def handler(args: argparse.Namespace) -> None:
    from mazinger.transcribe import transcribe

    transcribe(
        args.audio, args.output,
        method=args.method,
        model=args.model,
        device=args.device,
        batch_size=args.batch_size,
        compute_type=args.compute_type,
        language=args.language,
        max_chars=args.max_chars,
        max_duration=args.max_duration,
        skip_resegment=args.no_resegment,
        openai_api_key=args.openai_api_key,
        openai_base_url=args.openai_base_url,
    )
    print(f"SRT saved: {args.output}")
