"""mazinger thumbnails — extract key-frame thumbnails."""

from __future__ import annotations

import argparse

from mazinger.cli._groups import add_common, add_llm, make_openai_client


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("thumbnails", help="Extract key-frame thumbnails.")
    p.add_argument("--video", required=True, help="Path to video file.")
    p.add_argument("--srt", required=True, help="Path to SRT file.")
    p.add_argument("--output-dir", required=True, help="Output directory for thumbnails.")
    p.add_argument("--meta", default=None, help="Path to save metadata JSON.")
    add_llm(p)
    add_common(p)


def handler(args: argparse.Namespace) -> None:
    from mazinger.thumbnails import select_timestamps, extract_frames
    from mazinger.utils import save_json

    client = make_openai_client(args)
    with open(args.srt, encoding="utf-8") as fh:
        srt_text = fh.read()

    timestamps = select_timestamps(srt_text, client, llm_model=args.llm_model)
    results = extract_frames(args.video, timestamps, args.output_dir)

    meta_path = args.meta or f"{args.output_dir}/meta.json"
    save_json(results, meta_path)
    print(f"Extracted {len(results)} thumbnails -> {args.output_dir}")
