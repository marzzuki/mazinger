"""mazinger describe — generate video content description."""

from __future__ import annotations

import argparse

from mazinger.cli._groups import add_common, add_llm, make_openai_client


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("describe", help="Generate video content description.")
    p.add_argument("--srt", required=True, help="Path to SRT file.")
    p.add_argument("--thumbnails-meta", required=True, help="Path to thumbnails meta.json.")
    p.add_argument("-o", "--output", required=True, help="Output JSON path.")
    add_llm(p)
    add_common(p)


def handler(args: argparse.Namespace) -> None:
    from mazinger.describe import describe_content
    from mazinger.utils import load_json, save_json

    client = make_openai_client(args)
    with open(args.srt, encoding="utf-8") as fh:
        srt_text = fh.read()
    thumb_paths = load_json(args.thumbnails_meta)

    desc = describe_content(srt_text, thumb_paths, client, llm_model=args.llm_model)
    save_json(desc, args.output)
    print(f"Description saved: {args.output}")
