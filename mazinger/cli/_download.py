"""mazinger download — fetch video / ingest local file and extract audio."""

from __future__ import annotations

import argparse

from mazinger.cli._groups import add_common, add_cookies


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("download", help="Download video / ingest local file and extract audio.")
    p.add_argument("source", help="Video URL, local video path, or local audio path.")
    p.add_argument("--slug", default=None, help="Override project slug.")
    add_cookies(p)
    add_common(p)


def handler(args: argparse.Namespace) -> None:
    from mazinger.download import (
        is_url, is_audio_file, resolve_slug, slug_from_path,
        download_video, extract_audio, ingest_local_video, ingest_local_audio,
    )
    from mazinger.paths import ProjectPaths

    source = args.source
    remote = is_url(source)

    slug = args.slug
    if slug is None:
        if remote:
            slug, _ = resolve_slug(
                source,
                cookies_from_browser=args.cookies_from_browser,
                cookies=args.cookies,
            )
        else:
            slug = slug_from_path(source)

    proj = ProjectPaths(slug, base_dir=args.base_dir).ensure_dirs()

    if not remote and is_audio_file(source):
        ingest_local_audio(source, proj.audio)
    elif remote:
        download_video(
            source, proj.video,
            cookies_from_browser=args.cookies_from_browser,
            cookies=args.cookies,
        )
        extract_audio(proj.video, proj.audio)
    else:
        ingest_local_video(source, proj.video, proj.audio)

    print(proj.summary())
