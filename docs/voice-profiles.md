# Voice Profiles

Voice profiles provide a convenient way to clone a speaker's voice without passing `--voice-sample` and `--voice-script` manually. Profiles are hosted on a public HuggingFace dataset and downloaded on first use.

**Dataset:** https://huggingface.co/datasets/bakrianoo/mazinger-dubber-profiles

## Using a Profile

### CLI

```bash
# Full pipeline
mazinger dub "https://youtube.com/watch?v=VIDEO_ID" --clone-profile abubakr

# Speak sub-command
mazinger speak --srt translated.srt --original-audio audio.mp3 \
    --clone-profile abubakr -o dubbed.wav
```

You can override the voice script while keeping the profile's audio sample:

```bash
mazinger dub "https://youtube.com/watch?v=VIDEO_ID" \
    --clone-profile abubakr --voice-script custom_transcript.txt
```

### Python

```python
from mazinger.profiles import fetch_profile

voice_path, script_path = fetch_profile("abubakr")
```

`fetch_profile` returns a tuple of `(voice_sample_path, voice_script_path)`. Files are cached in `/tmp/mazinger-dubber-profiles/` and reused on subsequent calls.

Non-WAV voice files are automatically converted to 16-kHz mono WAV for TTS compatibility.

## Available Profiles

| Profile | Language | Description |
|---------|----------|-------------|
| `abubakr` | English | Abu Bakr Soliman |

## Profile Structure

Each profile is a folder in the HuggingFace dataset containing two files:

| File | Description |
|------|-------------|
| `script.txt` | Plain-text transcript that matches the voice sample |
| `voice.*` | Voice sample audio (`.wav`, `.m4a`, or `.mp3` — auto-detected) |

## Creating and Uploading a Profile

### 1. Prepare the files

Create a folder with your profile name containing:

- A voice sample recording (10–30 seconds of clear speech works well)
- A plain-text transcript of exactly what is spoken in the recording

```
profiles/my-name/
├── voice.m4a      # or voice.wav, voice.mp3
└── script.txt
```

### 2. Log in to HuggingFace

```bash
hf auth login
```

### 3. Upload

```bash
hf upload bakrianoo/mazinger-dubber-profiles \
    ./profiles/my-name my-name \
    --repo-type=dataset \
    --commit-message "Add profile: my-name"
```

After uploading, the profile is available immediately:

```bash
mazinger dub "https://youtube.com/watch?v=VIDEO_ID" --clone-profile my-name
```

## Tips for Good Voice Samples

- Record in a quiet environment with minimal background noise
- Speak naturally at your normal pace
- 10–30 seconds is enough — longer samples don't improve quality significantly
- The transcript must match the recording exactly, word for word (for Qwen TTS)
- Chatterbox does not use the transcript, only the audio sample
