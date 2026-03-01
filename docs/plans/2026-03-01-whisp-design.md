# whisp — Minimal Whisper Transcription Service for Pi 5

## Purpose

Standalone HTTP service that accepts WAV audio and returns transcribed text.
Built for Raspberry Pi 5 (4x Cortex-A76, no GPU). Serves a single Telegram bot on localhost.

## API Contract

```
POST /transcribe
Content-Type: multipart/form-data
Body: file=<audio.wav>  (any sample rate, mono, 16-bit)

200 OK
Content-Type: application/json
{"text": "transcribed words"}

400 — missing "file" field or unreadable WAV
500 — inference failure
```

The bot (chatgpt-bot) sends 48kHz mono WAV from Opus-decoded Telegram voice messages.
`read_audio_data()` (via miniaudio) resamples to 16kHz internally. No resampler needed.

## Architecture

Single C++ binary (~200 lines). Links whisper.cpp as a library.

- `httplib.h` for HTTP (vendored in whisper.cpp/examples/server/)
- `whisper.h` for inference
- `common-whisper.h` + `miniaudio.h` for WAV decoding and resampling via `read_audio_data()`
- `json.hpp` (nlohmann) for JSON response formatting
- Single mutex — one inference at a time (sufficient for single-user bot on localhost)

**Note:** `detect_language` must NOT be set to `true` — it causes whisper to exit after
language detection without transcribing. Setting `language = "auto"` alone is sufficient.

## CLI

```
./whisp -m <model-path> [-p 8765] [-t 4] [-l auto]
```

| Flag | Default | Description |
|------|---------|-------------|
| `-m` | (required) | Path to GGML model file |
| `-p` | `8765` | Listen port |
| `-t` | `4` | Inference threads (Pi 5 has 4 cores) |
| `-l` | `auto` | Language (`auto` for multilingual detect) |

## Project Structure

```
whisp/
├── CMakeLists.txt          # links whisper.cpp as subdirectory
├── src/
│   └── main.cpp            # the entire service
├── whisper.cpp/            # existing checkout, git submodule
└── deploy/
    └── whisp.service       # systemd unit for Pi 5
```

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
```

Builds natively on Pi 5. Produces a single static-ish binary.

## Deployment

systemd unit runs on boot, restarts on crash. Model path configured via environment variable.

## Pi 5 Constraints

- **Model:** recommend `ggml-small-q5_0.bin` (~170MB) for multilingual quality/speed balance
- **Memory:** model stays resident in RAM, fits comfortably in 8GB
- **Threads:** 4 (all cores)
- **Concurrency:** single mutex, no queue needed — bot sends one voice message at a time

## Bot Integration

Bot config (`config.json`):
```json
{
  "whisper_endpoint": "http://localhost:8765/transcribe"
}
```

No changes to the bot's audio pipeline. It already sends multipart WAV to the configured endpoint and parses `{"text":"..."}` from the response.
