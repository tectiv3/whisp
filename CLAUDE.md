# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

whisp is a minimal HTTP transcription service wrapping whisper.cpp. Single C++ binary, single endpoint (`POST /transcribe`), designed to run on a Raspberry Pi 5 serving a Telegram bot on localhost.

## Build

Requires whisper.cpp checked out as a subdirectory (git submodule):

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
```

Produces `build/whisp`.

## Run

```bash
./build/whisp -m <model-path> [-p 8765] [-t 4] [-l auto]
./build/whisp --config config.json    # JSON config, CLI flags override
```

## Test

```bash
./test.sh [model-path] [wav-file]
```

Starts the server, sends test requests (valid audio, missing field, bad audio), prints results, then kills the server. Defaults to whisper.cpp's bundled tiny model and jfk.wav sample.

## Architecture

Everything lives in `src/main.cpp` (~160 lines). No abstractions, no modules.

- **HTTP**: `httplib.h` (vendored inside whisper.cpp/examples/server/)
- **Audio decode/resample**: `read_audio_data()` from whisper.cpp's `common-whisper.h` (uses miniaudio internally, resamples any input to 16kHz)
- **JSON**: nlohmann `json.hpp` (vendored in whisper.cpp)
- **Concurrency**: single `std::mutex` — one inference at a time, intentional for single-user use

## Key Constraints

- `detect_language` must NOT be set to `true` — it causes whisper to exit after language detection without transcribing. `language = "auto"` alone handles multilingual detection.
- GPU is disabled (`use_gpu = false`) — target is Pi 5 with no GPU.
- The bot sends 48kHz mono WAV; miniaudio handles resampling internally.

## Deployment

`deploy/whisp.service` is a systemd unit for the Pi 5. Config lives at `/opt/whisp/config.json` in production.
