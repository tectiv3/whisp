# whisp

A minimal HTTP transcription service wrapping [whisper.cpp](https://github.com/ggml-org/whisper.cpp). One small C++ binary, one endpoint (`POST /transcribe`). Built to run on a Raspberry Pi 5 (or an Intel Mac) and serve a single localhost client such as a Telegram bot.

- Single inference at a time, guarded by one mutex — intentional for single-user use.
- Accepts any audio format; miniaudio (via whisper.cpp) decodes and resamples to 16 kHz internally.
- CPU-only (`use_gpu = false`).

## Requirements

- A C++17 compiler and CMake ≥ 3.14
- [whisper.cpp](https://github.com/ggml-org/whisper.cpp) checked out as a `whisper.cpp/` subdirectory
- A GGML whisper model (e.g. `ggml-small.bin`)

Fetch whisper.cpp and a model:

```bash
git clone https://github.com/ggml-org/whisper.cpp
sh ./whisper.cpp/models/download-ggml-model.sh small
```

## Build

The same CMake build works on Linux and macOS. Produces `build/whisp`.

### Linux (Raspberry Pi 5, x86-64)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

On a Pi 5, `-DGGML_NATIVE=ON` (the default) auto-detects the Cortex-A76 NEON support. The only system dependency beyond the toolchain is CMake; install with `sudo apt install build-essential cmake`.

### macOS (native)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(sysctl -n hw.ncpu)
```

whisper.cpp links Accelerate automatically for the BLAS path. Metal initializes but inference falls back to CPU/BLAS, which is the intended path here (`use_gpu = false`). Requires the Xcode command-line tools (`xcode-select --install`) and CMake (`brew install cmake`).

### Cross-compile for Intel Mac (from Apple Silicon)

Targeting a MacBook Air with an Ice Lake i5. `-DGGML_NATIVE=OFF` prevents host CPU detection; `-march=icelake-client` enables AVX-512 and the Accelerate BLAS path on the target (~2× over a generic x86-64 build).

```bash
cmake -B build-x86 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES=x86_64 \
  -DGGML_NATIVE=OFF \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_CXX_FLAGS="-march=icelake-client" \
  -DCMAKE_C_FLAGS="-march=icelake-client"
cmake --build build-x86 -j$(nproc)
```

Copy `build-x86/whisp` to the target, then sign it before adding to the macOS firewall:

```bash
codesign --sign - build-x86/whisp
```

## Run

```bash
./build/whisp -m <model-path> [-p 8765] [-t 4] [-l auto]
./build/whisp --config config.json    # JSON config; CLI flags override
```

The server logs `whisp: listening on 0.0.0.0:<port>` once the model is loaded.

## Configure

| Flag | Config key | Default | Description |
|------|-----------|---------|-------------|
| `-m`   | `model`    | *(required)* | Path to the GGML model file |
| `-p`   | `port`     | `8765`  | TCP port to listen on |
| `-t`   | `threads`  | `4`     | Inference threads |
| `-l`   | `language` | `auto`  | Language code, or `auto` to detect |
| `--config` | — | — | Path to a JSON config file |

A config file is loaded first; any CLI flags given alongside `--config` override the file's values. Example `config.json`:

```json
{
  "model": "/opt/whisp/models/ggml-small.bin",
  "port": 8765,
  "threads": 4,
  "language": "auto"
}
```

> **Note:** leave `language` as `auto` for multilingual detection. Do not enable whisper's `detect_language` — it causes whisper to exit after detecting the language without transcribing.

## Use

`POST /transcribe` with the audio as a multipart form field named `file`:

```bash
curl -s http://localhost:8765/transcribe -F file=@audio.wav
```

Response:

```json
{"text": "the transcribed text"}
```

Errors return a non-2xx status with a JSON body:

| Status | Body | Cause |
|--------|------|-------|
| `400` | `{"error":"missing 'file' field"}` | No `file` form field |
| `400` | `{"error":"failed to decode audio"}` | Audio could not be decoded |
| `500` | `{"error":"inference failed"}` | whisper inference error |

A smoke test is included:

```bash
./test.sh [model-path] [wav-path]
# defaults: whisper.cpp/models/ggml-tiny.bin  whisper.cpp/samples/jfk.wav
```

## Deploy

### Raspberry Pi 5 — systemd

Install the binary and config, then enable the unit. The unit in `deploy/whisp.service` expects the binary at `/usr/local/bin/whisp` and config at `/opt/whisp/config.json`.

```bash
sudo install -Dm755 build/whisp /usr/local/bin/whisp
sudo install -Dm644 config.json /opt/whisp/config.json   # edit paths first

sudo cp deploy/whisp.service /etc/systemd/system/whisp.service
sudo systemctl daemon-reload
sudo systemctl enable --now whisp
```

Manage and inspect:

```bash
sudo systemctl status whisp
sudo systemctl restart whisp
sudo journalctl -u whisp -f      # follow logs
```

### Intel Mac — launchd

Binary and config live at `~/bin/`. Edit the paths in `deploy/whisp.plist` to match, then:

```bash
cp deploy/whisp.plist ~/Library/LaunchAgents/me.whisp.plist
launchctl load ~/Library/LaunchAgents/me.whisp.plist
```

Logs: `~/Library/Logs/whisp.log`

Manage:

```bash
launchctl stop me.whisp
launchctl start me.whisp
launchctl unload ~/Library/LaunchAgents/me.whisp.plist
```
