# whisp Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a minimal C++ HTTP service that wraps whisper.cpp to transcribe WAV audio, matching the API expected by chatgpt-bot.

**Architecture:** Single C++ binary linking whisper.cpp as a CMake subdirectory. Uses httplib.h for HTTP, miniaudio (via whisper.cpp's common lib) for WAV decoding/resampling, and nlohmann/json for response formatting. One endpoint, one mutex, one job.

**Tech Stack:** C++17, CMake, whisper.cpp (GGML), httplib.h, miniaudio, nlohmann/json — all vendored in whisper.cpp already.

**Design doc:** `docs/plans/2026-03-01-whisp-design.md`

---

### Task 1: Project scaffolding — CMakeLists.txt

**Files:**
- Create: `CMakeLists.txt`

**Step 1: Write CMakeLists.txt**

```cmake
cmake_minimum_required(VERSION 3.14)
project(whisp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)

# Build whisper library only, skip examples/tests
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
add_subdirectory(whisper.cpp)

add_executable(whisp
    src/main.cpp
    whisper.cpp/examples/common.cpp
    whisper.cpp/examples/common-whisper.cpp
)

target_include_directories(whisp PRIVATE
    whisper.cpp/include
    whisper.cpp/examples
    whisper.cpp/examples/server
)

target_link_libraries(whisp PRIVATE
    whisper
    Threads::Threads
    ${CMAKE_DL_LIBS}
)
```

**Step 2: Create src directory and a stub main.cpp to verify build**

```cpp
#include "whisper.h"
#include <cstdio>

int main() {
    printf("whisp stub\n");
    return 0;
}
```

**Step 3: Verify the build compiles and links**

Run: `cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)`
Expected: Compiles successfully, produces `build/whisp` binary.

**Step 4: Commit**

```bash
git add CMakeLists.txt src/main.cpp
git commit -m "scaffold: CMake project linking whisper.cpp"
```

---

### Task 2: Implement the service — src/main.cpp

**Files:**
- Modify: `src/main.cpp`

This is the entire service. ~150 lines.

**Step 1: Write src/main.cpp**

```cpp
#include "whisper.h"
#include "common-whisper.h"
#include "httplib.h"
#include "json.hpp"

#include <cstdio>
#include <mutex>
#include <string>
#include <vector>

using json = nlohmann::json;

struct params {
    std::string model_path;
    std::string language = "auto";
    int port    = 8765;
    int threads = 4;
};

static void usage(const char * argv0) {
    fprintf(stderr, "usage: %s -m <model> [-p port] [-t threads] [-l language]\n", argv0);
}

static bool parse_args(int argc, char ** argv, params & p) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "-m") && i + 1 < argc) {
            p.model_path = argv[++i];
        } else if ((arg == "-p") && i + 1 < argc) {
            p.port = std::stoi(argv[++i]);
        } else if ((arg == "-t") && i + 1 < argc) {
            p.threads = std::stoi(argv[++i]);
        } else if ((arg == "-l") && i + 1 < argc) {
            p.language = argv[++i];
        } else {
            usage(argv[0]);
            return false;
        }
    }
    if (p.model_path.empty()) {
        usage(argv[0]);
        return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    params p;
    if (!parse_args(argc, argv, p)) {
        return 1;
    }

    // Load model
    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = false;

    fprintf(stderr, "whisp: loading model '%s'\n", p.model_path.c_str());
    struct whisper_context * ctx = whisper_init_from_file_with_params(p.model_path.c_str(), cparams);
    if (!ctx) {
        fprintf(stderr, "whisp: failed to load model\n");
        return 1;
    }
    fprintf(stderr, "whisp: model loaded\n");

    std::mutex mtx;
    httplib::Server svr;

    svr.Post("/transcribe", [&](const httplib::Request & req, httplib::Response & res) {
        if (!req.has_file("file")) {
            res.status = 400;
            res.set_content("{\"error\":\"missing 'file' field\"}", "application/json");
            return;
        }

        auto audio = req.get_file_value("file");

        // Decode WAV (miniaudio resamples to 16kHz internally)
        std::vector<float> pcmf32;
        std::vector<std::vector<float>> pcmf32s;
        if (!read_audio_data(audio.content, pcmf32, pcmf32s, false)) {
            res.status = 400;
            res.set_content("{\"error\":\"failed to decode audio\"}", "application/json");
            return;
        }

        fprintf(stderr, "whisp: received %zu samples (%.1fs)\n",
                pcmf32.size(), float(pcmf32.size()) / WHISPER_SAMPLE_RATE);

        // Run inference (one at a time)
        std::lock_guard<std::mutex> lock(mtx);

        whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        wparams.print_realtime   = false;
        wparams.print_progress   = false;
        wparams.print_timestamps = false;
        wparams.print_special    = false;
        wparams.no_timestamps    = true;
        wparams.language         = p.language.c_str();
        wparams.detect_language  = (p.language == "auto");
        wparams.n_threads        = p.threads;
        wparams.temperature      = 0.0f;
        wparams.no_context       = true;

        if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
            res.status = 500;
            res.set_content("{\"error\":\"inference failed\"}", "application/json");
            return;
        }

        // Collect transcript
        std::string text;
        const int n = whisper_full_n_segments(ctx);
        for (int i = 0; i < n; i++) {
            text += whisper_full_get_segment_text(ctx, i);
        }

        // Trim leading space (whisper quirk)
        if (!text.empty() && text[0] == ' ') {
            text.erase(0, 1);
        }

        json result = {{"text", text}};
        res.set_content(result.dump(), "application/json");
    });

    fprintf(stderr, "whisp: listening on 0.0.0.0:%d\n", p.port);
    svr.listen("0.0.0.0", p.port);

    whisper_free(ctx);
    return 0;
}
```

**Step 2: Build**

Run: `cmake --build build -j$(nproc)`
Expected: Compiles and links successfully.

**Step 3: Smoke test with a WAV file**

Download a test model and audio sample, then test:

```bash
# Get tiny model for testing (if not already available)
cd whisper.cpp/models && bash download-ggml-model.sh tiny && cd ../..

# Start the server
./build/whisp -m whisper.cpp/models/ggml-tiny.bin &

# Send a test WAV
curl -s http://localhost:8765/transcribe \
  -F file=@whisper.cpp/samples/jfk.wav | python3 -m json.tool

# Expected output: {"text": "And so my fellow Americans..."}

# Kill the server
kill %1
```

**Step 4: Test error cases**

```bash
./build/whisp -m whisper.cpp/models/ggml-tiny.bin &

# Missing file field
curl -s -X POST http://localhost:8765/transcribe
# Expected: 400 {"error":"missing 'file' field"}

# Bad audio data
echo "not a wav" > /tmp/bad.wav
curl -s http://localhost:8765/transcribe -F file=@/tmp/bad.wav
# Expected: 400 {"error":"failed to decode audio"}

kill %1
```

**Step 5: Commit**

```bash
git add src/main.cpp
git commit -m "feat: implement transcription service"
```

---

### Task 3: systemd service unit

**Files:**
- Create: `deploy/whisp.service`

**Step 1: Write the service unit**

```ini
[Unit]
Description=whisp transcription service
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/whisp -m /opt/whisp/models/ggml-small-q5_0.bin -t 4 -l auto
Restart=always
RestartSec=3
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

**Step 2: Commit**

```bash
git add deploy/whisp.service
git commit -m "deploy: add systemd service unit"
```

---

### Task 4: Update design doc with resolved details

**Files:**
- Modify: `docs/plans/2026-03-01-whisp-design.md`

**Step 1: Update the design doc**

Fix the design doc to reflect actual implementation:
- `common.h` + `dr_wav.h` → `common-whisper.h` + `miniaudio.h` (miniaudio handles WAV, not dr_wav)
- Add note about `read_audio_data()` being the actual function name
- Add note about `stb_vorbis` being pulled in transitively

**Step 2: Commit**

```bash
git add docs/plans/2026-03-01-whisp-design.md
git commit -m "docs: update design with resolved implementation details"
```
