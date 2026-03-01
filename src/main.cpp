#include "whisper.h"
#include "common-whisper.h"
#include "httplib.h"
#include "json.hpp"

#include <cstdio>
#include <fstream>
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
    fprintf(stderr, "usage: %s [-m <model>] [-p port] [-t threads] [-l language] [--config <path>]\n", argv0);
}

static bool load_config(const std::string & path, params & p) {
    std::ifstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "whisp: cannot open config '%s'\n", path.c_str());
        return false;
    }
    json cfg;
    try { cfg = json::parse(f); }
    catch (const json::parse_error & e) {
        fprintf(stderr, "whisp: config parse error: %s\n", e.what());
        return false;
    }
    if (cfg.contains("model"))    p.model_path = cfg["model"].get<std::string>();
    if (cfg.contains("port"))     p.port       = cfg["port"].get<int>();
    if (cfg.contains("threads"))  p.threads    = cfg["threads"].get<int>();
    if (cfg.contains("language")) p.language   = cfg["language"].get<std::string>();
    return true;
}

static bool parse_args(int argc, char ** argv, params & p) {
    // first pass: load config file so CLI flags can override
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            if (!load_config(argv[++i], p)) return false;
            break;
        }
    }

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            ++i; // already handled
        } else if ((arg == "-m") && i + 1 < argc) {
            p.model_path = argv[++i];
        } else if ((arg == "-p") && i + 1 < argc) {
            try { p.port = std::stoi(argv[++i]); }
            catch (...) { usage(argv[0]); return false; }
        } else if ((arg == "-t") && i + 1 < argc) {
            try { p.threads = std::stoi(argv[++i]); }
            catch (...) { usage(argv[0]); return false; }
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

        std::vector<float> pcmf32;
        std::vector<std::vector<float>> pcmf32s;
        if (!read_audio_data(audio.content, pcmf32, pcmf32s, false)) {
            res.status = 400;
            res.set_content("{\"error\":\"failed to decode audio\"}", "application/json");
            return;
        }

        fprintf(stderr, "whisp: %.1fs audio\n", float(pcmf32.size()) / WHISPER_SAMPLE_RATE);

        std::lock_guard<std::mutex> lock(mtx);

        whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        wparams.print_realtime   = false;
        wparams.print_progress   = false;
        wparams.print_timestamps = false;
        wparams.print_special    = false;
        wparams.language         = p.language.c_str();
        wparams.n_threads        = p.threads;
        wparams.no_context       = true;
        wparams.no_timestamps    = true;

        if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
            res.status = 500;
            res.set_content("{\"error\":\"inference failed\"}", "application/json");
            return;
        }

        std::string text;
        const int n = whisper_full_n_segments(ctx);
        for (int i = 0; i < n; i++) {
            text += whisper_full_get_segment_text(ctx, i);
        }

        // whisper prepends a space to each segment
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
