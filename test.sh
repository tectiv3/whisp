#!/bin/bash
set -e

MODEL="${1:-whisper.cpp/models/ggml-tiny.bin}"
WAV="${2:-whisper.cpp/samples/jfk.wav}"
PORT=8765

cd "$(dirname "$0")"

echo "Starting whisp with model: $MODEL"
./build/whisp -m "$MODEL" 2>/tmp/whisp_test.log &
PID=$!
trap "kill $PID 2>/dev/null; wait $PID 2>/dev/null" EXIT

for i in $(seq 1 30); do
    if curl -s -o /dev/null http://localhost:$PORT/transcribe -X POST 2>/dev/null; then
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "ERROR: server did not start"
        exit 1
    fi
    sleep 1
done

echo "=== POST /transcribe with $WAV ==="
curl -s http://localhost:$PORT/transcribe -F file=@"$WAV"
echo ""

echo "=== POST /transcribe missing file field (expect 400 + error JSON) ==="
echo "dummy" | curl -s -w "\nHTTP %{http_code}\n" \
    http://localhost:$PORT/transcribe -F notfile=@-

echo "=== POST /transcribe bad audio (expect 400) ==="
echo "not a wav" | curl -s -w "\nHTTP %{http_code}\n" \
    http://localhost:$PORT/transcribe -F file=@-
echo ""

echo "=== Server log ==="
grep "whisp:" /tmp/whisp_test.log
