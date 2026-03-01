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

# Wait for server to be ready
for i in $(seq 1 30); do
    if curl -s -o /dev/null -w '' http://localhost:$PORT/transcribe -X POST 2>/dev/null; then
        break
    fi
    sleep 1
done

echo "=== POST /transcribe with $WAV ==="
curl -s http://localhost:$PORT/transcribe -F file=@"$WAV"
echo ""

echo "=== POST /transcribe missing file (expect 400) ==="
curl -s -w "\nHTTP %{http_code}\n" -X POST http://localhost:$PORT/transcribe -H "Content-Type: multipart/form-data; boundary=----empty"
echo ""

echo "=== Server log ==="
grep "whisp:" /tmp/whisp_test.log
