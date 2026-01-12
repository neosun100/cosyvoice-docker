# CosyVoice TTS Server - Lightweight Edition

This is a lightweight, optimized version of the CosyVoice TTS server.
Forked from [neosun's repository](https://github.com/neosun/cosyvoice-docker),
we reduced the image size from ~29GB to ~12GB using multi-stage builds.

## Features

- OpenAI-compatible API (`/v1/audio/speech`)
- Voice management (create, list, delete custom voices)
- Multi-model support with runtime switching
- GPU acceleration support
- Lightweight Docker image (~12GB vs original 29GB)

## Quick Start

### Prerequisites

Download ttsfrd resource files (required for better text normalization):

```bash
git clone https://www.modelscope.cn/iic/CosyVoice-ttsfrd.git
cd CosyVoice-ttsfrd
unzip resource.zip -d .
# Place resource.zip in the backup/ directory
```

### 1. Download Model

```bash
pip install huggingface_hub
huggingface-cli download FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --local-dir ./models
```

### 2. Build Image

```bash
docker build -t cosyvoice:latest .
```

### 3. Run Service

```bash
# Copy and edit docker-compose configuration
cp docker-compose-example.yml docker-compose.yml
# Edit paths in docker-compose.yml to match your local directories

# Run service
docker-compose up -d
```

## Mount Points

| Mount Point | Container Path | Description |
|-------------|----------------|-------------|
| Models | `/models` | TTS model files (read-only) |
| Output | `/data/output` | Generated audio files |
| Voices | `/data/voices` | Custom voice storage |
| Cache | `/root/.cache` | HuggingFace/transformers cache |

## API Usage

### Generate Speech

```bash
curl -X POST http://localhost:8004/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world", "voice": "english_female"}' \
  -o output.wav
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `input` | string | ✅ | Text to synthesize |
| `voice` | string | ✅ | Voice ID (pretrained or custom) |
| `speed` | float | ❌ | Speech speed 0.5-2.0 (default 1.0) |
| `response_format` | string | ❌ | Output format: wav/pcm (default wav) |

### Create Custom Voice

```bash
curl -X POST http://localhost:8004/v1/audio/voices \
  -F "file=@reference.wav" \
  -F "name=My Voice" \
  -F "text=Reference text"
```

### List Voices

```bash
curl http://localhost:8004/v1/audio/voices
curl http://localhost:8004/v1/audio/voices?type=pretrained
curl http://localhost:8004/v1/audio/voices?type=custom
```

### Delete Voice

```bash
curl -X DELETE http://localhost:8004/v1/audio/voices/{voice_id}
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | - | Path to model directory |
| `OUTPUT_PATH` | `./data/output` | Path for generated audio |
| `VOICES_PATH` | `./data/voices` | Path for custom voices |

### Ports

- Default: 8004 (changed from 8005 to avoid conflicts)

## Project Structure

```
.
├── Dockerfile              # Multi-stage build for optimized image
├── docker-compose-example.yml  # Example configuration template
├── requirements.txt        # Python dependencies
├── server.py               # FastAPI server with OpenAI-compatible endpoints
├── voice_manager.py        # Custom voice management
├── backup/                 # Contains ttsfrd wheels and resource.zip
├── cosyvoice/              # CosyVoice core code
└── third_party/            # Third-party dependencies
```

## License

Apache License 2.0
