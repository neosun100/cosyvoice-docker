[English](README.md) | [简体中文](README_CN.md) | [繁體中文](README_TW.md) | [日本語](README_JP.md)

<div align="center">

# 🎙️ CosyVoice All-in-One Docker

[![Docker Pulls](https://img.shields.io/docker/pulls/neosun/cosyvoice?style=flat-square&logo=docker)](https://hub.docker.com/r/neosun/cosyvoice)
[![Docker Image Version](https://img.shields.io/docker/v/neosun/cosyvoice?style=flat-square&logo=docker&sort=semver)](https://hub.docker.com/r/neosun/cosyvoice)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=flat-square)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/neosun100/cosyvoice-docker?style=flat-square&logo=github)](https://github.com/neosun100/cosyvoice-docker)

**Production-ready Text-to-Speech service based on Fun-CosyVoice3-0.5B**

One Docker command to get Web UI + REST API + Voice Cloning

[Quick Start](#-quick-start) • [Features](#-features) • [API Docs](#-api-reference) • [Performance](#-performance-benchmarks)

</div>

---

## 📸 Screenshot

![Web UI](https://img.aws.xin/uPic/o1Qj12.png)

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎯 **Fun-CosyVoice3-0.5B** | Latest & best TTS model from Alibaba |
| 🎤 **Fun-ASR-Nano** | Auto speech recognition (replaces Whisper) |
| 🔌 **OpenAI Compatible API** | Drop-in replacement for `/v1/audio/speech` |
| 👤 **Custom Voice Management** | Upload once, use by voice_id |
| ⚡ **Real Streaming Output** | PCM chunk-by-chunk, ~1.2s TTFB |
| 🚀 **Embedding Cache** | 53% faster after first use |
| 🌐 **Web UI** | Beautiful interface with download button |
| 🌍 **Multi-language** | Chinese, English, Japanese, Korean + 18 dialects |

## 🚀 Quick Start

```bash
docker run -d \
  --name cosyvoice \
  --gpus '"device=0"' \
  -p 8188:8188 \
  -v cosyvoice-data:/data/voices \
  neosun/cosyvoice:latest
```

Then open http://localhost:8188 🎉

## 📦 Installation

### Prerequisites

- Docker 20.10+
- Docker Compose v2.0+ (optional)
- NVIDIA GPU with 8GB+ VRAM
- NVIDIA Container Toolkit

### Docker Run

```bash
# Pull the image
docker pull neosun/cosyvoice:v3.4.0

# Run with GPU
docker run -d \
  --name cosyvoice \
  --gpus '"device=0"' \
  -p 8188:8188 \
  -v /path/to/voices:/data/voices \
  --restart unless-stopped \
  neosun/cosyvoice:v3.4.0
```

### Docker Compose

```yaml
# docker-compose.yml
services:
  cosyvoice:
    image: neosun/cosyvoice:v3.4.0
    container_name: cosyvoice
    restart: unless-stopped
    ports:
      - "8188:8188"
    volumes:
      - ./voices:/data/voices
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["0"]
              capabilities: [gpu]
```

```bash
docker compose up -d
```

### Health Check

```bash
curl http://localhost:8188/health
# {"status":"healthy","gpu":{"model_loaded":true,...}}
```

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8188` | Service port |
| `MODEL_DIR` | `pretrained_models/Fun-CosyVoice3-0.5B` | TTS model path |

### Volume Mounts

| Path | Description |
|------|-------------|
| `/data/voices` | Custom voice storage (persistent) |

## 📡 API Reference

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/audio/speech` | POST | Text-to-Speech (OpenAI compatible) |
| `/v1/voices/create` | POST | Create custom voice |
| `/v1/voices/custom` | GET | List custom voices |
| `/v1/voices/{id}` | GET/DELETE | Get/Delete voice |
| `/v1/models` | GET | List models |
| `/health` | GET | Health check |
| `/docs` | GET | Swagger UI |

### Create Custom Voice

```bash
# With text
curl -X POST http://localhost:8188/v1/voices/create \
  -F "audio=@voice.wav" \
  -F "name=MyVoice" \
  -F "text=Reference text content"

# Auto transcribe (using Fun-ASR-Nano)
curl -X POST http://localhost:8188/v1/voices/create \
  -F "audio=@voice.wav" \
  -F "name=MyVoice"

# Response: {"voice_id": "abc123", "text": "auto transcribed text", ...}
```

### Text-to-Speech

```bash
# WAV format
curl http://localhost:8188/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world", "voice": "abc123"}' \
  -o output.wav

# PCM streaming (lowest latency)
curl http://localhost:8188/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world", "voice": "abc123", "response_format": "pcm"}' \
  -o output.pcm

# Convert PCM to WAV
ffmpeg -f s16le -ar 24000 -ac 1 -i output.pcm output.wav
```

### Python Example

```python
import requests

# Create voice
with open("voice.wav", "rb") as f:
    resp = requests.post(
        "http://localhost:8188/v1/voices/create",
        files={"audio": f},
        data={"name": "MyVoice"}
    )
    voice_id = resp.json()["voice_id"]

# Generate speech
resp = requests.post(
    "http://localhost:8188/v1/audio/speech",
    json={"input": "Hello world", "voice": voice_id}
)
with open("output.wav", "wb") as f:
    f.write(resp.content)
```

## 📊 Performance Benchmarks

**Test Environment:** NVIDIA L40S GPU

### First Token Latency (TTFB)

| Text Length | TTFB | Total Time | Audio Duration | RTF |
|-------------|------|------------|----------------|-----|
| Short (4 chars) | **1.20s** | 1.55s | 1.88s | 0.82x |
| Short (10 chars) | **1.34s** | 1.75s | 2.28s | 0.77x |
| Medium (30 chars) | **1.24s** | 4.98s | 6.88s | 0.72x |
| Medium (50 chars) | **1.27s** | 12.52s | 17.12s | 0.73x |
| Long (80 chars) | **1.24s** | 17.91s | 23.68s | 0.76x |
| Long (120 chars) | **1.35s** | 19.08s | 25.32s | 0.75x |

> RTF (Real-Time Factor) < 1.0 means generation is faster than playback

### Embedding Cache Effect

| Scenario | TTFB | Note |
|----------|------|------|
| First use (no cache) | ~3.5s | Extract + cache to GPU |
| Cached | **~1.2s** | Direct from cache |
| **Improvement** | **-53%** | |

### ASR (Fun-ASR-Nano) Benchmark

| Audio | Language | Duration | Recognition Time | Result |
|-------|----------|----------|------------------|--------|
| Voice sample | Chinese | ~7s | **0.40s** | 希望你以后能够做的比我还好哟。 |
| Voice sample | Chinese | ~7s | **0.83s** | 对，这就是我万人敬仰的太乙真人... |
| zh.mp3 | Chinese | ~3s | **0.40s** | 开放时间早上九点至下午五点。 |
| en.mp3 | English | ~5s | **0.70s** | The tribal chieftain called for the boy... |
| ja.mp3 | Japanese | ~5s | **0.84s** | うちの中学は弁当制で... |

> Average recognition time: **0.4-0.8s** per audio file

## 🗣️ Supported Languages

### TTS (Fun-CosyVoice3)
- **Main:** Chinese, English, Japanese, Korean
- **European:** German, Spanish, French, Italian, Russian
- **Chinese Dialects:** Cantonese, Sichuan, Dongbei, Shanghai, Minnan + 18 more

### ASR (Fun-ASR-Nano)
- **Languages:** Chinese, English, Japanese + auto detection
- **Dialects:** 7 major Chinese dialects + 26 regional accents
- **Features:** High-noise recognition, lyric recognition

## 🛠️ Tech Stack

- **TTS Model:** [Fun-CosyVoice3-0.5B](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)
- **ASR Model:** [Fun-ASR-Nano-2512](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512)
- **Framework:** FastAPI + Gradio
- **Runtime:** PyTorch + CUDA
- **Container:** Docker + NVIDIA Container Toolkit

## 📋 Changelog

| Version | Date | Changes |
|---------|------|---------|
| v3.4.0 | 2024-12-18 | Fun-ASR-Nano replaces Whisper |
| v3.3.0 | 2024-12-18 | UI: streaming default, download button, timer |
| v3.2.1 | 2024-12-18 | Auto preload all voices on startup |
| v3.2.0 | 2024-12-18 | Embedding cache (-53% TTFB) |
| v3.1.0 | 2024-12-18 | Polling optimization + model preload |
| v3.0.0 | 2024-12-18 | All-in-One Docker base version |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing`)
5. Open a Pull Request

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice) - Original CosyVoice project
- [FunAudioLLM/Fun-ASR](https://github.com/FunAudioLLM/Fun-ASR) - Fun-ASR-Nano model

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=neosun100/cosyvoice-docker&type=Date)](https://star-history.com/#neosun100/cosyvoice-docker)

## 📱 Follow Us

![WeChat](https://img.aws.xin/uPic/扫码_搜索联合传播样式-标准色版.png)
