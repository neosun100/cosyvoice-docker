[English](README.md) | [简体中文](README_CN.md) | [繁體中文](README_TW.md) | [日本語](README_JP.md)

<div align="center">

# 🎙️ CosyVoice All-in-One Docker

[![Docker Pulls](https://img.shields.io/docker/pulls/neosun/cosyvoice?style=flat-square&logo=docker)](https://hub.docker.com/r/neosun/cosyvoice)
[![Docker Image Version](https://img.shields.io/docker/v/neosun/cosyvoice?style=flat-square&logo=docker&sort=semver)](https://hub.docker.com/r/neosun/cosyvoice)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=flat-square)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/neosun100/cosyvoice-docker?style=flat-square&logo=github)](https://github.com/neosun100/cosyvoice-docker)

**基於 Fun-CosyVoice3-0.5B 的生產級語音合成服務**

一條 Docker 命令即可獲得 Web UI + REST API + 語音克隆

[快速開始](#-快速開始) • [功能特性](#-功能特性) • [API 文檔](#-api-接口) • [性能測試](#-性能基準測試)

</div>

---

## 📸 介面截圖

![Web UI](https://img.aws.xin/uPic/o1Qj12.png)

## ✨ 功能特性

| 功能 | 說明 |
|------|------|
| 🎯 **Fun-CosyVoice3-0.5B** | 阿里最新最優 TTS 模型 |
| 🎤 **Fun-ASR-Nano** | 自動語音識別（替代 Whisper） |
| 🔌 **OpenAI 相容 API** | 可直接替換 `/v1/audio/speech` |
| 👤 **自訂音色管理** | 上傳一次，ID 調用 |
| ⚡ **真正的串流輸出** | PCM 逐塊輸出，~1.2s 首包延遲 |
| 🚀 **Embedding 快取** | 首次使用後提速 53% |
| 🌐 **Web UI** | 精美介面，支援下載 |
| 🌍 **多語言支援** | 中英日韓 + 18 種方言 |

## 🚀 快速開始

```bash
docker run -d \
  --name cosyvoice \
  --gpus '"device=0"' \
  -p 8188:8188 \
  -v cosyvoice-data:/data/voices \
  neosun/cosyvoice:latest
```

然後開啟 http://localhost:8188 🎉

## 📦 安裝部署

### 前置條件

- Docker 20.10+
- Docker Compose v2.0+（可選）
- NVIDIA GPU，顯存 8GB+
- NVIDIA Container Toolkit

### Docker Run

```bash
# 拉取映像
docker pull neosun/cosyvoice:v3.4.0

# 啟動容器
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

### 健康檢查

```bash
curl http://localhost:8188/health
# {"status":"healthy","gpu":{"model_loaded":true,...}}
```

## ⚙️ 配置說明

### 環境變數

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `PORT` | `8188` | 服務埠 |
| `MODEL_DIR` | `pretrained_models/Fun-CosyVoice3-0.5B` | TTS 模型路徑 |

### 資料卷

| 路徑 | 說明 |
|------|------|
| `/data/voices` | 自訂音色儲存（持久化） |

## 📡 API 接口

### 端點列表

| 端點 | 方法 | 說明 |
|------|------|------|
| `/v1/audio/speech` | POST | 語音合成（OpenAI 相容） |
| `/v1/voices/create` | POST | 建立自訂音色 |
| `/v1/voices/custom` | GET | 列出自訂音色 |
| `/v1/voices/{id}` | GET/DELETE | 取得/刪除音色 |
| `/v1/models` | GET | 列出模型 |
| `/health` | GET | 健康檢查 |
| `/docs` | GET | Swagger 文檔 |

### 建立自訂音色

```bash
# 提供文字
curl -X POST http://localhost:8188/v1/voices/create \
  -F "audio=@voice.wav" \
  -F "name=我的音色" \
  -F "text=參考文字內容"

# 自動轉寫（使用 Fun-ASR-Nano）
curl -X POST http://localhost:8188/v1/voices/create \
  -F "audio=@voice.wav" \
  -F "name=我的音色"

# 回傳: {"voice_id": "abc123", "text": "自動識別的文字", ...}
```

### 語音合成

```bash
# WAV 格式
curl http://localhost:8188/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "你好世界", "voice": "abc123"}' \
  -o output.wav

# PCM 串流（最低延遲）
curl http://localhost:8188/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "你好世界", "voice": "abc123", "response_format": "pcm"}' \
  -o output.pcm

# PCM 轉 WAV
ffmpeg -f s16le -ar 24000 -ac 1 -i output.pcm output.wav
```

## 📊 性能基準測試

**測試環境：** NVIDIA L40S GPU

### 首包延遲 (TTFB)

| 文字長度 | 首包延遲 | 總時間 | 音訊時長 | RTF |
|---------|---------|--------|---------|-----|
| 短文字(4字) | **1.20s** | 1.55s | 1.88s | 0.82x |
| 短文字(10字) | **1.34s** | 1.75s | 2.28s | 0.77x |
| 中文字(30字) | **1.24s** | 4.98s | 6.88s | 0.72x |
| 中文字(50字) | **1.27s** | 12.52s | 17.12s | 0.73x |
| 長文字(80字) | **1.24s** | 17.91s | 23.68s | 0.76x |
| 長文字(120字) | **1.35s** | 19.08s | 25.32s | 0.75x |

> RTF (即時率) < 1.0 表示生成速度快於播放速度

### Embedding 快取效果

| 場景 | 首包延遲 | 說明 |
|------|---------|------|
| 首次使用（無快取） | ~3.5s | 提取特徵 + 快取到 GPU |
| 快取命中 | **~1.2s** | 直接從快取讀取 |
| **提升** | **-53%** | |

### ASR (Fun-ASR-Nano) 性能測試

| 音訊 | 語言 | 時長 | 識別耗時 | 識別結果 |
|------|------|------|---------|---------|
| 音色樣本 | 中文 | ~7s | **0.40s** | 希望你以後能夠做的比我還好喲。 |
| 音色樣本 | 中文 | ~7s | **0.83s** | 對，這就是我萬人敬仰的太乙真人... |
| zh.mp3 | 中文 | ~3s | **0.40s** | 開放時間早上九點至下午五點。 |
| en.mp3 | 英文 | ~5s | **0.70s** | The tribal chieftain called for the boy... |
| ja.mp3 | 日文 | ~5s | **0.84s** | うちの中学は弁当制で... |

> 平均識別耗時: **0.4-0.8s** / 音訊檔案

## 🗣️ 支援語言

### TTS (Fun-CosyVoice3)
- **主要語言**: 中文、英文、日語、韓語
- **歐洲語言**: 德語、西班牙語、法語、義大利語、俄語
- **中文方言**: 廣東話、四川話、東北話、上海話、閩南語等 18+ 種

### ASR (Fun-ASR-Nano)
- **支援語言**: 中文、英文、日語 + 自動檢測
- **中文方言**: 7 大方言 + 26 種地方口音
- **特性**: 高噪音識別、歌詞識別

## 🛠️ 技術棧

- **TTS 模型:** [Fun-CosyVoice3-0.5B](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)
- **ASR 模型:** [Fun-ASR-Nano-2512](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512)
- **框架:** FastAPI + Gradio
- **執行環境:** PyTorch + CUDA
- **容器:** Docker + NVIDIA Container Toolkit

## 📋 更新日誌

| 版本 | 日期 | 更新內容 |
|------|------|---------|
| v3.4.0 | 2024-12-18 | Fun-ASR-Nano 替代 Whisper |
| v3.3.0 | 2024-12-18 | UI 改進：串流預設、下載按鈕、計時器 |
| v3.2.1 | 2024-12-18 | 啟動時自動預熱所有音色 |
| v3.2.0 | 2024-12-18 | Embedding 快取（-53% TTFB） |
| v3.1.0 | 2024-12-18 | 輪詢優化 + 模型預熱 |
| v3.0.0 | 2024-12-18 | All-in-One Docker 基礎版 |

## 🤝 貢獻指南

歡迎貢獻！請隨時提交 Pull Request。

1. Fork 本倉庫
2. 建立功能分支 (`git checkout -b feature/amazing`)
3. 提交更改 (`git commit -m '新增功能'`)
4. 推送到分支 (`git push origin feature/amazing`)
5. 建立 Pull Request

## 📄 授權條款

本專案採用 Apache License 2.0 授權 - 詳見 [LICENSE](LICENSE) 檔案。

## 🙏 致謝

- [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice) - 原始 CosyVoice 專案
- [FunAudioLLM/Fun-ASR](https://github.com/FunAudioLLM/Fun-ASR) - Fun-ASR-Nano 模型

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=neosun100/cosyvoice-docker&type=Date)](https://star-history.com/#neosun100/cosyvoice-docker)

## 📱 關注公眾號

![公眾號](https://img.aws.xin/uPic/扫码_搜索联合传播样式-标准色版.png)
