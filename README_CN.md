[English](README.md) | [简体中文](README_CN.md) | [繁體中文](README_TW.md) | [日本語](README_JP.md)

<div align="center">

# 🎙️ CosyVoice All-in-One Docker

[![Docker Pulls](https://img.shields.io/docker/pulls/neosun/cosyvoice?style=flat-square&logo=docker)](https://hub.docker.com/r/neosun/cosyvoice)
[![Docker Image Version](https://img.shields.io/docker/v/neosun/cosyvoice?style=flat-square&logo=docker&sort=semver)](https://hub.docker.com/r/neosun/cosyvoice)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=flat-square)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/neosun100/cosyvoice-docker?style=flat-square&logo=github)](https://github.com/neosun100/cosyvoice-docker)

**基于 Fun-CosyVoice3-0.5B 的生产级语音合成服务**

一条 Docker 命令即可获得 Web UI + REST API + 语音克隆

[快速开始](#-快速开始) • [功能特性](#-功能特性) • [API 文档](#-api-接口) • [性能测试](#-性能基准测试)

</div>

---

## 📸 界面截图

![Web UI](https://img.aws.xin/uPic/o1Qj12.png)

## ✨ 功能特性

| 功能 | 说明 |
|------|------|
| 🎯 **Fun-CosyVoice3-0.5B** | 阿里最新最优 TTS 模型 |
| 🎤 **Fun-ASR-Nano** | 自动语音识别（替代 Whisper） |
| 🔌 **OpenAI 兼容 API** | 可直接替换 `/v1/audio/speech` |
| 👤 **自定义音色管理** | 上传一次，ID 调用 |
| ⚡ **真正的流式输出** | PCM 逐块输出，~1.2s 首包延迟 |
| 🚀 **Embedding 缓存** | 首次使用后提速 53% |
| 🌐 **Web UI** | 精美界面，支持下载 |
| 🌍 **多语言支持** | 中英日韩 + 18 种方言 |

## 🚀 快速开始

```bash
docker run -d \
  --name cosyvoice \
  --gpus '"device=0"' \
  -p 8188:8188 \
  -v cosyvoice-data:/data/voices \
  neosun/cosyvoice:latest
```

然后打开 http://localhost:8188 🎉

## 📦 安装部署

### 前置条件

- Docker 20.10+
- Docker Compose v2.0+（可选）
- NVIDIA GPU，显存 8GB+
- NVIDIA Container Toolkit

### Docker Run

```bash
# 拉取镜像
docker pull neosun/cosyvoice:v3.4.0

# 启动容器
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

### 健康检查

```bash
curl http://localhost:8188/health
# {"status":"healthy","gpu":{"model_loaded":true,...}}
```

## ⚙️ 配置说明

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `PORT` | `8188` | 服务端口 |
| `MODEL_DIR` | `pretrained_models/Fun-CosyVoice3-0.5B` | TTS 模型路径 |

### 数据卷

| 路径 | 说明 |
|------|------|
| `/data/voices` | 自定义音色存储（持久化） |

## 📡 API 接口

### 端点列表

| 端点 | 方法 | 说明 |
|------|------|------|
| `/v1/audio/speech` | POST | 语音合成（OpenAI 兼容） |
| `/v1/voices/create` | POST | 创建自定义音色 |
| `/v1/voices/custom` | GET | 列出自定义音色 |
| `/v1/voices/{id}` | GET/DELETE | 获取/删除音色 |
| `/v1/models` | GET | 列出模型 |
| `/health` | GET | 健康检查 |
| `/docs` | GET | Swagger 文档 |

### 创建自定义音色

```bash
# 提供文本
curl -X POST http://localhost:8188/v1/voices/create \
  -F "audio=@voice.wav" \
  -F "name=我的音色" \
  -F "text=参考文本内容"

# 自动转写（使用 Fun-ASR-Nano）
curl -X POST http://localhost:8188/v1/voices/create \
  -F "audio=@voice.wav" \
  -F "name=我的音色"

# 返回: {"voice_id": "abc123", "text": "自动识别的文本", ...}
```

### 语音合成

```bash
# WAV 格式
curl http://localhost:8188/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "你好世界", "voice": "abc123"}' \
  -o output.wav

# PCM 流式（最低延迟）
curl http://localhost:8188/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "你好世界", "voice": "abc123", "response_format": "pcm"}' \
  -o output.pcm

# PCM 转 WAV
ffmpeg -f s16le -ar 24000 -ac 1 -i output.pcm output.wav
```

### Python 示例

```python
import requests

# 创建音色
with open("voice.wav", "rb") as f:
    resp = requests.post(
        "http://localhost:8188/v1/voices/create",
        files={"audio": f},
        data={"name": "我的音色"}
    )
    voice_id = resp.json()["voice_id"]

# 生成语音
resp = requests.post(
    "http://localhost:8188/v1/audio/speech",
    json={"input": "你好世界", "voice": voice_id}
)
with open("output.wav", "wb") as f:
    f.write(resp.content)
```

## 📊 性能基准测试

**测试环境：** NVIDIA L40S GPU

### 首包延迟 (TTFB)

| 文本长度 | 首包延迟 | 总时间 | 音频时长 | RTF |
|---------|---------|--------|---------|-----|
| 短文本(4字) | **1.20s** | 1.55s | 1.88s | 0.82x |
| 短文本(10字) | **1.34s** | 1.75s | 2.28s | 0.77x |
| 中文本(30字) | **1.24s** | 4.98s | 6.88s | 0.72x |
| 中文本(50字) | **1.27s** | 12.52s | 17.12s | 0.73x |
| 长文本(80字) | **1.24s** | 17.91s | 23.68s | 0.76x |
| 长文本(120字) | **1.35s** | 19.08s | 25.32s | 0.75x |

> RTF (实时率) < 1.0 表示生成速度快于播放速度

### Embedding 缓存效果

| 场景 | 首包延迟 | 说明 |
|------|---------|------|
| 首次使用（无缓存） | ~3.5s | 提取特征 + 缓存到 GPU |
| 缓存命中 | **~1.2s** | 直接从缓存读取 |
| **提升** | **-53%** | |

### ASR (Fun-ASR-Nano) 性能测试

| 音频 | 语言 | 时长 | 识别耗时 | 识别结果 |
|------|------|------|---------|---------|
| 音色样本 | 中文 | ~7s | **0.40s** | 希望你以后能够做的比我还好哟。 |
| 音色样本 | 中文 | ~7s | **0.83s** | 对，这就是我万人敬仰的太乙真人... |
| zh.mp3 | 中文 | ~3s | **0.40s** | 开放时间早上九点至下午五点。 |
| en.mp3 | 英文 | ~5s | **0.70s** | The tribal chieftain called for the boy... |
| ja.mp3 | 日文 | ~5s | **0.84s** | うちの中学は弁当制で... |

> 平均识别耗时: **0.4-0.8s** / 音频文件

## 🗣️ 支持语言

### TTS (Fun-CosyVoice3)
- **主要语言**: 中文、英文、日语、韩语
- **欧洲语言**: 德语、西班牙语、法语、意大利语、俄语
- **中文方言**: 广东话、四川话、东北话、上海话、闽南语等 18+ 种

### ASR (Fun-ASR-Nano)
- **支持语言**: 中文、英文、日语 + 自动检测
- **中文方言**: 7 大方言 + 26 种地方口音
- **特性**: 高噪声识别、歌词识别

## 🛠️ 技术栈

- **TTS 模型:** [Fun-CosyVoice3-0.5B](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)
- **ASR 模型:** [Fun-ASR-Nano-2512](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512)
- **框架:** FastAPI + Gradio
- **运行时:** PyTorch + CUDA
- **容器:** Docker + NVIDIA Container Toolkit

## 📋 更新日志

| 版本 | 日期 | 更新内容 |
|------|------|---------|
| v3.4.0 | 2024-12-18 | Fun-ASR-Nano 替代 Whisper |
| v3.3.0 | 2024-12-18 | UI 改进：流式默认、下载按钮、计时器 |
| v3.2.1 | 2024-12-18 | 启动时自动预热所有音色 |
| v3.2.0 | 2024-12-18 | Embedding 缓存（-53% TTFB） |
| v3.1.0 | 2024-12-18 | 轮询优化 + 模型预热 |
| v3.0.0 | 2024-12-18 | All-in-One Docker 基础版 |

## 🤝 贡献指南

欢迎贡献！请随时提交 Pull Request。

1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/amazing`)
3. 提交更改 (`git commit -m '添加新功能'`)
4. 推送到分支 (`git push origin feature/amazing`)
5. 创建 Pull Request

## 📄 许可证

本项目采用 Apache License 2.0 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice) - 原始 CosyVoice 项目
- [FunAudioLLM/Fun-ASR](https://github.com/FunAudioLLM/Fun-ASR) - Fun-ASR-Nano 模型

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=neosun100/cosyvoice-docker&type=Date)](https://star-history.com/#neosun100/cosyvoice-docker)

## 📱 关注公众号

![公众号](https://img.aws.xin/uPic/扫码_搜索联合传播样式-标准色版.png)
