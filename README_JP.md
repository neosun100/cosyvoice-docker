[English](README.md) | [简体中文](README_CN.md) | [繁體中文](README_TW.md) | [日本語](README_JP.md)

<div align="center">

# 🎙️ CosyVoice All-in-One Docker

[![Docker Pulls](https://img.shields.io/docker/pulls/neosun/cosyvoice?style=flat-square&logo=docker)](https://hub.docker.com/r/neosun/cosyvoice)
[![Docker Image Version](https://img.shields.io/docker/v/neosun/cosyvoice?style=flat-square&logo=docker&sort=semver)](https://hub.docker.com/r/neosun/cosyvoice)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=flat-square)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/neosun100/cosyvoice-docker?style=flat-square&logo=github)](https://github.com/neosun100/cosyvoice-docker)

**Fun-CosyVoice3-0.5B ベースの本番環境対応音声合成サービス**

Docker コマンド一つで Web UI + REST API + 音声クローニング

[クイックスタート](#-クイックスタート) • [機能](#-機能) • [API ドキュメント](#-api-リファレンス) • [パフォーマンス](#-パフォーマンスベンチマーク)

</div>

---

## 📸 スクリーンショット

![Web UI](https://img.aws.xin/uPic/o1Qj12.png)

## ✨ 機能

| 機能 | 説明 |
|------|------|
| 🎯 **Fun-CosyVoice3-0.5B** | Alibaba の最新・最高品質 TTS モデル |
| 🎤 **Fun-ASR-Nano** | 自動音声認識（Whisper の代替） |
| 🔌 **OpenAI 互換 API** | `/v1/audio/speech` の直接置き換え可能 |
| 👤 **カスタム音声管理** | 一度アップロード、ID で呼び出し |
| ⚡ **リアルストリーミング出力** | PCM チャンク出力、~1.2s 初回レイテンシ |
| 🚀 **Embedding キャッシュ** | 初回使用後 53% 高速化 |
| 🌐 **Web UI** | 美しいインターフェース、ダウンロード対応 |
| 🌍 **多言語対応** | 中英日韓 + 18 方言 |

## 🚀 クイックスタート

```bash
docker run -d \
  --name cosyvoice \
  --gpus '"device=0"' \
  -p 8188:8188 \
  -v cosyvoice-data:/data/voices \
  neosun/cosyvoice:latest
```

http://localhost:8188 を開く 🎉

## 📦 インストール

### 前提条件

- Docker 20.10+
- Docker Compose v2.0+（オプション）
- NVIDIA GPU、VRAM 8GB+
- NVIDIA Container Toolkit

### Docker Run

```bash
# イメージをプル
docker pull neosun/cosyvoice:v3.4.0

# コンテナを起動
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

### ヘルスチェック

```bash
curl http://localhost:8188/health
# {"status":"healthy","gpu":{"model_loaded":true,...}}
```

## ⚙️ 設定

### 環境変数

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `PORT` | `8188` | サービスポート |
| `MODEL_DIR` | `pretrained_models/Fun-CosyVoice3-0.5B` | TTS モデルパス |

### ボリューム

| パス | 説明 |
|------|------|
| `/data/voices` | カスタム音声ストレージ（永続化） |

## 📡 API リファレンス

### エンドポイント

| エンドポイント | メソッド | 説明 |
|---------------|---------|------|
| `/v1/audio/speech` | POST | 音声合成（OpenAI 互換） |
| `/v1/voices/create` | POST | カスタム音声作成 |
| `/v1/voices/custom` | GET | カスタム音声一覧 |
| `/v1/voices/{id}` | GET/DELETE | 音声取得/削除 |
| `/v1/models` | GET | モデル一覧 |
| `/health` | GET | ヘルスチェック |
| `/docs` | GET | Swagger UI |

### カスタム音声作成

```bash
# テキスト付き
curl -X POST http://localhost:8188/v1/voices/create \
  -F "audio=@voice.wav" \
  -F "name=マイボイス" \
  -F "text=参照テキスト内容"

# 自動文字起こし（Fun-ASR-Nano 使用）
curl -X POST http://localhost:8188/v1/voices/create \
  -F "audio=@voice.wav" \
  -F "name=マイボイス"

# レスポンス: {"voice_id": "abc123", "text": "自動認識されたテキスト", ...}
```

### 音声合成

```bash
# WAV 形式
curl http://localhost:8188/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "こんにちは世界", "voice": "abc123"}' \
  -o output.wav

# PCM ストリーミング（最低レイテンシ）
curl http://localhost:8188/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "こんにちは世界", "voice": "abc123", "response_format": "pcm"}' \
  -o output.pcm

# PCM を WAV に変換
ffmpeg -f s16le -ar 24000 -ac 1 -i output.pcm output.wav
```

## 📊 パフォーマンスベンチマーク

**テスト環境:** NVIDIA L40S GPU

### 初回トークンレイテンシ (TTFB)

| テキスト長 | TTFB | 合計時間 | 音声長 | RTF |
|-----------|------|---------|--------|-----|
| 短文(4文字) | **1.20s** | 1.55s | 1.88s | 0.82x |
| 短文(10文字) | **1.34s** | 1.75s | 2.28s | 0.77x |
| 中文(30文字) | **1.24s** | 4.98s | 6.88s | 0.72x |
| 中文(50文字) | **1.27s** | 12.52s | 17.12s | 0.73x |
| 長文(80文字) | **1.24s** | 17.91s | 23.68s | 0.76x |
| 長文(120文字) | **1.35s** | 19.08s | 25.32s | 0.75x |

> RTF (リアルタイムファクター) < 1.0 は生成が再生より速いことを意味

### Embedding キャッシュ効果

| シナリオ | TTFB | 備考 |
|---------|------|------|
| 初回使用（キャッシュなし） | ~3.5s | 特徴抽出 + GPU にキャッシュ |
| キャッシュヒット | **~1.2s** | キャッシュから直接読み込み |
| **改善** | **-53%** | |

### ASR (Fun-ASR-Nano) ベンチマーク

| 音声 | 言語 | 長さ | 認識時間 | 認識結果 |
|------|------|------|---------|---------|
| 音声サンプル | 中国語 | ~7s | **0.40s** | 希望你以后能够做的比我还好哟。 |
| 音声サンプル | 中国語 | ~7s | **0.83s** | 对，这就是我万人敬仰的太乙真人... |
| zh.mp3 | 中国語 | ~3s | **0.40s** | 开放时间早上九点至下午五点。 |
| en.mp3 | 英語 | ~5s | **0.70s** | The tribal chieftain called for the boy... |
| ja.mp3 | 日本語 | ~5s | **0.84s** | うちの中学は弁当制で... |

> 平均認識時間: **0.4-0.8s** / 音声ファイル

## 🗣️ 対応言語

### TTS (Fun-CosyVoice3)
- **主要言語**: 中国語、英語、日本語、韓国語
- **ヨーロッパ言語**: ドイツ語、スペイン語、フランス語、イタリア語、ロシア語
- **中国語方言**: 広東語、四川語、東北語、上海語、閩南語など 18+ 種

### ASR (Fun-ASR-Nano)
- **対応言語**: 中国語、英語、日本語 + 自動検出
- **中国語方言**: 7 大方言 + 26 地方アクセント
- **特徴**: 高ノイズ認識、歌詞認識

## 🛠️ 技術スタック

- **TTS モデル:** [Fun-CosyVoice3-0.5B](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)
- **ASR モデル:** [Fun-ASR-Nano-2512](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512)
- **フレームワーク:** FastAPI + Gradio
- **ランタイム:** PyTorch + CUDA
- **コンテナ:** Docker + NVIDIA Container Toolkit

## 📋 変更履歴

| バージョン | 日付 | 変更内容 |
|-----------|------|---------|
| v3.4.0 | 2024-12-18 | Fun-ASR-Nano が Whisper を置き換え |
| v3.3.0 | 2024-12-18 | UI 改善：ストリーミングデフォルト、ダウンロードボタン、タイマー |
| v3.2.1 | 2024-12-18 | 起動時に全音声を自動プリロード |
| v3.2.0 | 2024-12-18 | Embedding キャッシュ（-53% TTFB） |
| v3.1.0 | 2024-12-18 | ポーリング最適化 + モデルプリロード |
| v3.0.0 | 2024-12-18 | All-in-One Docker 基本版 |

## 🤝 コントリビューション

コントリビューション歓迎！お気軽に Pull Request を送ってください。

1. リポジトリをフォーク
2. 機能ブランチを作成 (`git checkout -b feature/amazing`)
3. 変更をコミット (`git commit -m '素晴らしい機能を追加'`)
4. ブランチにプッシュ (`git push origin feature/amazing`)
5. Pull Request を作成

## 📄 ライセンス

このプロジェクトは Apache License 2.0 でライセンスされています - 詳細は [LICENSE](LICENSE) ファイルを参照。

## 🙏 謝辞

- [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice) - オリジナル CosyVoice プロジェクト
- [FunAudioLLM/Fun-ASR](https://github.com/FunAudioLLM/Fun-ASR) - Fun-ASR-Nano モデル

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=neosun100/cosyvoice-docker&type=Date)](https://star-history.com/#neosun100/cosyvoice-docker)

## 📱 フォローする

![WeChat](https://img.aws.xin/uPic/扫码_搜索联合传播样式-标准色版.png)
