"""
CosyVoice All-in-One Service: UI + API + MCP
"""
import os
import sys
import gc
import time
import uuid
import json
import asyncio
import threading
from pathlib import Path
from typing import Optional, Generator
from contextlib import asynccontextmanager

import torch
import torchaudio
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR / "third_party/Matcha-TTS"))

from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.file_utils import load_wav

# Fun-ASR-Nano for auto transcription
_asr_model = None

def get_asr_model():
    global _asr_model
    if _asr_model is None:
        from funasr import AutoModel
        print("Loading Fun-ASR-Nano model...")
        _asr_model = AutoModel(
            model="FunAudioLLM/Fun-ASR-Nano-2512",
            trust_remote_code=True,
            remote_code="./model.py",
            device="cuda:0",
        )
        print("Fun-ASR-Nano loaded!")
    return _asr_model

def transcribe_audio(audio_path: str) -> str:
    """Use Fun-ASR-Nano to transcribe audio file"""
    model = get_asr_model()
    res = model.generate(
        input=[audio_path],
        cache={},
        batch_size=1,
        language="auto",
        itn=True,
    )
    return res[0]["text"].strip() if res else ""

# Directories
INPUT_DIR = Path(os.getenv("INPUT_DIR", "/data/input"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/data/output"))
VOICES_DIR = Path(os.getenv("VOICES_DIR", "/data/voices"))
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VOICES_DIR.mkdir(parents=True, exist_ok=True)

# Voice Manager - 管理自定义音色
class VoiceManager:
    def __init__(self, voices_dir: Path):
        self.voices_dir = voices_dir
        self.index_file = voices_dir / "voices.json"
        self.voices = self._load_index()
    
    def _load_index(self) -> dict:
        if self.index_file.exists():
            return json.loads(self.index_file.read_text())
        return {}
    
    def _save_index(self):
        self.index_file.write_text(json.dumps(self.voices, ensure_ascii=False, indent=2))
    
    def create(self, name: str, text: str, audio_data: bytes) -> str:
        voice_id = uuid.uuid4().hex[:12]
        voice_dir = self.voices_dir / voice_id
        voice_dir.mkdir(exist_ok=True)
        
        audio_path = voice_dir / "prompt.wav"
        audio_path.write_bytes(audio_data)
        
        self.voices[voice_id] = {
            "id": voice_id,
            "name": name,
            "text": text,
            "audio_path": str(audio_path),
            "created_at": int(time.time())
        }
        self._save_index()
        return voice_id
    
    def get(self, voice_id: str) -> Optional[dict]:
        return self.voices.get(voice_id)
    
    def list_all(self) -> list:
        return [{"id": v["id"], "name": v["name"], "text": v["text"], "created_at": v["created_at"]} 
                for v in self.voices.values()]
    
    def delete(self, voice_id: str) -> bool:
        if voice_id not in self.voices:
            return False
        voice_dir = self.voices_dir / voice_id
        if voice_dir.exists():
            import shutil
            shutil.rmtree(voice_dir)
        del self.voices[voice_id]
        self._save_index()
        return True

voice_manager = VoiceManager(VOICES_DIR)

# GPU Manager - 禁用自动卸载，启动时预热
class GPUManager:
    def __init__(self):
        self.model = None
        self.model_dir = None
        self.lock = threading.Lock()
        self.prompt_cache = {}  # 缓存 prompt 特征
        
    def get_model(self, model_dir: str = None):
        with self.lock:
            if model_dir is None:
                model_dir = os.getenv("MODEL_DIR", "pretrained_models/Fun-CosyVoice3-0.5B")
            if self.model is None or self.model_dir != model_dir:
                self._load_model(model_dir)
            return self.model
    
    def _load_model(self, model_dir: str):
        if self.model is not None:
            self.offload()
        print(f"Loading model from {model_dir}...")
        self.model = AutoModel(model_dir=model_dir)
        self.model_dir = model_dir
        print(f"Model loaded successfully!")
    
    def preload(self):
        """启动时预热模型和所有音色的 embedding"""
        print("Preloading model...")
        model = self.get_model()
        print("Model preloaded!")
        
        # 预热所有已保存音色的 embedding
        voices = voice_manager.list_all()
        if voices:
            print(f"Preloading {len(voices)} voice embeddings...")
            for v in voices:
                voice = voice_manager.get(v["id"])
                if voice and os.path.exists(voice["audio_path"]):
                    try:
                        # 调用一次 frontend_zero_shot 触发缓存
                        model.frontend.frontend_zero_shot(
                            "预热", voice["text"], voice["audio_path"], 
                            24000, ""
                        )
                        print(f"  ✓ Cached: {v['name']} ({v['id']})")
                    except Exception as e:
                        print(f"  ✗ Failed: {v['name']} - {e}")
            print(f"Voice embeddings cached: {len(model.frontend.prompt_cache)}")
        
        print("Model preloaded and ready!")
    
    def offload(self):
        """手动卸载模型"""
        if self.model:
            del self.model
            self.model = None
            self.model_dir = None
            self.prompt_cache.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("GPU memory released")
    
    def get_prompt_cache(self, voice_id: str):
        """获取缓存的 prompt 特征"""
        return self.prompt_cache.get(voice_id)
    
    def set_prompt_cache(self, voice_id: str, cache_data: dict):
        """缓存 prompt 特征"""
        self.prompt_cache[voice_id] = cache_data
    
    def status(self) -> dict:
        gpu_info = {"available": torch.cuda.is_available()}
        if torch.cuda.is_available():
            gpu_info.update({
                "device": torch.cuda.get_device_name(0),
                "memory_used": f"{torch.cuda.memory_allocated()/1024**3:.2f} GB",
                "memory_total": f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB",
            })
        # 获取真实的 frontend 缓存数量
        cache_size = len(self.model.frontend.prompt_cache) if self.model else 0
        return {
            "model_loaded": self.model is not None,
            "model_dir": self.model_dir,
            "gpu": gpu_info,
            "prompt_cache_size": cache_size
        }

gpu_manager = GPUManager()

# FastAPI App - 启动时预热模型
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时预热模型
    gpu_manager.preload()
    yield
    # 关闭时不自动卸载（保持模型在显存中）

app = FastAPI(
    title="CosyVoice API",
    description="""
## CosyVoice Text-to-Speech API

基于大语言模型的语音合成服务，支持：
- **零样本克隆** (zero_shot): 使用3-30秒参考音频克隆任意音色
- **跨语种克隆** (cross_lingual): 跨语言语音合成
- **指令控制** (instruct): 方言、情感、语速等控制
- **预训练音色** (sft): 使用内置音色

### 支持语言
中文、英文、日语、韩语、德语、西班牙语、法语、意大利语、俄语，以及18+种中文方言
    """,
    version="3.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Models
class TTSRequest(BaseModel):
    text: str
    mode: str = "zero_shot"  # zero_shot, cross_lingual, instruct, sft
    prompt_text: Optional[str] = ""
    instruct_text: Optional[str] = ""
    spk_id: Optional[str] = ""
    speed: float = 1.0
    stream: bool = False

class TaskStatus(BaseModel):
    task_id: str
    status: str
    progress: float = 0
    output_file: Optional[str] = None
    error: Optional[str] = None

tasks = {}

# Helper
def save_audio(speech: torch.Tensor, sample_rate: int, filename: str) -> str:
    output_path = OUTPUT_DIR / filename
    torchaudio.save(str(output_path), speech, sample_rate)
    return str(output_path)

def generate_audio_stream(model_output, sample_rate: int, cleanup_path: str = None):
    """Generate PCM audio stream with fade-in and DC offset removal"""
    is_first_chunk = True
    dc_offset = 0.0
    alpha = 0.001  # DC offset sliding average coefficient
    
    try:
        for chunk in model_output:
            wav_chunk = chunk['tts_speech'].numpy().flatten()
            
            # Remove DC offset using sliding average
            chunk_mean = np.mean(wav_chunk)
            dc_offset = dc_offset * (1 - alpha) + chunk_mean * alpha
            wav_chunk = wav_chunk - dc_offset
            
            # Apply fade-in to first chunk
            if is_first_chunk:
                fade_len = min(2048, len(wav_chunk))
                fade = np.linspace(0, 1, fade_len)
                wav_chunk[:fade_len] *= fade
                is_first_chunk = False
            
            # Convert to int16 PCM
            audio = (wav_chunk * 32767).astype(np.int16).tobytes()
            yield audio
    finally:
        # Cleanup temp file after streaming completes
        if cleanup_path and Path(cleanup_path).exists():
            Path(cleanup_path).unlink()

# ============== OpenAI-Compatible API ==============

class SpeechRequest(BaseModel):
    model: str = "cosyvoice-v3"
    input: str
    voice: str = "default"
    response_format: str = "wav"  # wav, pcm
    speed: float = 1.0
    instruct: Optional[str] = None  # 指令文本（方言、情感等）

@app.post("/v1/audio/speech")
async def openai_speech(request: SpeechRequest):
    """OpenAI-compatible TTS API"""
    model = gpu_manager.get_model()
    
    # 检查是否是自定义音色
    custom_voice = voice_manager.get(request.voice)
    
    if custom_voice:
        # 使用自定义音色
        prompt_audio = custom_voice["audio_path"]
        # 添加 <|endofprompt|> 前缀修复音频重复问题 (GitHub Issue #967, #1704)
        prompt_text = f'<|endofprompt|>{custom_voice["text"]}'
        
        if request.instruct:
            # instruct 模式
            if hasattr(model, 'inference_instruct2'):
                output = model.inference_instruct2(
                    request.input, request.instruct, prompt_audio,
                    stream=(request.response_format == "pcm"), speed=request.speed
                )
            else:
                output = model.inference_zero_shot(
                    request.input, prompt_text, prompt_audio,
                    stream=(request.response_format == "pcm"), speed=request.speed
                )
        else:
            # zero_shot 模式
            output = model.inference_zero_shot(
                request.input, prompt_text, prompt_audio,
                stream=(request.response_format == "pcm"), speed=request.speed
            )
    else:
        # 使用预训练音色（如果有）
        available_spks = model.list_available_spks()
        if request.voice in available_spks:
            output = model.inference_sft(
                request.input, request.voice,
                stream=(request.response_format == "pcm"), speed=request.speed
            )
        else:
            raise HTTPException(400, f"Voice '{request.voice}' not found. Use /v1/voices to list available voices or create custom voice via /v1/voices/create")
    
    if request.response_format == "pcm":
        return StreamingResponse(
            generate_audio_stream(output, model.sample_rate),
            media_type="audio/pcm",
            headers={"X-Sample-Rate": str(model.sample_rate)}
        )
    
    # 收集所有 chunks 并返回 WAV
    speeches = [chunk['tts_speech'] for chunk in output]
    full_speech = torch.cat(speeches, dim=1)
    filename = f"speech_{uuid.uuid4().hex[:8]}.wav"
    output_path = save_audio(full_speech, model.sample_rate, filename)
    return FileResponse(output_path, media_type="audio/wav", filename=filename)

@app.post("/v1/voices/create")
async def create_voice(
    audio: UploadFile = File(...),
    name: str = Form(...),
    text: str = Form("")
):
    """创建自定义音色"""
    content = await audio.read()
    
    # 保存临时文件用于转写
    temp_path = INPUT_DIR / f"temp_{uuid.uuid4().hex}.wav"
    temp_path.write_bytes(content)
    
    try:
        # 如果没有提供文本，使用 Fun-ASR 转写
        if not text:
            text = transcribe_audio(str(temp_path))
        
        voice_id = voice_manager.create(name, text, content)
        return {
            "success": True,
            "voice_id": voice_id,
            "name": name,
            "text": text,
            "message": f"音色创建成功，使用 voice='{voice_id}' 调用 /v1/audio/speech"
        }
    finally:
        if temp_path.exists():
            temp_path.unlink()

@app.get("/v1/voices/custom")
async def list_custom_voices():
    """列出所有自定义音色"""
    return {"voices": voice_manager.list_all()}

@app.get("/v1/voices/{voice_id}")
async def get_voice(voice_id: str):
    """获取音色详情"""
    voice = voice_manager.get(voice_id)
    if not voice:
        raise HTTPException(404, "Voice not found")
    return voice

@app.delete("/v1/voices/{voice_id}")
async def delete_voice(voice_id: str):
    """删除自定义音色"""
    if voice_manager.delete(voice_id):
        return {"success": True, "message": "Voice deleted"}
    raise HTTPException(404, "Voice not found")

@app.get("/v1/voices")
async def list_voices():
    """列出所有可用音色（预训练 + 自定义）"""
    model = gpu_manager.get_model()
    preset_voices = model.list_available_spks()
    custom_voices = voice_manager.list_all()
    return {
        "preset_voices": preset_voices,
        "custom_voices": custom_voices
    }

@app.get("/v1/models")
async def list_models():
    """列出可用模型"""
    return {
        "models": [
            {"id": "cosyvoice-v3", "name": "Fun-CosyVoice3-0.5B", "description": "最新版本，效果最好"},
            {"id": "cosyvoice-v2", "name": "CosyVoice2-0.5B", "description": "稳定版本"},
        ]
    }

# ============== Legacy API ==============

@app.get("/health")
async def health():
    return {"status": "healthy", "gpu": gpu_manager.status()}

@app.get("/api/status")
async def api_status():
    return gpu_manager.status()

@app.post("/api/offload")
async def offload_gpu():
    gpu_manager.offload()
    return {"status": "success", "message": "GPU memory released"}

@app.get("/api/speakers")
async def list_speakers():
    model = gpu_manager.get_model()
    return {"speakers": model.list_available_spks()}

@app.post("/api/tts")
async def tts(
    voice: str = Form(""),
    text: str = Form(...),
    mode: str = Form("zero_shot"),
    prompt_text: str = Form(""),
    instruct_text: str = Form(""),
    spk_id: str = Form(""),
    speed: float = Form(1.0),
    stream: bool = Form(False),
    prompt_wav: Optional[UploadFile] = File(None)
):
    model = gpu_manager.get_model()
    prompt_audio = None
    
    # 优先使用自定义音色
    custom_voice = voice_manager.get(voice) if voice else None
    if custom_voice:
        prompt_audio = custom_voice["audio_path"]
        # 添加 <|endofprompt|> 前缀修复音频重复问题
        prompt_text = f'<|endofprompt|>{custom_voice["text"]}'
    elif prompt_wav:
        content = await prompt_wav.read()
        temp_path = INPUT_DIR / f"prompt_{uuid.uuid4().hex}.wav"
        temp_path.write_bytes(content)
        prompt_audio = str(temp_path)
    
    try:
        # 参数验证
        if mode == "zero_shot":
            if not prompt_audio:
                raise HTTPException(400, "zero_shot mode requires prompt_wav or voice (custom voice ID)")
            # 自动识别 prompt_text (仅当未使用自定义音色且没有提供 prompt_text 时)
            if not prompt_text and not custom_voice:
                print("Auto transcribing prompt audio with Fun-ASR...")
                prompt_text = transcribe_audio(prompt_audio)
                # 添加前缀修复重复问题
                prompt_text = f'<|endofprompt|>{prompt_text}'
                print(f"Transcribed: {prompt_text}")
        elif mode == "cross_lingual":
            if not prompt_audio:
                raise HTTPException(400, "cross_lingual mode requires prompt_wav or voice (custom voice ID)")
        elif mode == "instruct":
            if not prompt_audio:
                raise HTTPException(400, "instruct mode requires prompt_wav or voice (custom voice ID)")
            if not instruct_text:
                raise HTTPException(400, "instruct mode requires instruct_text")
        elif mode == "sft":
            if not spk_id:
                raise HTTPException(400, "sft mode requires spk_id (speaker ID)")
        
        if mode == "sft":
            output = model.inference_sft(text, spk_id, stream=stream, speed=speed)
        elif mode == "zero_shot":
            output = model.inference_zero_shot(text, prompt_text, prompt_audio, stream=stream, speed=speed)
        elif mode == "cross_lingual":
            output = model.inference_cross_lingual(text, prompt_audio, stream=stream, speed=speed)
        elif mode == "instruct":
            if hasattr(model, 'inference_instruct2'):
                output = model.inference_instruct2(text, instruct_text, prompt_audio, stream=stream, speed=speed)
            else:
                output = model.inference_instruct(text, spk_id, instruct_text, stream=stream, speed=speed)
        else:
            raise HTTPException(400, f"Unknown mode: {mode}")
        
        if stream:
            return StreamingResponse(
                generate_audio_stream(output, model.sample_rate, cleanup_path=prompt_audio),
                media_type="audio/pcm"
            )
        
        # Collect all chunks
        speeches = []
        for chunk in output:
            speeches.append(chunk['tts_speech'])
        
        full_speech = torch.cat(speeches, dim=1)
        filename = f"tts_{uuid.uuid4().hex}.wav"
        output_path = save_audio(full_speech, model.sample_rate, filename)
        
        # Cleanup temp file for non-streaming mode
        if prompt_audio and Path(prompt_audio).exists():
            Path(prompt_audio).unlink()
        
        return FileResponse(output_path, media_type="audio/wav", filename=filename)
    
    except Exception as e:
        # Cleanup on error
        if prompt_audio and Path(prompt_audio).exists():
            Path(prompt_audio).unlink()
        raise

@app.post("/api/tts/async")
async def tts_async(
    background_tasks: BackgroundTasks,
    text: str = Form(...),
    mode: str = Form("zero_shot"),
    prompt_text: str = Form(""),
    instruct_text: str = Form(""),
    spk_id: str = Form(""),
    speed: float = Form(1.0),
    prompt_wav: Optional[UploadFile] = File(None)
):
    task_id = uuid.uuid4().hex
    tasks[task_id] = {"status": "pending", "progress": 0}
    
    prompt_path = None
    if prompt_wav:
        content = await prompt_wav.read()
        prompt_path = INPUT_DIR / f"prompt_{task_id}.wav"
        prompt_path.write_bytes(content)
    
    def process():
        try:
            tasks[task_id]["status"] = "processing"
            model = gpu_manager.get_model()
            
            if mode == "sft":
                output = model.inference_sft(text, spk_id, stream=False, speed=speed)
            elif mode == "zero_shot":
                output = model.inference_zero_shot(text, prompt_text, str(prompt_path) if prompt_path else None, stream=False, speed=speed)
            elif mode == "cross_lingual":
                output = model.inference_cross_lingual(text, str(prompt_path) if prompt_path else None, stream=False, speed=speed)
            elif mode == "instruct":
                if hasattr(model, 'inference_instruct2'):
                    output = model.inference_instruct2(text, instruct_text, str(prompt_path) if prompt_path else None, stream=False, speed=speed)
                else:
                    output = model.inference_instruct(text, spk_id, instruct_text, stream=False, speed=speed)
            
            speeches = [chunk['tts_speech'] for chunk in output]
            full_speech = torch.cat(speeches, dim=1)
            filename = f"tts_{task_id}.wav"
            save_audio(full_speech, model.sample_rate, filename)
            
            tasks[task_id] = {"status": "completed", "progress": 100, "output_file": filename}
        except Exception as e:
            tasks[task_id] = {"status": "failed", "error": str(e)}
        finally:
            if prompt_path and prompt_path.exists():
                prompt_path.unlink()
    
    background_tasks.add_task(process)
    return {"task_id": task_id}

@app.get("/api/task/{task_id}")
async def get_task(task_id: str):
    if task_id not in tasks:
        raise HTTPException(404, "Task not found")
    return tasks[task_id]

@app.get("/api/download/{filename}")
async def download(filename: str):
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(str(path), media_type="audio/wav", filename=filename)

# UI
@app.get("/", response_class=HTMLResponse)
async def ui():
    return HTML_TEMPLATE


HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CosyVoice3 TTS</title>
    <style>
        :root {
            --bg: #0a0a0f; --card: #12121a; --card-hover: #1a1a25;
            --primary: #6366f1; --primary-glow: rgba(99,102,241,0.3);
            --accent: #22d3ee; --accent-glow: rgba(34,211,238,0.2);
            --success: #10b981; --warning: #f59e0b; --danger: #ef4444;
            --text: #f1f5f9; --text-muted: #94a3b8; --border: #1e293b;
            --gradient: linear-gradient(135deg, #6366f1, #22d3ee);
        }
        [data-theme="light"] {
            --bg: #f8fafc; --card: #ffffff; --card-hover: #f1f5f9;
            --primary: #4f46e5; --primary-glow: rgba(79,70,229,0.2);
            --accent: #0891b2; --accent-glow: rgba(8,145,178,0.15);
            --text: #1e293b; --text-muted: #64748b; --border: #e2e8f0;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { 
            font-family: 'Inter', -apple-system, sans-serif; 
            background: var(--bg); color: var(--text); min-height: 100vh;
            background-image: radial-gradient(ellipse at top, rgba(99,102,241,0.1) 0%, transparent 50%),
                              radial-gradient(ellipse at bottom right, rgba(34,211,238,0.05) 0%, transparent 50%);
        }
        .container { max-width: 1000px; margin: 0 auto; padding: 30px 20px; }
        header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px; padding-bottom: 20px; border-bottom: 1px solid var(--border); }
        .logo { display: flex; align-items: center; gap: 12px; }
        .logo-icon { width: 48px; height: 48px; border-radius: 12px; background: var(--gradient); display: flex; align-items: center; justify-content: center; font-size: 24px; box-shadow: 0 4px 20px var(--primary-glow); }
        .logo h1 { font-size: 1.5em; font-weight: 700; background: var(--gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .logo span { font-size: 0.75em; color: var(--text-muted); }
        .header-controls { display: flex; gap: 8px; }
        .header-controls select, .header-controls button { padding: 8px 14px; border: 1px solid var(--border); border-radius: 8px; background: var(--card); color: var(--text); cursor: pointer; }
        .card { background: var(--card); border-radius: 16px; padding: 24px; margin-bottom: 20px; border: 1px solid var(--border); position: relative; overflow: hidden; }
        .card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px; background: linear-gradient(90deg, transparent, var(--primary), transparent); opacity: 0; transition: opacity 0.3s; }
        .card:hover::before { opacity: 1; }
        .card-title { font-size: 0.85em; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 16px; display: flex; align-items: center; gap: 8px; }
        .card-title::before { content: ''; width: 3px; height: 14px; background: var(--gradient); border-radius: 2px; }
        textarea, input[type="text"], input[type="number"], select { width: 100%; padding: 14px 16px; border: 1px solid var(--border); border-radius: 10px; background: var(--bg); color: var(--text); font-size: 1em; outline: none; }
        textarea { min-height: 120px; resize: vertical; line-height: 1.6; }
        textarea:focus, input:focus, select:focus { border-color: var(--primary); box-shadow: 0 0 0 3px var(--primary-glow); }
        label { display: block; margin-bottom: 8px; color: var(--text-muted); font-size: 0.9em; font-weight: 500; }
        .form-group { margin-bottom: 18px; }
        .tabs { display: flex; gap: 6px; margin-bottom: 20px; background: var(--bg); padding: 4px; border-radius: 10px; }
        .tab { flex: 1; padding: 12px 16px; border-radius: 8px; cursor: pointer; text-align: center; font-size: 0.9em; font-weight: 500; color: var(--text-muted); border: none; background: transparent; }
        .tab:hover { color: var(--text); }
        .tab.active { background: var(--primary); color: white; box-shadow: 0 2px 10px var(--primary-glow); }
        .voice-row { display: flex; gap: 10px; align-items: center; }
        .voice-row select { flex: 1; }
        .icon-btn { width: 42px; height: 42px; border-radius: 10px; border: 1px solid var(--border); background: var(--card); color: var(--text); cursor: pointer; font-size: 1.1em; display: flex; align-items: center; justify-content: center; }
        .icon-btn:hover { border-color: var(--primary); }
        .icon-btn.danger:hover { border-color: var(--danger); color: var(--danger); }
        .upload-area { border: 2px dashed var(--border); border-radius: 12px; padding: 30px; text-align: center; cursor: pointer; }
        .upload-area:hover, .upload-area.dragover { border-color: var(--primary); background: rgba(99,102,241,0.05); }
        .upload-icon { font-size: 2.5em; margin-bottom: 10px; }
        .upload-text { color: var(--text-muted); }
        .upload-file { color: var(--accent); margin-top: 10px; font-weight: 500; }
        .btn-generate { width: 100%; padding: 16px 32px; border: none; border-radius: 12px; background: var(--gradient); color: white; font-size: 1.1em; font-weight: 600; cursor: pointer; box-shadow: 0 4px 20px var(--primary-glow); }
        .btn-generate:hover { transform: translateY(-2px); box-shadow: 0 6px 30px var(--primary-glow); }
        .btn-generate:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }
        .stats-panel { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 20px; padding: 16px; background: var(--bg); border-radius: 12px; }
        .stat-item { text-align: center; padding: 16px 12px; background: var(--card); border-radius: 10px; border: 1px solid var(--border); transition: all 0.3s; }
        .stat-item.highlight { border-color: var(--accent); box-shadow: 0 0 20px var(--accent-glow); }
        .stat-label { font-size: 0.75em; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }
        .stat-value { font-size: 1.4em; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
        .stat-value.primary { color: var(--primary); }
        .stat-value.accent { color: var(--accent); }
        .stat-value.success { color: var(--success); }
        .stat-value.warning { color: var(--warning); }
        .progress-container { margin-top: 16px; }
        .progress-bar { height: 6px; background: var(--border); border-radius: 3px; overflow: hidden; }
        .progress-fill { height: 100%; background: var(--gradient); width: 0%; transition: width 0.3s; }
        .progress-text { margin-top: 10px; text-align: center; font-size: 0.9em; color: var(--text-muted); display: flex; align-items: center; justify-content: center; gap: 8px; }
        .progress-text .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--success); animation: pulse 1s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        .audio-container { margin-top: 20px; display: none; }
        .audio-container.show { display: block; }
        .audio-player { width: 100%; padding: 20px; background: var(--bg); border-radius: 12px; border: 1px solid var(--border); }
        .audio-player audio { width: 100%; height: 48px; }
        .audio-actions { display: flex; gap: 10px; margin-top: 12px; }
        .btn-download { flex: 1; padding: 12px 20px; border: 1px solid var(--accent); border-radius: 10px; background: transparent; color: var(--accent); font-weight: 500; cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 8px; }
        .btn-download:hover { background: var(--accent); color: var(--bg); }
        .gpu-bar { display: flex; justify-content: space-between; align-items: center; padding: 14px 18px; background: var(--bg); border-radius: 10px; border: 1px solid var(--border); }
        .gpu-info { display: flex; align-items: center; gap: 10px; font-size: 0.9em; }
        .gpu-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--success); }
        .btn-release { padding: 8px 16px; border: 1px solid var(--border); border-radius: 8px; background: transparent; color: var(--text-muted); font-size: 0.85em; cursor: pointer; }
        .btn-release:hover { border-color: var(--warning); color: var(--warning); }
        .checkbox-label { display: flex; align-items: center; gap: 10px; cursor: pointer; font-size: 0.9em; }
        .checkbox-label input { width: 18px; height: 18px; accent-color: var(--primary); }
        .row { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; }
        .hidden { display: none !important; }
        @media (max-width: 640px) { .stats-panel { grid-template-columns: repeat(2, 1fr); } .row { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo">
                <div class="logo-icon">🎙️</div>
                <div><h1>CosyVoice3</h1><span>AI Text-to-Speech</span></div>
            </div>
            <div class="header-controls">
                <select id="lang" onchange="setLang(this.value)"><option value="zh-CN">简体中文</option><option value="en">English</option></select>
                <button onclick="window.open('https://github.com/neosun100/cosyvoice-docker', '_blank')" title="GitHub">⭐</button>
                <button onclick="window.open('/docs', '_blank')" title="API 文档">📄</button>
                <button onclick="toggleTheme()" title="切换主题">🌓</button>
            </div>
        </header>
        <div class="card">
            <div class="card-title">输入文本</div>
            <textarea id="text" placeholder="请输入要合成的文本...">收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐。</textarea>
        </div>
        <div class="card">
            <div class="card-title">合成模式</div>
            <div class="tabs">
                <button class="tab active" data-mode="zero_shot">零样本克隆</button>
                <button class="tab" data-mode="cross_lingual">跨语种</button>
                <button class="tab" data-mode="instruct">指令控制</button>
            </div>
            <div id="prompt-section">
                <div class="form-group">
                    <label>选择音色</label>
                    <div class="voice-row">
                        <select id="voice-select" onchange="onVoiceSelect()"><option value="">-- 上传新音频 --</option></select>
                        <button class="icon-btn" onclick="refreshVoices()" title="刷新">🔄</button>
                        <button class="icon-btn danger" onclick="deleteSelectedVoice()" title="删除">🗑️</button>
                    </div>
                </div>
                <div id="upload-section">
                    <div class="form-group">
                        <label>参考音频 (3-30秒)</label>
                        <div class="upload-area" id="upload-area">
                            <input type="file" id="prompt-file" accept="audio/*" class="hidden">
                            <div class="upload-icon">📁</div>
                            <div class="upload-text">点击或拖拽上传音频文件</div>
                            <div class="upload-file" id="file-name"></div>
                        </div>
                    </div>
                    <div class="row">
                        <div class="form-group"><label>音色名称</label><input type="text" id="voice-name" placeholder="如：张三的声音"></div>
                        <div class="form-group" id="prompt-text-group"><label>参考文本 (留空自动识别)</label><input type="text" id="prompt-text" placeholder="留空将使用 ASR 自动识别"></div>
                    </div>
                    <label class="checkbox-label"><input type="checkbox" id="save-voice" checked><span>保存为自定义音色</span></label>
                </div>
            </div>
            <div id="instruct-section" class="hidden">
                <div class="form-group"><label>指令文本</label><input type="text" id="instruct-text" placeholder="用四川话说这句话"></div>
            </div>
            <div class="row" style="margin-top: 16px;">
                <div class="form-group" style="margin-bottom: 0;"><label>语速</label><input type="number" id="speed" value="1.0" min="0.5" max="2.0" step="0.1"></div>
                <div class="form-group" style="margin-bottom: 0; display: flex; align-items: flex-end;">
                    <label class="checkbox-label" style="margin-bottom: 0; padding: 14px 0;"><input type="checkbox" id="stream-mode" checked><span>流式输出 (低延迟)</span></label>
                </div>
            </div>
        </div>
        <div class="card">
            <button class="btn-generate" id="generate-btn" onclick="generate()"><span id="btn-text">🚀 生成语音</span></button>
            <div class="progress-container hidden" id="progress-container">
                <div class="progress-bar"><div class="progress-fill" id="progress"></div></div>
                <div class="progress-text"><span class="dot"></span><span id="progress-status">正在生成...</span></div>
            </div>
            <div class="stats-panel hidden" id="stats-panel">
                <div class="stat-item" id="stat-ttfb"><div class="stat-label">首字节延迟</div><div class="stat-value primary" id="ttfb-value">--</div></div>
                <div class="stat-item" id="stat-start"><div class="stat-label">开始播放</div><div class="stat-value accent" id="start-value">--</div></div>
                <div class="stat-item" id="stat-total"><div class="stat-label">总时间</div><div class="stat-value success" id="total-value">--</div></div>
                <div class="stat-item" id="stat-size"><div class="stat-label">数据大小</div><div class="stat-value warning" id="size-value">--</div></div>
            </div>
            <div class="audio-container" id="audio-container">
                <div class="audio-player">
                    <audio id="audio-output" controls></audio>
                    <div class="audio-actions"><button class="btn-download" id="download-btn">📥 下载音频</button></div>
                </div>
            </div>
        </div>
        <div class="card">
            <div class="gpu-bar">
                <div class="gpu-info"><span class="gpu-dot"></span><span id="gpu-info">GPU: 加载中...</span></div>
                <button class="btn-release" onclick="offloadGPU()">释放显存</button>
            </div>
        </div>
    </div>
    <script>
        let currentLang = 'zh-CN', currentMode = 'zero_shot', promptFile = null, selectedVoiceId = null;
        let startTime = 0;

        function setLang(lang) { currentLang = lang; refreshVoices(); }
        function toggleTheme() { document.body.dataset.theme = document.body.dataset.theme === 'light' ? '' : 'light'; }

        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', () => {
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                currentMode = tab.dataset.mode;
                document.getElementById('instruct-section').classList.toggle('hidden', currentMode !== 'instruct');
                document.getElementById('prompt-text-group').classList.toggle('hidden', currentMode === 'cross_lingual');
            });
        });

        async function refreshVoices() {
            try {
                const res = await fetch('/v1/voices/custom');
                const data = await res.json();
                const select = document.getElementById('voice-select');
                select.innerHTML = '<option value="">-- 上传新音频 --</option>';
                data.voices.forEach(v => { select.innerHTML += `<option value="${v.id}">${v.name}</option>`; });
            } catch (e) { console.error(e); }
        }

        function onVoiceSelect() {
            selectedVoiceId = document.getElementById('voice-select').value;
            document.getElementById('upload-section').classList.toggle('hidden', !!selectedVoiceId);
        }

        async function deleteSelectedVoice() {
            if (!selectedVoiceId) return alert('请先选择要删除的音色');
            if (!confirm('确定删除此音色？')) return;
            await fetch(`/v1/voices/${selectedVoiceId}`, { method: 'DELETE' });
            selectedVoiceId = null; refreshVoices();
        }

        const uploadArea = document.getElementById('upload-area');
        const fileInput = document.getElementById('prompt-file');
        uploadArea.onclick = () => fileInput.click();
        fileInput.onchange = e => { promptFile = e.target.files[0]; document.getElementById('file-name').textContent = promptFile ? `📎 ${promptFile.name}` : ''; };
        uploadArea.ondragover = e => { e.preventDefault(); uploadArea.classList.add('dragover'); };
        uploadArea.ondragleave = () => uploadArea.classList.remove('dragover');
        uploadArea.ondrop = e => { e.preventDefault(); uploadArea.classList.remove('dragover'); promptFile = e.dataTransfer.files[0]; document.getElementById('file-name').textContent = promptFile ? `📎 ${promptFile.name}` : ''; };

        async function generate() {
            const text = document.getElementById('text').value.trim();
            if (!text) return alert('请输入文本');
            const stream = document.getElementById('stream-mode').checked;
            const speed = parseFloat(document.getElementById('speed').value) || 1.0;
            const btn = document.getElementById('generate-btn');
            const btnText = document.getElementById('btn-text');

            // 立即重置所有状态
            btn.disabled = true; btnText.textContent = '生成中...';
            document.getElementById('progress-container').classList.remove('hidden');
            document.getElementById('stats-panel').classList.remove('hidden');
            document.getElementById('audio-container').classList.remove('show');
            document.getElementById('progress').style.width = '0%';
            document.getElementById('progress-status').innerHTML = '<span class="dot"></span> 正在生成...';
            document.getElementById('audio-output').src = '';
            ['ttfb', 'start', 'total', 'size'].forEach(id => {
                document.getElementById(id + '-value').textContent = '--';
                document.getElementById('stat-' + id).classList.remove('highlight');
            });

            startTime = performance.now();
            let totalBytes = 0, firstChunk = true, ttfbTime = 0, audioStartTime = 0;

            try {
                const formData = new FormData();
                formData.append('text', text);
                formData.append('mode', currentMode);
                formData.append('speed', speed);
                formData.append('stream', stream ? '1' : '0');

                if (selectedVoiceId) {
                    formData.append('voice', selectedVoiceId);
                } else if (promptFile) {
                    formData.append('prompt_audio', promptFile);
                    formData.append('prompt_text', document.getElementById('prompt-text').value);
                    if (document.getElementById('save-voice').checked) formData.append('voice_name', document.getElementById('voice-name').value || '未命名');
                } else { alert('请选择音色或上传参考音频'); btn.disabled = false; btnText.textContent = '🚀 生成语音'; return; }

                if (currentMode === 'instruct') formData.append('instruct', document.getElementById('instruct-text').value);

                if (stream) {
                    const res = await fetch('/api/tts', { method: 'POST', body: formData });
                    if (!res.ok) throw new Error(await res.text());
                    const reader = res.body.getReader();
                    const chunks = [];
                    const audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 24000 });
                    let nextStartTime = audioCtx.currentTime, playbackStarted = false;

                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;
                        if (firstChunk) {
                            ttfbTime = performance.now() - startTime;
                            document.getElementById('ttfb-value').textContent = (ttfbTime / 1000).toFixed(3) + 's';
                            document.getElementById('stat-ttfb').classList.add('highlight');
                            firstChunk = false;
                        }
                        chunks.push(value); totalBytes += value.length;
                        document.getElementById('size-value').textContent = (totalBytes / 1024).toFixed(0) + ' KB';
                        // 实时更新总时间
                        document.getElementById('total-value').textContent = ((performance.now() - startTime) / 1000).toFixed(2) + 's';

                        const int16 = new Int16Array(value.buffer);
                        const float32 = new Float32Array(int16.length);
                        for (let i = 0; i < int16.length; i++) float32[i] = int16[i] / 32768;
                        const buffer = audioCtx.createBuffer(1, float32.length, 24000);
                        buffer.getChannelData(0).set(float32);
                        const source = audioCtx.createBufferSource();
                        source.buffer = buffer; source.connect(audioCtx.destination);
                        if (nextStartTime < audioCtx.currentTime) nextStartTime = audioCtx.currentTime;
                        source.start(nextStartTime); nextStartTime += buffer.duration;

                        if (!playbackStarted) {
                            audioStartTime = performance.now() - startTime;
                            document.getElementById('start-value').textContent = (audioStartTime / 1000).toFixed(3) + 's';
                            document.getElementById('stat-start').classList.add('highlight');
                            playbackStarted = true;
                        }
                        document.getElementById('progress').style.width = Math.min(95, (totalBytes / (text.length * 500)) * 100) + '%';
                    }

                    const totalLength = chunks.reduce((acc, c) => acc + c.length, 0);
                    const allPcm = new Uint8Array(totalLength);
                    let offset = 0; for (const chunk of chunks) { allPcm.set(chunk, offset); offset += chunk.length; }
                    const wavBlob = createWav(allPcm, 24000);
                    const url = URL.createObjectURL(wavBlob);
                    document.getElementById('audio-output').src = url;
                    document.getElementById('download-btn').onclick = () => { const a = document.createElement('a'); a.href = url; a.download = 'cosyvoice_' + Date.now() + '.wav'; a.click(); };
                } else {
                    const res = await fetch('/api/tts', { method: 'POST', body: formData });
                    if (!res.ok) throw new Error(await res.text());
                    ttfbTime = performance.now() - startTime;
                    document.getElementById('ttfb-value').textContent = (ttfbTime / 1000).toFixed(3) + 's';
                    const blob = await res.blob(); totalBytes = blob.size;
                    document.getElementById('size-value').textContent = (totalBytes / 1024).toFixed(0) + ' KB';
                    const url = URL.createObjectURL(blob);
                    document.getElementById('audio-output').src = url;
                    audioStartTime = performance.now() - startTime;
                    document.getElementById('start-value').textContent = (audioStartTime / 1000).toFixed(3) + 's';
                    document.getElementById('download-btn').onclick = () => { const a = document.createElement('a'); a.href = url; a.download = 'cosyvoice_' + Date.now() + '.wav'; a.click(); };
                }

                const totalTime = performance.now() - startTime;
                document.getElementById('total-value').textContent = (totalTime / 1000).toFixed(2) + 's';
                document.getElementById('stat-total').classList.add('highlight');
                document.getElementById('stat-size').classList.add('highlight');
                document.getElementById('progress').style.width = '100%';
                document.getElementById('progress-status').innerHTML = '<span style="color: var(--success);">✅ PCM 完成！</span> ' + (totalBytes / 1024).toFixed(0) + 'KB, 约' + (totalBytes / 24000 / 2).toFixed(1) + '秒';
                document.getElementById('audio-container').classList.add('show');
                refreshVoices();
            } catch (e) { alert('生成失败: ' + e.message); console.error(e); }
            finally { btn.disabled = false; btnText.textContent = '🚀 生成语音'; }
        }

        function createWav(pcmData, sampleRate) {
            const buffer = new ArrayBuffer(44 + pcmData.length);
            const view = new DataView(buffer);
            const writeString = (o, s) => { for (let i = 0; i < s.length; i++) view.setUint8(o + i, s.charCodeAt(i)); };
            writeString(0, 'RIFF'); view.setUint32(4, 36 + pcmData.length, true); writeString(8, 'WAVE'); writeString(12, 'fmt ');
            view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true); view.setUint32(24, sampleRate, true);
            view.setUint32(28, sampleRate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
            writeString(36, 'data'); view.setUint32(40, pcmData.length, true); new Uint8Array(buffer, 44).set(pcmData);
            return new Blob([buffer], { type: 'audio/wav' });
        }

        async function offloadGPU() { await fetch('/api/offload', { method: 'POST' }); updateStatus(); }
        async function updateStatus() {
            try {
                const res = await fetch('/api/status');
                const data = await res.json();
                document.getElementById('gpu-info').textContent = data.model_loaded ? `Model: ${data.model_dir} | GPU: ${data.gpu.memory_used}` : 'Model not loaded';
            } catch (e) {}
        }

        updateStatus(); refreshVoices(); setInterval(updateStatus, 30000);
    </script>
</body>
</html>
'''

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8188"))
    uvicorn.run(app, host="0.0.0.0", port=port)
