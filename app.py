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

# Multi-Precision Model Manager
class MultiPrecisionModelManager:
    """管理多精度模型的加载、卸载和切换"""
    
    SUPPORTED_PRECISIONS = ["fp16", "int8", "int4"]
    
    def __init__(self):
        self.models = {}  # {precision: model}
        self.model_dir = None
        self.current_precision = "fp16"
        self.lock = threading.Lock()
        self.frontend = None  # 共享的 frontend
        
    def _load_single_model(self, precision: str):
        """加载单个精度的模型"""
        if precision not in self.SUPPORTED_PRECISIONS:
            raise ValueError(f"Unsupported precision: {precision}. Supported: {self.SUPPORTED_PRECISIONS}")
        
        model_dir = self.model_dir or os.getenv("MODEL_DIR", "pretrained_models/Fun-CosyVoice3-0.5B")
        
        print(f"Loading {precision} model from {model_dir}...")
        start_time = time.time()
        
        # 使用修改后的 AutoModel 支持 precision 参数
        from cosyvoice.cli.cosyvoice import CosyVoice3
        model = CosyVoice3(model_dir=model_dir, precision=precision)
        
        load_time = time.time() - start_time
        print(f"✓ {precision} model loaded in {load_time:.2f}s")
        
        # 共享 frontend（只需要一个，包含 prompt_cache）
        if self.frontend is None:
            self.frontend = model.frontend
        else:
            # 替换新模型的 frontend 为共享的 frontend
            # 这样所有模型共享同一个 prompt_cache
            model.frontend = self.frontend
        
        return model
    
    def load_model(self, precision: str = "fp16"):
        """加载指定精度的模型到 GPU"""
        with self.lock:
            if precision in self.models:
                print(f"{precision} model already loaded")
                return True
            
            try:
                self.models[precision] = self._load_single_model(precision)
                return True
            except Exception as e:
                print(f"Failed to load {precision} model: {e}")
                return False
    
    def unload_model(self, precision: str):
        """卸载指定精度的模型"""
        with self.lock:
            if precision not in self.models:
                print(f"{precision} model not loaded")
                return False
            
            del self.models[precision]
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"✓ {precision} model unloaded")
            
            # 如果卸载的是当前使用的模型，切换到其他可用模型
            if precision == self.current_precision and self.models:
                self.current_precision = list(self.models.keys())[0]
                print(f"Switched to {self.current_precision}")
            
            return True
    
    def get_model(self, precision: str = None):
        """获取指定精度的模型，如果未指定则使用当前精度"""
        with self.lock:
            if precision is None:
                precision = self.current_precision
            
            if precision not in self.models:
                # 自动加载
                self.models[precision] = self._load_single_model(precision)
            
            return self.models[precision]
    
    def set_current_precision(self, precision: str):
        """设置当前使用的精度"""
        if precision not in self.SUPPORTED_PRECISIONS:
            raise ValueError(f"Unsupported precision: {precision}")
        self.current_precision = precision
    
    def preload(self, precisions: list = None):
        """预加载指定精度的模型"""
        if precisions is None:
            precisions = [os.getenv("DEFAULT_PRECISION", "fp16")]
        
        self.model_dir = os.getenv("MODEL_DIR", "pretrained_models/Fun-CosyVoice3-0.5B")
        
        for precision in precisions:
            self.load_model(precision)
        
        # 预热所有已保存音色的 embedding
        if self.frontend:
            voices = voice_manager.list_all()
            if voices:
                print(f"Preloading {len(voices)} voice embeddings...")
                model = self.get_model()
                for v in voices:
                    voice = voice_manager.get(v["id"])
                    if voice and os.path.exists(voice["audio_path"]):
                        try:
                            model.frontend.frontend_zero_shot(
                                "预热", voice["text"], voice["audio_path"], 
                                24000, ""
                            )
                            print(f"  ✓ Cached: {v['name']} ({v['id']})")
                        except Exception as e:
                            print(f"  ✗ Failed: {v['name']} - {e}")
                print(f"Voice embeddings cached: {len(model.frontend.prompt_cache)}")
        
        print("Model preloaded and ready!")
    
    def status(self) -> dict:
        """获取模型状态"""
        gpu_info = {"available": torch.cuda.is_available()}
        if torch.cuda.is_available():
            gpu_info.update({
                "device": torch.cuda.get_device_name(0),
                "memory_used": f"{torch.cuda.memory_allocated()/1024**3:.2f} GB",
                "memory_total": f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB",
            })
        
        loaded_models = list(self.models.keys())
        cache_size = len(self.frontend.prompt_cache) if self.frontend else 0
        
        return {
            "loaded_models": loaded_models,
            "current_precision": self.current_precision,
            "model_dir": self.model_dir,
            "gpu": gpu_info,
            "prompt_cache_size": cache_size,
            "supported_precisions": self.SUPPORTED_PRECISIONS,
        }

# 兼容旧的 GPUManager 接口
class GPUManager:
    def __init__(self):
        self.manager = MultiPrecisionModelManager()
        
    def get_model(self, model_dir: str = None, precision: str = None):
        if model_dir:
            self.manager.model_dir = model_dir
        return self.manager.get_model(precision)
    
    def preload(self):
        self.manager.preload()
    
    def offload(self):
        for precision in list(self.manager.models.keys()):
            self.manager.unload_model(precision)
    
    def status(self) -> dict:
        return self.manager.status()

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
    precision: Optional[str] = None  # 模型精度: fp16, int8, int4

@app.post("/v1/audio/speech")
async def openai_speech(request: SpeechRequest):
    """OpenAI-compatible TTS API"""
    model = gpu_manager.get_model(precision=request.precision)
    
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
    """列出可用模型和精度选项"""
    status = gpu_manager.status()
    return {
        "models": [
            {"id": "cosyvoice-v3", "name": "Fun-CosyVoice3-0.5B", "description": "最新版本，效果最好"},
        ],
        "precisions": {
            "supported": status.get("supported_precisions", ["fp16", "int8", "int4"]),
            "loaded": status.get("loaded_models", []),
            "current": status.get("current_precision", "fp16"),
        },
        "gpu": status.get("gpu", {}),
    }

@app.post("/v1/models/load")
async def load_model(precision: str = "fp16"):
    """加载指定精度的模型到 GPU"""
    if precision not in ["fp16", "int8", "int4"]:
        raise HTTPException(400, f"Unsupported precision: {precision}. Supported: fp16, int8, int4")
    
    success = gpu_manager.manager.load_model(precision)
    if success:
        return {"status": "success", "message": f"{precision} model loaded", "models": gpu_manager.status()}
    else:
        raise HTTPException(500, f"Failed to load {precision} model")

@app.post("/v1/models/unload")
async def unload_model(precision: str):
    """卸载指定精度的模型"""
    if precision not in ["fp16", "int8", "int4"]:
        raise HTTPException(400, f"Unsupported precision: {precision}")
    
    success = gpu_manager.manager.unload_model(precision)
    if success:
        return {"status": "success", "message": f"{precision} model unloaded", "models": gpu_manager.status()}
    else:
        raise HTTPException(404, f"{precision} model not loaded")

@app.post("/v1/models/switch")
async def switch_model(precision: str):
    """切换当前使用的模型精度"""
    if precision not in ["fp16", "int8", "int4"]:
        raise HTTPException(400, f"Unsupported precision: {precision}")
    
    # 如果模型未加载，先加载
    if precision not in gpu_manager.manager.models:
        gpu_manager.manager.load_model(precision)
    
    gpu_manager.manager.set_current_precision(precision)
    return {"status": "success", "current_precision": precision, "models": gpu_manager.status()}

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
    
    if prompt_wav:
        content = await prompt_wav.read()
        temp_path = INPUT_DIR / f"prompt_{uuid.uuid4().hex}.wav"
        temp_path.write_bytes(content)
        prompt_audio = str(temp_path)
    
    try:
        # 参数验证
        if mode == "zero_shot":
            if not prompt_audio:
                raise HTTPException(400, "zero_shot mode requires prompt_wav (reference audio)")
            # 自动识别 prompt_text
            if not prompt_text:
                print("Auto transcribing prompt audio with Fun-ASR...")
                prompt_text = transcribe_audio(prompt_audio)
                print(f"Transcribed: {prompt_text}")
        elif mode == "cross_lingual":
            if not prompt_audio:
                raise HTTPException(400, "cross_lingual mode requires prompt_wav (reference audio)")
        elif mode == "instruct":
            if not prompt_audio:
                raise HTTPException(400, "instruct mode requires prompt_wav (reference audio)")
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

HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CosyVoice - Text to Speech</title>
    <style>
        :root {
            --bg: #1a1a2e; --card: #16213e; --primary: #0f3460; --accent: #e94560;
            --text: #eee; --text-muted: #aaa; --border: #0f3460; --danger: #c0392b;
        }
        [data-theme="light"] {
            --bg: #f5f5f5; --card: #fff; --primary: #e3f2fd; --accent: #1976d2;
            --text: #333; --text-muted: #666; --border: #ddd; --danger: #e74c3c;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }
        .container { max-width: 900px; margin: 0 auto; padding: 20px; }
        header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
        h1 { font-size: 1.8em; background: linear-gradient(135deg, var(--accent), #ff6b6b); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .controls { display: flex; gap: 10px; }
        select, button { padding: 8px 16px; border: 1px solid var(--border); border-radius: 6px; background: var(--card); color: var(--text); cursor: pointer; }
        button:hover { background: var(--accent); color: white; }
        .card { background: var(--card); border-radius: 12px; padding: 20px; margin-bottom: 20px; border: 1px solid var(--border); }
        .card h3 { margin-bottom: 15px; color: var(--accent); }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; color: var(--text-muted); font-size: 0.9em; }
        textarea, input[type="text"], input[type="number"] { width: 100%; padding: 12px; border: 1px solid var(--border); border-radius: 8px; background: var(--bg); color: var(--text); font-size: 1em; resize: vertical; }
        textarea { min-height: 100px; }
        .row { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
        .btn-primary { background: var(--accent); color: white; border: none; padding: 14px 28px; font-size: 1.1em; width: 100%; }
        .btn-primary:disabled { opacity: 0.6; cursor: not-allowed; }
        audio { width: 100%; margin-top: 15px; }
        .status { padding: 10px; border-radius: 6px; background: var(--primary); margin-top: 10px; }
        .gpu-status { display: flex; justify-content: space-between; align-items: center; }
        .upload-area { border: 2px dashed var(--border); border-radius: 8px; padding: 20px; text-align: center; cursor: pointer; transition: all 0.3s; }
        .upload-area:hover { border-color: var(--accent); }
        .upload-area.dragover { background: var(--primary); }
        .hidden { display: none; }
        .tabs { display: flex; gap: 5px; margin-bottom: 15px; }
        .tab { padding: 10px 20px; border-radius: 6px 6px 0 0; cursor: pointer; background: var(--bg); }
        .tab.active { background: var(--accent); color: white; }
        .progress-bar { height: 4px; background: var(--border); border-radius: 2px; overflow: hidden; margin-top: 10px; }
        .progress-bar-fill { height: 100%; background: var(--accent); width: 0%; transition: width 0.3s; }
        @media (max-width: 600px) { .row { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎙️ CosyVoice</h1>
            <div class="controls">
                <select id="lang" onchange="setLang(this.value)">
                    <option value="zh-CN">简体中文</option>
                    <option value="en">English</option>
                    <option value="zh-TW">繁體中文</option>
                    <option value="ja">日本語</option>
                </select>
                <button onclick="toggleTheme()">🌓</button>
            </div>
        </header>

        <div class="card">
            <h3 data-i18n="input">输入文本</h3>
            <div class="form-group">
                <textarea id="text" placeholder="请输入要合成的文本..." data-i18n-placeholder="textPlaceholder">收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐。</textarea>
            </div>
        </div>

        <div class="card">
            <h3 data-i18n="mode">合成模式</h3>
            <div class="tabs">
                <div class="tab active" data-mode="zero_shot" data-i18n="zeroShot">零样本克隆</div>
                <div class="tab" data-mode="cross_lingual" data-i18n="crossLingual">跨语种</div>
                <div class="tab" data-mode="instruct" data-i18n="instruct">指令控制</div>
            </div>

            <div id="prompt-section">
                <!-- 音色选择 -->
                <div class="form-group">
                    <label data-i18n="voiceSelect">选择音色</label>
                    <div style="display: flex; gap: 10px; align-items: center;">
                        <select id="voice-select" style="flex: 1;" onchange="onVoiceSelect()">
                            <option value="">-- 上传新音频 --</option>
                        </select>
                        <button onclick="refreshVoices()" title="刷新列表" style="padding: 8px 12px;">🔄</button>
                        <button onclick="deleteSelectedVoice()" title="删除选中音色" style="padding: 8px 12px; background: var(--danger, #c0392b);">🗑️</button>
                    </div>
                </div>
                
                <!-- 上传新音频区域 -->
                <div id="upload-section">
                    <div class="form-group">
                        <label data-i18n="promptAudio">参考音频 (3-30秒)</label>
                        <div class="upload-area" id="upload-area">
                            <input type="file" id="prompt-file" accept="audio/*" class="hidden">
                            <p data-i18n="uploadHint">点击或拖拽上传音频文件</p>
                            <p id="file-name" style="color: var(--accent); margin-top: 10px;"></p>
                        </div>
                    </div>
                    <div class="form-group">
                        <label data-i18n="voiceName">音色名称 (可选，用于保存)</label>
                        <input type="text" id="voice-name" placeholder="如：张三的声音">
                    </div>
                    <div class="form-group" id="prompt-text-group">
                        <label data-i18n="promptText">参考文本 (留空则自动识别)</label>
                        <input type="text" id="prompt-text" placeholder="留空将使用 Fun-ASR 自动识别">
                    </div>
                    <div class="form-group">
                        <label style="display: flex; align-items: center; gap: 10px;">
                            <input type="checkbox" id="save-voice" checked> <span data-i18n="saveVoice">保存为自定义音色</span>
                        </label>
                    </div>
                </div>
            </div>

            <div id="instruct-section" class="hidden">
                <div class="form-group">
                    <label data-i18n="instructText">指令文本</label>
                    <input type="text" id="instruct-text" placeholder="用四川话说这句话">
                </div>
            </div>

            <div class="row">
                <div class="form-group">
                    <label data-i18n="speed">语速 (0.5-2.0)</label>
                    <input type="number" id="speed" value="1.0" min="0.5" max="2.0" step="0.1">
                </div>
            </div>
        </div>

        <div class="card">
            <div class="row" style="align-items: center;">
                <div class="form-group" style="margin-bottom: 0;">
                    <label style="display: flex; align-items: center; gap: 10px;">
                        <input type="checkbox" id="stream-mode" checked> <span data-i18n="streamMode">流式输出 (低延迟)</span>
                    </label>
                </div>
            </div>
            <button class="btn-primary" id="generate-btn" onclick="generate()" data-i18n="generate" style="margin-top: 15px;">生成语音</button>
            <div class="progress-bar"><div class="progress-bar-fill" id="progress"></div></div>
            <div id="timer" class="status hidden" style="text-align: center; font-size: 1.1em;"></div>
            <audio id="audio-output" controls class="hidden"></audio>
            <button id="download-btn" class="hidden" style="margin-top: 10px; padding: 10px 20px; background: var(--accent); color: white; border: none; border-radius: 8px; cursor: pointer; font-size: 1em;">
                📥 <span data-i18n="download">下载音频</span>
            </button>
        </div>

        <div class="card">
            <div class="gpu-status">
                <span id="gpu-info" data-i18n="gpuStatus">GPU 状态: 加载中...</span>
                <button onclick="offloadGPU()" data-i18n="releaseGPU">释放显存</button>
            </div>
        </div>
    </div>

    <script>
        const i18n = {
            'zh-CN': { input: '输入文本', mode: '合成模式', zeroShot: '零样本克隆', crossLingual: '跨语种', instruct: '指令控制', promptAudio: '参考音频 (3-30秒)', promptText: '参考文本 (留空自动识别)', instructText: '指令文本', speed: '语速', generate: '生成语音', gpuStatus: 'GPU 状态', releaseGPU: '释放显存', uploadHint: '点击或拖拽上传音频', textPlaceholder: '请输入要合成的文本...', streamMode: '流式输出 (低延迟)', generating: '生成中', completed: '完成', firstChunk: '首包', totalTime: '总耗时', audioDuration: '音频', voiceSelect: '选择音色', voiceName: '音色名称', saveVoice: '保存为自定义音色', newUpload: '-- 上传新音频 --', download: '下载音频' },
            'en': { input: 'Input Text', mode: 'Synthesis Mode', zeroShot: 'Zero-shot Clone', crossLingual: 'Cross-lingual', instruct: 'Instruct', promptAudio: 'Reference Audio (3-30s)', promptText: 'Reference Text (auto if empty)', instructText: 'Instruction', speed: 'Speed', generate: 'Generate', gpuStatus: 'GPU Status', releaseGPU: 'Release GPU', uploadHint: 'Click or drag to upload', textPlaceholder: 'Enter text to synthesize...', streamMode: 'Streaming (Low Latency)', generating: 'Generating', completed: 'Completed', firstChunk: 'TTFB', totalTime: 'Total', audioDuration: 'Audio', voiceSelect: 'Select Voice', voiceName: 'Voice Name', saveVoice: 'Save as custom voice', newUpload: '-- Upload new audio --', download: 'Download' },
            'zh-TW': { input: '輸入文本', mode: '合成模式', zeroShot: '零樣本克隆', crossLingual: '跨語種', instruct: '指令控制', promptAudio: '參考音頻', promptText: '參考文本', instructText: '指令文本', speed: '語速', generate: '生成語音', gpuStatus: 'GPU 狀態', releaseGPU: '釋放顯存', uploadHint: '點擊或拖拽上傳', textPlaceholder: '請輸入要合成的文本...', streamMode: '流式輸出 (低延遲)', generating: '生成中', completed: '完成', firstChunk: '首包', totalTime: '總耗時', audioDuration: '音頻', voiceSelect: '選擇音色', voiceName: '音色名稱', saveVoice: '保存為自定義音色', newUpload: '-- 上傳新音頻 --', download: '下載音頻' },
            'ja': { input: '入力テキスト', mode: '合成モード', zeroShot: 'ゼロショット', crossLingual: '多言語', instruct: '指示制御', promptAudio: '参照音声', promptText: '参照テキスト', instructText: '指示テキスト', speed: '速度', generate: '生成', gpuStatus: 'GPU状態', releaseGPU: 'GPU解放', uploadHint: 'クリックまたはドラッグ', textPlaceholder: 'テキストを入力...', streamMode: 'ストリーミング', generating: '生成中', completed: '完了', firstChunk: '初回', totalTime: '合計', audioDuration: '音声', voiceSelect: '音声選択', voiceName: '音声名', saveVoice: 'カスタム音声として保存', newUpload: '-- 新規アップロード --', download: 'ダウンロード' }
        };
        let currentLang = 'zh-CN', currentMode = 'zero_shot', promptFile = null, selectedVoiceId = null;

        function setLang(lang) {
            currentLang = lang;
            document.querySelectorAll('[data-i18n]').forEach(el => {
                const key = el.dataset.i18n;
                if (i18n[lang][key]) el.textContent = i18n[lang][key];
            });
            document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
                const key = el.dataset.i18nPlaceholder;
                if (i18n[lang][key]) el.placeholder = i18n[lang][key];
            });
            refreshVoices();
        }

        function toggleTheme() {
            document.body.dataset.theme = document.body.dataset.theme === 'light' ? '' : 'light';
        }

        // Voice management
        async function refreshVoices() {
            try {
                const res = await fetch('/v1/voices/custom');
                const data = await res.json();
                const select = document.getElementById('voice-select');
                const t = i18n[currentLang];
                select.innerHTML = `<option value="">${t.newUpload}</option>` + 
                    data.voices.map(v => `<option value="${v.id}" data-text="${v.text}">${v.name} (${v.id})</option>`).join('');
                if (selectedVoiceId) select.value = selectedVoiceId;
                onVoiceSelect();
            } catch (e) { console.error('Failed to load voices:', e); }
        }
        
        function onVoiceSelect() {
            const select = document.getElementById('voice-select');
            selectedVoiceId = select.value;
            const uploadSection = document.getElementById('upload-section');
            uploadSection.classList.toggle('hidden', !!selectedVoiceId);
            
            // 如果选择了已有音色，填充 prompt_text
            if (selectedVoiceId) {
                const option = select.options[select.selectedIndex];
                document.getElementById('prompt-text').value = option.dataset.text || '';
            }
        }
        
        async function deleteSelectedVoice() {
            if (!selectedVoiceId) { alert('请先选择要删除的音色'); return; }
            if (!confirm('确定删除此音色？')) return;
            try {
                await fetch(`/v1/voices/${selectedVoiceId}`, { method: 'DELETE' });
                selectedVoiceId = null;
                refreshVoices();
            } catch (e) { alert('删除失败: ' + e.message); }
        }

        document.querySelectorAll('.tab').forEach(tab => {
            tab.onclick = () => {
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                currentMode = tab.dataset.mode;
                document.getElementById('prompt-section').classList.remove('hidden');
                document.getElementById('prompt-text-group').classList.toggle('hidden', currentMode === 'cross_lingual');
                document.getElementById('instruct-section').classList.toggle('hidden', currentMode !== 'instruct');
            };
        });

        const uploadArea = document.getElementById('upload-area');
        const fileInput = document.getElementById('prompt-file');
        uploadArea.onclick = () => fileInput.click();
        uploadArea.ondragover = e => { e.preventDefault(); uploadArea.classList.add('dragover'); };
        uploadArea.ondragleave = () => uploadArea.classList.remove('dragover');
        uploadArea.ondrop = e => { e.preventDefault(); uploadArea.classList.remove('dragover'); handleFile(e.dataTransfer.files[0]); };
        fileInput.onchange = e => handleFile(e.target.files[0]);
        function handleFile(file) { if (file) { promptFile = file; document.getElementById('file-name').textContent = file.name; } }

        // Web Audio API streaming player
        const SAMPLE_RATE = 24000;
        const MIN_BUFFER_SIZE = 12000;  // 500ms buffer at 24kHz
        const FADE_SAMPLES = 1024;
        let audioContext = null;
        let activeSources = [];
        let nextPlayTime = 0;
        
        function stopAllAudio() {
            activeSources.forEach(s => { try { s.stop(); } catch(e) {} });
            activeSources = [];
        }
        
        function applyFadeIn(arr) {
            const len = Math.min(FADE_SAMPLES, arr.length);
            for (let i = 0; i < len; i++) arr[i] *= i / len;
        }
        
        function createWavBlob(pcmData, sampleRate) {
            const numChannels = 1, bitsPerSample = 16;
            const byteRate = sampleRate * numChannels * bitsPerSample / 8;
            const blockAlign = numChannels * bitsPerSample / 8;
            const dataSize = pcmData.length;
            const buffer = new ArrayBuffer(44 + dataSize);
            const view = new DataView(buffer);
            
            const writeString = (offset, str) => { for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i)); };
            writeString(0, 'RIFF');
            view.setUint32(4, 36 + dataSize, true);
            writeString(8, 'WAVE');
            writeString(12, 'fmt ');
            view.setUint32(16, 16, true);
            view.setUint16(20, 1, true);
            view.setUint16(22, numChannels, true);
            view.setUint32(24, sampleRate, true);
            view.setUint32(28, byteRate, true);
            view.setUint16(32, blockAlign, true);
            view.setUint16(34, bitsPerSample, true);
            writeString(36, 'data');
            view.setUint32(40, dataSize, true);
            new Uint8Array(buffer, 44).set(pcmData);
            return new Blob([buffer], { type: 'audio/wav' });
        }
        
        let currentAudioBlob = null;
        document.getElementById('download-btn').onclick = () => {
            if (currentAudioBlob) {
                const url = URL.createObjectURL(currentAudioBlob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `cosyvoice_${Date.now()}.wav`;
                a.click();
                URL.revokeObjectURL(url);
            }
        };

        async function generate() {
            const btn = document.getElementById('generate-btn');
            const progress = document.getElementById('progress');
            const audio = document.getElementById('audio-output');
            const timer = document.getElementById('timer');
            const downloadBtn = document.getElementById('download-btn');
            const isStream = document.getElementById('stream-mode').checked;
            const t = i18n[currentLang];
            
            btn.disabled = true;
            progress.style.width = '10%';
            timer.classList.remove('hidden');
            audio.classList.add('hidden');
            downloadBtn.classList.add('hidden');
            stopAllAudio();
            
            let audioBlob = null;  // Store audio for download
            const startTime = Date.now();
            let timerInterval = setInterval(() => {
                const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
                timer.textContent = `⏱️ ${t.generating}... ${elapsed}s`;
            }, 100);

            try {
                progress.style.width = '30%';
                let res;
                
                // 如果选择了已有音色，使用 OpenAI 风格 API
                if (selectedVoiceId) {
                    const body = {
                        input: document.getElementById('text').value,
                        voice: selectedVoiceId,
                        response_format: isStream ? 'pcm' : 'wav',
                        speed: parseFloat(document.getElementById('speed').value)
                    };
                    if (currentMode === 'instruct') {
                        body.instruct = document.getElementById('instruct-text').value;
                    }
                    res = await fetch('/v1/audio/speech', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(body)
                    });
                } else {
                    // 上传新音频
                    if (!promptFile) throw new Error('请上传参考音频或选择已有音色');
                    
                    const formData = new FormData();
                    formData.append('text', document.getElementById('text').value);
                    formData.append('mode', currentMode);
                    formData.append('speed', document.getElementById('speed').value);
                    formData.append('stream', isStream);
                    formData.append('prompt_wav', promptFile);
                    formData.append('prompt_text', document.getElementById('prompt-text').value);
                    if (currentMode === 'instruct') formData.append('instruct_text', document.getElementById('instruct-text').value);
                    
                    res = await fetch('/api/tts', { method: 'POST', body: formData });
                    
                    // 如果勾选了保存音色，保存它
                    if (document.getElementById('save-voice').checked && res.ok) {
                        const voiceName = document.getElementById('voice-name').value || promptFile.name;
                        const saveForm = new FormData();
                        saveForm.append('audio', promptFile);
                        saveForm.append('name', voiceName);
                        saveForm.append('text', document.getElementById('prompt-text').value);
                        fetch('/v1/voices/create', { method: 'POST', body: saveForm })
                            .then(() => refreshVoices());
                    }
                }
                
                if (!res.ok) throw new Error(await res.text());
                
                if (isStream) {
                    // Initialize Web Audio API
                    if (!audioContext) audioContext = new AudioContext({ sampleRate: SAMPLE_RATE });
                    if (audioContext.state === 'suspended') await audioContext.resume();
                    nextPlayTime = audioContext.currentTime + 0.15;
                    
                    const reader = res.body.getReader();
                    let pendingBytes = new Uint8Array(0);
                    let allPcmBytes = [];  // Collect all PCM data for download
                    let samples = [];
                    let totalSamples = 0;
                    let isFirstChunk = true;
                    let firstChunkTime = null;
                    
                    function playBuffer() {
                        if (samples.length === 0) return;
                        const float32 = new Float32Array(samples);
                        if (isFirstChunk) { applyFadeIn(float32); isFirstChunk = false; }
                        
                        const audioBuffer = audioContext.createBuffer(1, float32.length, SAMPLE_RATE);
                        audioBuffer.getChannelData(0).set(float32);
                        
                        const source = audioContext.createBufferSource();
                        source.buffer = audioBuffer;
                        source.connect(audioContext.destination);
                        activeSources.push(source);
                        source.onended = () => { const idx = activeSources.indexOf(source); if (idx > -1) activeSources.splice(idx, 1); };
                        
                        if (nextPlayTime < audioContext.currentTime - 0.1) nextPlayTime = audioContext.currentTime + 0.05;
                        source.start(nextPlayTime);
                        nextPlayTime += audioBuffer.duration;
                        totalSamples += samples.length;
                        samples = [];
                    }
                    
                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) { playBuffer(); break; }
                        
                        allPcmBytes.push(value);  // Collect for download
                        
                        if (!firstChunkTime) {
                            firstChunkTime = Date.now();
                            const ttfb = ((firstChunkTime - startTime) / 1000).toFixed(2);
                            timer.textContent = `⏱️ ${t.firstChunk}: ${ttfb}s | ${t.generating}...`;
                        }
                        
                        // Combine with pending bytes
                        const combined = new Uint8Array(pendingBytes.length + value.length);
                        combined.set(pendingBytes);
                        combined.set(value, pendingBytes.length);
                        
                        // Ensure byte alignment (Int16 = 2 bytes)
                        const validLength = Math.floor(combined.length / 2) * 2;
                        const validData = combined.slice(0, validLength);
                        pendingBytes = combined.slice(validLength);
                        
                        // Convert PCM to float samples
                        const int16 = new Int16Array(validData.buffer, validData.byteOffset, validData.length / 2);
                        for (let i = 0; i < int16.length; i++) samples.push(int16[i] / 32768);
                        
                        progress.style.width = `${30 + Math.min(60, samples.length / 1000)}%`;
                        if (samples.length >= MIN_BUFFER_SIZE) playBuffer();
                    }
                    
                    // Create WAV blob for download
                    const totalLength = allPcmBytes.reduce((sum, arr) => sum + arr.length, 0);
                    const pcmData = new Uint8Array(totalLength);
                    let offset = 0;
                    for (const chunk of allPcmBytes) { pcmData.set(chunk, offset); offset += chunk.length; }
                    audioBlob = createWavBlob(pcmData, SAMPLE_RATE);
                    
                    clearInterval(timerInterval);
                    const totalTime = ((Date.now() - startTime) / 1000).toFixed(2);
                    const ttfb = firstChunkTime ? ((firstChunkTime - startTime) / 1000).toFixed(2) : '-';
                    const audioDuration = (totalSamples / 24000).toFixed(2);
                    timer.textContent = `✅ ${t.firstChunk}: ${ttfb}s | ${t.totalTime}: ${totalTime}s | ${t.audioDuration}: ${audioDuration}s`;
                    timer.style.background = 'var(--accent)';
                    downloadBtn.classList.remove('hidden');
                    progress.style.width = '100%';
                } else {
                    const blob = await res.blob();
                    audioBlob = blob;
                    audio.src = URL.createObjectURL(blob);
                    clearInterval(timerInterval);
                    const totalTime = ((Date.now() - startTime) / 1000).toFixed(2);
                    timer.textContent = `✅ ${t.totalTime}: ${totalTime}s`;
                    timer.style.background = 'var(--accent)';
                    audio.classList.remove('hidden');
                    downloadBtn.classList.remove('hidden');
                    audio.play();
                    progress.style.width = '100%';
                }
            } catch (e) {
                clearInterval(timerInterval);
                timer.textContent = '❌ Error: ' + e.message;
                timer.style.background = '#c0392b';
            } finally {
                currentAudioBlob = audioBlob;
                btn.disabled = false;
                setTimeout(() => {
                    progress.style.width = '0%';
                    timer.style.background = 'var(--primary)';
                }, 2000);
            }
        }

        async function offloadGPU() {
            await fetch('/api/offload', { method: 'POST' });
            updateStatus();
        }

        async function updateStatus() {
            try {
                const res = await fetch('/api/status');
                const data = await res.json();
                const info = data.model_loaded 
                    ? `Model: ${data.model_dir} | GPU: ${data.gpu.memory_used}`
                    : 'Model not loaded';
                document.getElementById('gpu-info').textContent = info;
            } catch (e) {}
        }

        setLang('zh-CN');
        updateStatus();
        refreshVoices();
        setInterval(updateStatus, 30000);
    </script>
</body>
</html>'''

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8188"))
    uvicorn.run(app, host="0.0.0.0", port=port)
