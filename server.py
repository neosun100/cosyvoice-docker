"""
CosyVoice TTS Server - REST API

Features:
    - OpenAI-compatible /v1/audio/speech endpoint
    - Voice management (create, list, delete)
    - Multi-model support with dynamic switching
    - Optional vLLM acceleration
    - Streaming audio output (PCM/WAV)

Mount Points:
    /models       - TTS models (read-only)
    /data/output  - Generated audio files
    /data/voices  - Custom voice storage
    /root/.cache  - HuggingFace/transformers cache
"""

import os
import sys
import gc
import uuid
import logging
import threading
from pathlib import Path
from typing import Optional, Literal
from contextlib import asynccontextmanager

import torch
import torchaudio
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Setup paths
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR / "third_party/Matcha-TTS"))

# Configure logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# =============================================================================
# vLLM Registration (must be done before importing CosyVoice)
# =============================================================================
ENABLE_VLLM = os.getenv("ENABLE_VLLM", "false").lower() == "true"

if ENABLE_VLLM:
    try:
        from vllm import ModelRegistry
        from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM

        ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)
        logger.info("vLLM model registered successfully")
    except ImportError as e:
        logger.warning(f"vLLM not available, falling back to standard inference: {e}")
        ENABLE_VLLM = False

from cosyvoice.cli.cosyvoice import AutoModel
from voice_manager import VoiceManager

# =============================================================================
# Directory Configuration
# =============================================================================
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models/Fun-CosyVoice3-0.5B"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/data/output"))
VOICES_DIR = Path(os.getenv("VOICES_DIR", "/data/voices"))


# =============================================================================
# Model Manager
# =============================================================================
class ModelManager:
    """Thread-safe model manager with multi-model support."""

    def __init__(self):
        self.model = None
        self.model_dir = None
        self.lock = threading.Lock()
        self.enable_vllm = ENABLE_VLLM

    def load(self, model_dir: str) -> None:
        """Load model from specified directory."""
        with self.lock:
            if self.model is not None and self.model_dir == str(model_dir):
                return self.model

            if self.model is not None:
                self._unload()

            logger.info(f"Loading model from {model_dir}...")
            logger.info(
                f"vLLM acceleration: {'enabled' if self.enable_vllm else 'disabled'}"
            )

            model_kwargs = {"model_dir": str(model_dir)}

            if self.enable_vllm:
                model_kwargs["load_vllm"] = True
                if (
                    "CosyVoice3" in str(model_dir)
                    or "cosyvoice3" in str(model_dir).lower()
                ):
                    model_kwargs["fp16"] = False
                else:
                    model_kwargs["fp16"] = True

            self.model = AutoModel(**model_kwargs)
            self.model_dir = str(model_dir)
            logger.info(f"Model loaded: {model_dir}")
            return self.model

    def _unload(self) -> None:
        """Unload current model and free GPU memory."""
        if self.model:
            del self.model
            self.model = None
            self.model_dir = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Model unloaded, GPU memory released")

    def get(self, model_dir: Optional[str] = None):
        """Get model instance, loading if necessary."""
        target_dir = model_dir or str(MODEL_DIR)

        if self.model is None or self.model_dir != target_dir:
            self.load(target_dir)

        return self.model

    def unload(self) -> None:
        """Public method to unload model."""
        with self.lock:
            self._unload()

    def preload(self, voice_manager: "VoiceManager") -> None:
        """
        Warm up model and voice embeddings.

        1. Get pretrained voice list and set to voice_manager
        2. Warm up pretrained voices (run short inference to trigger JIT/CUDA compilation)
        3. Warm up custom voice embeddings (trigger caching)
        """
        model = self.get()
        logger.info("=" * 50)
        logger.info("Starting model preload...")

        # 1. Get pretrained voice list
        pretrained_spks = model.list_available_spks()
        voice_manager.set_pretrained_voices(pretrained_spks)
        logger.info(
            f"Found {len(pretrained_spks)} pretrained voices: {pretrained_spks}"
        )

        # 2. Warm up pretrained voices (trigger JIT/CUDA compilation)
        if pretrained_spks:
            try:
                warmup_text = "warmup"
                spk = pretrained_spks[0]
                logger.info(f"Warming up pretrained voice: {spk}")
                for _ in model.inference_sft(warmup_text, spk, stream=False):
                    pass
                logger.info(f"✓ Pretrained voice warmup completed: {spk}")
            except Exception as e:
                logger.warning(f"✗ Pretrained voice warmup failed: {e}")

        # 3. Warm up custom voice embeddings
        custom_voices = voice_manager.list_custom()
        if custom_voices:
            logger.info(f"Preloading {len(custom_voices)} custom voice embeddings...")
            for voice in custom_voices:
                if voice.audio_path and Path(voice.audio_path).exists():
                    try:
                        # Call frontend_zero_shot to trigger embedding cache
                        model.frontend.frontend_zero_shot(
                            "warmup",
                            voice.text or "",
                            voice.audio_path,
                            model.sample_rate,
                            "",
                        )
                        logger.info(f"  ✓ Cached: {voice.name} ({voice.voice_id})")
                    except Exception as e:
                        logger.warning(f"  ✗ Failed: {voice.name} - {e}")

            cache_size = self.get_cache_size()
            logger.info(f"Voice embeddings cached: {cache_size}")

        logger.info("Model preload completed!")
        logger.info("=" * 50)

    def get_cache_size(self) -> int:
        """Get current embedding cache size."""
        if (
            self.model
            and hasattr(self.model, "frontend")
            and hasattr(self.model.frontend, "prompt_cache")
        ):
            return len(self.model.frontend.prompt_cache)
        return 0

    def status(self) -> dict:
        """Get current model and GPU status."""
        gpu_info = {"available": torch.cuda.is_available()}

        if torch.cuda.is_available():
            gpu_info.update(
                {
                    "device": torch.cuda.get_device_name(0),
                    "memory_used_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
                    "memory_total_gb": round(
                        torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
                    ),
                }
            )

        return {
            "model_loaded": self.model is not None,
            "model_dir": self.model_dir,
            "vllm_enabled": self.enable_vllm,
            "voice_cache_size": self.get_cache_size(),
            "gpu": gpu_info,
        }


# Initialize managers
model_manager = ModelManager()
voice_manager = VoiceManager(VOICES_DIR)


# =============================================================================
# FastAPI Application
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: preload model and voices on startup."""
    logger.info("=" * 60)
    logger.info("CosyVoice TTS Server Starting...")
    logger.info("=" * 60)

    try:
        # Load model and preload voices
        model_manager.load(MODEL_DIR)
        model_manager.preload(voice_manager)
    except Exception as e:
        logger.error(f"Failed to preload model: {e}")

    logger.info("=" * 60)
    logger.info("Server Ready!")
    logger.info("=" * 60)

    yield

    logger.info("=" * 60)
    logger.info("Server Shutting Down...")
    logger.info("=" * 60)
    model_manager.unload()


app = FastAPI(
    title="CosyVoice TTS API",
    description="High-quality Text-to-Speech API with voice cloning support",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Request/Response Models
# =============================================================================
class SpeechRequest(BaseModel):
    """OpenAI-compatible speech synthesis request."""

    model: str = Field(default="cosyvoice", description="Model identifier")
    input: str = Field(..., description="Text to synthesize")
    voice: str = Field(..., description="Voice ID (pretrained or custom)")
    response_format: Literal["wav", "pcm"] = Field(
        default="wav", description="Output format"
    )
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speech speed")
    instruct: Optional[str] = Field(
        default=None, description="Instruction for style control"
    )


class VoiceResponse(BaseModel):
    """Voice information response."""

    voice_id: str = Field(..., description="Voice ID")
    name: str = Field(..., description="Voice name")
    text: str = Field(..., description="Audio transcript")
    type: Literal["pretrained", "custom"] = Field(
        default="custom", description="Voice type"
    )
    description: Optional[str] = Field(None, description="Voice description")
    created_at: Optional[int] = Field(None, description="Creation timestamp")


class VoiceListResponse(BaseModel):
    """Voice list response."""

    voices: list[VoiceResponse]


class TTSRequest(BaseModel):
    """Full-featured TTS request with all modes."""

    text: str = Field(..., description="Text to synthesize")
    mode: Literal["zero_shot", "cross_lingual", "instruct", "sft"] = Field(
        default="zero_shot", description="Synthesis mode"
    )
    prompt_text: str = Field(default="", description="Reference audio transcript")
    instruct_text: str = Field(default="", description="Style instruction")
    speaker_id: str = Field(default="", description="Speaker ID for sft mode")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speech speed")
    stream: bool = Field(default=False, description="Enable streaming output")


class ModelSwitchRequest(BaseModel):
    """Request to switch model."""

    model_dir: str = Field(..., description="Path to model directory")


# =============================================================================
# Audio Utilities
# =============================================================================
def save_audio(speech: torch.Tensor, sample_rate: int, filename: str) -> Path:
    """Save audio tensor to WAV file."""
    output_path = OUTPUT_DIR / filename
    torchaudio.save(str(output_path), speech, sample_rate)
    return output_path


def generate_pcm_stream(model_output, sample_rate: int):
    """Generate PCM audio stream from model output."""
    for chunk in model_output:
        wav_chunk = chunk["tts_speech"].numpy().flatten()
        audio = (wav_chunk * 32767).astype(np.int16).tobytes()
        yield audio


# =============================================================================
# OpenAI-Compatible API: /v1/audio/*
# =============================================================================


@app.post("/v1/audio/speech")
async def create_speech(request: SpeechRequest):
    """
    Generate speech from text (OpenAI-compatible).

    Uses inference_instruct2 for synthesis, supporting style instructions.
    """
    model = model_manager.get()
    voice = voice_manager.get(request.voice)

    if not voice:
        raise HTTPException(
            status_code=400,
            detail=f"Voice '{request.voice}' not found. Use GET /v1/audio/voices to list available voices.",
        )

    # Prepare instruct text
    instruct_text = request.instruct or ""

    if voice.type == "custom":
        # Custom voice: use zero_shot with prompt audio
        if not voice.audio_path or not Path(voice.audio_path).exists():
            raise HTTPException(status_code=500, detail="Voice audio file not found")

        # Use inference_instruct2 for custom voice
        if instruct_text:
            output = model.inference_instruct2(
                request.input,
                instruct_text,
                voice.audio_path,
                stream=(request.response_format == "pcm"),
                speed=request.speed,
            )
        else:
            instruct_text = instruct_text.replace("zero_shot", "")
            instruct_text = f"You are a helpful assistant.{instruct_text}<|endofprompt|>{voice.text}"

            output = model.inference_zero_shot(
                request.input,
                instruct_text,
                voice.audio_path,
                stream=(request.response_format == "pcm"),
                speed=request.speed,
            )

    else:
        # Pretrained voice: use SFT mode
        output = model.inference_sft(
            request.input,
            request.voice,
            stream=(request.response_format == "pcm"),
            speed=request.speed,
        )

    if request.response_format == "pcm":
        return StreamingResponse(
            generate_pcm_stream(output, model.sample_rate),
            media_type="audio/pcm",
            headers={"X-Sample-Rate": str(model.sample_rate)},
        )

    # Collect and return WAV
    speeches = [chunk["tts_speech"] for chunk in output]
    full_speech = torch.cat(speeches, dim=1)
    filename = f"speech_{uuid.uuid4().hex[:8]}.wav"
    output_path = save_audio(full_speech, model.sample_rate, filename)

    return FileResponse(str(output_path), media_type="audio/wav", filename=filename)


@app.post("/v1/audio/voices", response_model=VoiceResponse)
async def create_voice(
    file: UploadFile = File(..., description="Audio file (3-30 seconds)"),
    name: str = Form(..., description="Voice name"),
    text: str = Form("", description="Audio transcript (optional)"),
    description: str = Form("", description="Voice description (optional)"),
):
    """
    Create a custom voice from audio file.

    Upload a 3-30 second audio file to create a voice that can be used for synthesis.
    """
    content = await file.read()

    if len(content) < 1000:
        raise HTTPException(status_code=400, detail="Audio file too small")

    voice = voice_manager.create(
        name=name,
        audio_data=content,
        text=text,
        description=description,
    )

    return VoiceResponse(
        voice_id=voice.voice_id,
        name=voice.name,
        type=voice.type,
        text=voice.text,
        description=voice.description,
        created_at=voice.created_at,
    )


@app.get("/v1/audio/voices", response_model=VoiceListResponse)
async def list_voices(
    type: Optional[Literal["all", "pretrained", "custom"]] = Query(
        default="all", description="Filter by voice type"
    ),
):
    """
    List available voices.

    Returns both pretrained and custom voices by default.
    """
    if type == "pretrained":
        voices = voice_manager.list_pretrained()
    elif type == "custom":
        voices = voice_manager.list_custom()
    else:
        voices = voice_manager.list_all()

    return VoiceListResponse(
        voices=[
            VoiceResponse(
                voice_id=v.voice_id,
                name=v.name,
                type=v.type,
                text=v.text,
                description=v.description,
                created_at=v.created_at,
            )
            for v in voices
        ]
    )


@app.delete("/v1/audio/voices/{voice_id}")
async def delete_voice(voice_id: str):
    """
    Delete a custom voice.

    Only custom voices can be deleted. Pretrained voices cannot be deleted.
    """
    voice = voice_manager.get(voice_id)

    if not voice:
        raise HTTPException(status_code=404, detail="Voice not found")

    if voice.type == "pretrained":
        raise HTTPException(status_code=400, detail="Cannot delete pretrained voice")

    if not voice_manager.delete(voice_id):
        raise HTTPException(status_code=500, detail="Failed to delete voice")

    return {"deleted": True, "voice_id": voice_id}


@app.get("/v1/audio/voices/{voice_id}", response_model=VoiceResponse)
async def get_voice(voice_id: str):
    """Get voice details."""
    voice = voice_manager.get(voice_id)

    if not voice:
        raise HTTPException(status_code=404, detail="Voice not found")

    return VoiceResponse(
        voice_id=voice.voice_id,
        name=voice.name,
        type=voice.type,
        text=voice.text,
        description=voice.description,
        created_at=voice.created_at,
    )


# =============================================================================
# Legacy API: /api/*
# =============================================================================


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model": model_manager.status()}


@app.get("/api/status")
async def get_status():
    """Get detailed system status including voice counts."""
    status = model_manager.status()
    status["voices"] = {
        "pretrained_count": len(voice_manager.list_pretrained()),
        "custom_count": len(voice_manager.list_custom()),
    }
    return status


@app.post("/api/model/switch")
async def switch_model(request: ModelSwitchRequest):
    """Switch to a different model."""
    try:
        model = model_manager.load(request.model_dir)
        # Update pretrained voices
        pretrained = model.list_available_spks()
        voice_manager.set_pretrained_voices(pretrained)
        return {"status": "success", "model_dir": request.model_dir}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/model/unload")
async def unload_model():
    """Unload model and release GPU memory."""
    model_manager.unload()
    return {"status": "success", "message": "Model unloaded"}


@app.post("/api/tts")
async def tts(
    text: str = Form(...),
    mode: str = Form("zero_shot"),
    prompt_text: str = Form(""),
    instruct_text: str = Form(""),
    speaker_id: str = Form(""),
    speed: float = Form(1.0),
    stream: bool = Form(False),
    prompt_wav: Optional[UploadFile] = File(None),
):
    """
    Full-featured TTS endpoint supporting all synthesis modes.

    Modes:
    - zero_shot: Clone voice from prompt audio (requires prompt_wav + prompt_text)
    - cross_lingual: Cross-language synthesis (requires prompt_wav)
    - instruct: Style-controlled synthesis (requires prompt_wav + instruct_text)
    - sft: Use pretrained speaker (requires speaker_id)
    """
    model = model_manager.get()
    prompt_audio_path = None

    try:
        # Handle prompt audio upload
        if prompt_wav:
            content = await prompt_wav.read()
            prompt_audio_path = OUTPUT_DIR / f"prompt_{uuid.uuid4().hex}.wav"
            prompt_audio_path.write_bytes(content)

        # Validate and execute based on mode
        if mode == "zero_shot":
            if not prompt_audio_path:
                raise HTTPException(400, "zero_shot mode requires prompt_wav")
            if not prompt_text:
                raise HTTPException(400, "zero_shot mode requires prompt_text")
            output = model.inference_zero_shot(
                text, prompt_text, str(prompt_audio_path), stream=stream, speed=speed
            )

        elif mode == "cross_lingual":
            if not prompt_audio_path:
                raise HTTPException(400, "cross_lingual mode requires prompt_wav")
            output = model.inference_cross_lingual(
                text, str(prompt_audio_path), stream=stream, speed=speed
            )

        elif mode == "instruct":
            if not prompt_audio_path:
                raise HTTPException(400, "instruct mode requires prompt_wav")
            if not instruct_text:
                raise HTTPException(400, "instruct mode requires instruct_text")

            if hasattr(model, "inference_instruct2"):
                output = model.inference_instruct2(
                    text,
                    instruct_text,
                    str(prompt_audio_path),
                    stream=stream,
                    speed=speed,
                )
            else:
                output = model.inference_instruct(
                    text, speaker_id, instruct_text, stream=stream, speed=speed
                )

        elif mode == "sft":
            if not speaker_id:
                raise HTTPException(400, "sft mode requires speaker_id")
            output = model.inference_sft(text, speaker_id, stream=stream, speed=speed)

        else:
            raise HTTPException(400, f"Unknown mode: {mode}")

        # Handle streaming response
        if stream:

            def stream_with_cleanup():
                try:
                    yield from generate_pcm_stream(output, model.sample_rate)
                finally:
                    if prompt_audio_path and prompt_audio_path.exists():
                        prompt_audio_path.unlink()

            return StreamingResponse(
                stream_with_cleanup(),
                media_type="audio/pcm",
                headers={"X-Sample-Rate": str(model.sample_rate)},
            )

        # Collect and return WAV
        speeches = [chunk["tts_speech"] for chunk in output]
        full_speech = torch.cat(speeches, dim=1)
        filename = f"tts_{uuid.uuid4().hex[:8]}.wav"
        output_path = save_audio(full_speech, model.sample_rate, filename)

        # Cleanup prompt file
        if prompt_audio_path and prompt_audio_path.exists():
            prompt_audio_path.unlink()

        return FileResponse(str(output_path), media_type="audio/wav", filename=filename)

    except HTTPException:
        raise
    except Exception as e:
        if prompt_audio_path and prompt_audio_path.exists():
            prompt_audio_path.unlink()
        logger.error(f"TTS error: {e}")
        raise HTTPException(500, str(e))


@app.get("/api/download/{filename}")
async def download_file(filename: str):
    """Download generated audio file."""
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(str(path), media_type="audio/wav", filename=filename)


# =============================================================================
# Main Entry
# =============================================================================
if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="CosyVoice TTS Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8188, help="Server port")
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of worker processes"
    )
    args = parser.parse_args()

    logger.info(f"Starting CosyVoice TTS Server on {args.host}:{args.port}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"vLLM acceleration: {'enabled' if ENABLE_VLLM else 'disabled'}")

    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
    )
