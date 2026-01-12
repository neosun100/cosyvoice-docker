import json
import shutil
import uuid
import time
import logging
from pathlib import Path
from typing import Optional, List, Dict, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class VoiceInfo(BaseModel):
    """音色信息"""

    voice_id: str = Field(..., description="Voice ID")
    name: str = Field(..., description="Voice name")
    type: Literal["pretrained", "custom"] = Field(..., description="Voice type")
    text: Optional[str] = Field(None, description="Audio transcript")
    audio_path: Optional[str] = Field(
        None, description="Audio file path (relative to voices_dir)"
    )
    description: Optional[str] = Field(None, description="Voice description")
    created_at: Optional[int] = Field(None, description="Creation timestamp")

    class Config:
        # Exclude None values when converting to dict
        use_enum_values = True


class VoiceManager:
    """
    音色管理器

    - 启动时从 voices.json 加载自定义音色
    - 运行时管理音色的增删查
    - 支持预训练音色和自定义音色混合查询
    """

    def __init__(self, voices_dir: Path):
        self.voices_dir = Path(voices_dir)
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.voices_dir / "voices.json"

        # 自定义音色 {voice_id: VoiceInfo}
        self._custom_voices: Dict[str, VoiceInfo] = {}
        # 预训练音色 {voice_id: VoiceInfo}
        self._pretrained_voices: Dict[str, VoiceInfo] = {}

        self._load()

    def _load(self) -> None:
        """从 voices.json 加载自定义音色"""
        if not self.index_file.exists():
            logger.info("No voices.json found, starting with empty voice list")
            return

        try:
            data = json.loads(self.index_file.read_text(encoding="utf-8"))
            for voice_id, info in data.items():
                # audio_path 存储的是相对路径，需要转换为绝对路径验证
                audio_path_rel = info.get("audio_path")
                if audio_path_rel:
                    audio_path_abs = self.voices_dir / audio_path_rel
                    if not audio_path_abs.exists():
                        logger.warning(
                            f"Voice {voice_id} audio file not found: {audio_path_abs}"
                        )
                        continue

                self._custom_voices[voice_id] = VoiceInfo(
                    voice_id=voice_id,
                    name=info.get("name", voice_id),
                    type="custom",
                    text=info.get("text"),
                    audio_path=audio_path_rel,  # 保存相对路径
                    description=info.get("description"),
                    created_at=info.get("created_at"),
                )
            logger.info(f"Loaded {len(self._custom_voices)} custom voices")
        except Exception as e:
            logger.error(f"Failed to load voices.json: {e}")

    def _save(self) -> None:
        """保存自定义音色到 voices.json"""
        data = {
            voice_id: voice.model_dump(exclude_none=True)
            for voice_id, voice in self._custom_voices.items()
        }
        self.index_file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def set_pretrained_voices(self, voice_names: List[str]) -> None:
        """
        设置预训练音色列表（模型加载后调用）

        Args:
            voice_names: 模型支持的预训练音色名称列表
        """
        self._pretrained_voices.clear()
        for name in voice_names:
            voice_id = name  # 预训练音色使用名称作为 ID
            self._pretrained_voices[voice_id] = VoiceInfo(
                voice_id=voice_id,
                name=name,
                type="pretrained",
            )
        logger.info(f"Set {len(self._pretrained_voices)} pretrained voices")

    def create(
        self,
        name: str,
        audio_data: bytes,
        text: str = "",
        description: str = "",
    ) -> VoiceInfo:
        """
        创建自定义音色

        Args:
            name: 音色名称
            audio_data: 音频文件内容
            text: 音频对应的文本
            description: 音色描述

        Returns:
            VoiceInfo: 创建的音色信息
        """
        voice_id = uuid.uuid4().hex[:12]
        voice_dir = self.voices_dir / voice_id
        voice_dir.mkdir(exist_ok=True)

        # 保存音频文件
        audio_path = voice_dir / "prompt.wav"
        audio_path.write_bytes(audio_data)

        # 存储相对路径（相对于 voices_dir）
        audio_path_rel = f"{voice_id}/prompt.wav"

        voice = VoiceInfo(
            voice_id=voice_id,
            name=name,
            type="custom",
            text=text or None,
            audio_path=audio_path_rel,  # 使用相对路径
            description=description or None,
            created_at=int(time.time()),
        )

        self._custom_voices[voice_id] = voice
        self._save()

        logger.info(f"Created voice: {voice_id} ({name})")
        return voice

    def get(self, voice_id: str) -> Optional[VoiceInfo]:
        """
        获取音色信息

        Args:
            voice_id: 音色 ID

        Returns:
            VoiceInfo 或 None（audio_path 返回绝对路径）
        """
        # 先查自定义音色
        if voice_id in self._custom_voices:
            voice = self._custom_voices[voice_id]
            # 如果有相对路径，转换为绝对路径返回
            if voice.audio_path:
                return voice.model_copy(
                    update={"audio_path": str(self.voices_dir / voice.audio_path)}
                )
            return voice
        # 再查预训练音色
        if voice_id in self._pretrained_voices:
            return self._pretrained_voices[voice_id]
        return None

    def delete(self, voice_id: str) -> bool:
        """
        删除自定义音色

        Args:
            voice_id: 音色 ID

        Returns:
            是否删除成功
        """
        if voice_id not in self._custom_voices:
            return False

        # 删除音频文件目录
        voice_dir = self.voices_dir / voice_id
        if voice_dir.exists():
            shutil.rmtree(voice_dir)

        del self._custom_voices[voice_id]
        self._save()

        logger.info(f"Deleted voice: {voice_id}")
        return True

    def list_all(self, include_pretrained: bool = True) -> List[VoiceInfo]:
        """
        列出所有音色

        Args:
            include_pretrained: 是否包含预训练音色

        Returns:
            音色列表（audio_path 返回绝对路径）
        """
        # 自定义音色需要转换为绝对路径
        custom_voices = [
            v.model_copy(update={"audio_path": str(self.voices_dir / v.audio_path)})
            if v.audio_path
            else v
            for v in self._custom_voices.values()
        ]

        if include_pretrained:
            custom_voices.extend(self._pretrained_voices.values())
        return custom_voices

    def list_custom(self) -> List[VoiceInfo]:
        """列出自定义音色（audio_path 返回绝对路径）"""
        return [
            v.model_copy(update={"audio_path": str(self.voices_dir / v.audio_path)})
            if v.audio_path
            else v
            for v in self._custom_voices.values()
        ]

    def list_pretrained(self) -> List[VoiceInfo]:
        """列出预训练音色"""
        return list(self._pretrained_voices.values())

    def exists(self, voice_id: str) -> bool:
        """检查音色是否存在"""
        return voice_id in self._custom_voices or voice_id in self._pretrained_voices

    def is_custom(self, voice_id: str) -> bool:
        """检查是否为自定义音色"""
        return voice_id in self._custom_voices

    def is_pretrained(self, voice_id: str) -> bool:
        """检查是否为预训练音色"""
        return voice_id in self._pretrained_voices
