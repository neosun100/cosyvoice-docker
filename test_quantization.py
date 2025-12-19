#!/usr/bin/env python3
"""
CosyVoice Quantization Test Script
测试不同精度模型的性能对比

Usage:
    python test_quantization.py [--precisions fp16,int8,int4] [--text "测试文本"]
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torchaudio

ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR / "third_party/Matcha-TTS"))

from cosyvoice.cli.cosyvoice import CosyVoice3


def get_gpu_memory():
    """获取 GPU 显存使用情况"""
    if torch.cuda.is_available():
        return {
            "allocated": torch.cuda.memory_allocated() / 1024**3,
            "reserved": torch.cuda.memory_reserved() / 1024**3,
            "max_allocated": torch.cuda.max_memory_allocated() / 1024**3,
        }
    return {}


def test_model(model_dir: str, precision: str, test_text: str, prompt_audio: str, prompt_text: str):
    """测试单个精度模型"""
    print(f"\n{'='*60}")
    print(f"Testing {precision} model")
    print(f"{'='*60}")
    
    # 清理 GPU 缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    results = {
        "precision": precision,
        "test_text": test_text,
        "timestamp": datetime.now().isoformat(),
    }
    
    # 1. 测试模型加载时间
    print(f"\n[1/4] Loading {precision} model...")
    load_start = time.time()
    try:
        model = CosyVoice3(model_dir=model_dir, precision=precision)
        load_time = time.time() - load_start
        results["load_time"] = load_time
        results["load_success"] = True
        print(f"  ✓ Load time: {load_time:.2f}s")
    except Exception as e:
        print(f"  ✗ Failed to load: {e}")
        results["load_success"] = False
        results["error"] = str(e)
        return results
    
    # 2. 测试显存占用
    print(f"\n[2/4] Measuring GPU memory...")
    gpu_mem = get_gpu_memory()
    results["gpu_memory"] = gpu_mem
    print(f"  Allocated: {gpu_mem.get('allocated', 0):.2f} GB")
    print(f"  Reserved: {gpu_mem.get('reserved', 0):.2f} GB")
    
    # 3. 测试推理速度 (TTFB + Total)
    print(f"\n[3/4] Testing inference speed...")
    
    # 非流式推理
    inference_start = time.time()
    first_chunk_time = None
    chunks = []
    
    for i, chunk in enumerate(model.inference_zero_shot(
        test_text, prompt_text, prompt_audio, stream=True
    )):
        if first_chunk_time is None:
            first_chunk_time = time.time() - inference_start
        chunks.append(chunk['tts_speech'])
    
    total_time = time.time() - inference_start
    
    # 合并音频
    full_speech = torch.cat(chunks, dim=1)
    audio_duration = full_speech.shape[1] / model.sample_rate
    
    results["ttfb"] = first_chunk_time
    results["total_time"] = total_time
    results["audio_duration"] = audio_duration
    results["rtf"] = total_time / audio_duration if audio_duration > 0 else 0
    
    print(f"  TTFB: {first_chunk_time:.2f}s")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Audio duration: {audio_duration:.2f}s")
    print(f"  RTF: {results['rtf']:.2f}x")
    
    # 4. 保存测试音频
    print(f"\n[4/4] Saving test audio...")
    output_dir = ROOT_DIR / "test_outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"test_{precision}.wav"
    torchaudio.save(str(output_path), full_speech, model.sample_rate)
    results["output_file"] = str(output_path)
    print(f"  Saved to: {output_path}")
    
    # 清理
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return results


def generate_report(results: list, output_path: str):
    """生成测试报告"""
    report = {
        "test_date": datetime.now().isoformat(),
        "gpu_info": {
            "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "total_memory": f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB" if torch.cuda.is_available() else "N/A",
        },
        "results": results,
    }
    
    # 生成 Markdown 报告
    md_report = f"""# CosyVoice Quantization Test Report

**Test Date:** {report['test_date']}
**GPU:** {report['gpu_info']['name']} ({report['gpu_info']['total_memory']})

## Summary

| Precision | Load Time | TTFB | Total Time | RTF | GPU Memory |
|-----------|-----------|------|------------|-----|------------|
"""
    
    for r in results:
        if r.get("load_success"):
            md_report += f"| {r['precision']} | {r['load_time']:.2f}s | {r['ttfb']:.2f}s | {r['total_time']:.2f}s | {r['rtf']:.2f}x | {r['gpu_memory'].get('allocated', 0):.2f} GB |\n"
        else:
            md_report += f"| {r['precision']} | FAILED | - | - | - | - |\n"
    
    md_report += """
## Detailed Results

"""
    
    for r in results:
        md_report += f"### {r['precision']}\n\n"
        if r.get("load_success"):
            md_report += f"- **Load Time:** {r['load_time']:.2f}s\n"
            md_report += f"- **TTFB (Time to First Byte):** {r['ttfb']:.2f}s\n"
            md_report += f"- **Total Inference Time:** {r['total_time']:.2f}s\n"
            md_report += f"- **Audio Duration:** {r['audio_duration']:.2f}s\n"
            md_report += f"- **RTF (Real-Time Factor):** {r['rtf']:.2f}x\n"
            md_report += f"- **GPU Memory Allocated:** {r['gpu_memory'].get('allocated', 0):.2f} GB\n"
            md_report += f"- **Output File:** `{r['output_file']}`\n"
        else:
            md_report += f"- **Error:** {r.get('error', 'Unknown error')}\n"
        md_report += "\n"
    
    md_report += """
## Notes

- **TTFB**: Time from request to first audio chunk (streaming mode)
- **RTF**: Real-Time Factor, < 1.0 means faster than real-time
- **GPU Memory**: Memory allocated after model loading

## Recommendations

"""
    
    # 添加建议
    successful = [r for r in results if r.get("load_success")]
    if len(successful) >= 2:
        # 比较 TTFB
        best_ttfb = min(successful, key=lambda x: x['ttfb'])
        best_memory = min(successful, key=lambda x: x['gpu_memory'].get('allocated', float('inf')))
        
        md_report += f"- **Best TTFB:** {best_ttfb['precision']} ({best_ttfb['ttfb']:.2f}s)\n"
        md_report += f"- **Lowest Memory:** {best_memory['precision']} ({best_memory['gpu_memory'].get('allocated', 0):.2f} GB)\n"
    
    # 保存报告
    with open(output_path, 'w') as f:
        f.write(md_report)
    
    # 同时保存 JSON
    json_path = output_path.replace('.md', '.json')
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Report saved to: {output_path}")
    print(f"JSON data saved to: {json_path}")
    print(f"{'='*60}")
    
    return md_report


def main():
    parser = argparse.ArgumentParser(description="Test CosyVoice quantization performance")
    parser.add_argument("--model-dir", default=os.getenv("MODEL_DIR", "pretrained_models/Fun-CosyVoice3-0.5B"),
                        help="Model directory")
    parser.add_argument("--precisions", default="fp16,int8,int4",
                        help="Comma-separated list of precisions to test")
    parser.add_argument("--text", default="你好，这是一个量化模型测试。我们正在对比不同精度模型的性能表现。",
                        help="Test text for TTS")
    parser.add_argument("--prompt-audio", default="asset/zero_shot_prompt.wav",
                        help="Prompt audio file")
    parser.add_argument("--prompt-text", default="希望你以后能够做的比我还好哟。",
                        help="Prompt text")
    parser.add_argument("--output", default="test_outputs/quantization_report.md",
                        help="Output report path")
    
    args = parser.parse_args()
    
    precisions = [p.strip() for p in args.precisions.split(",")]
    
    print(f"CosyVoice Quantization Test")
    print(f"Model: {args.model_dir}")
    print(f"Precisions: {precisions}")
    print(f"Test text: {args.text}")
    
    # 确保输出目录存在
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    results = []
    for precision in precisions:
        result = test_model(
            model_dir=args.model_dir,
            precision=precision,
            test_text=args.text,
            prompt_audio=args.prompt_audio,
            prompt_text=args.prompt_text,
        )
        results.append(result)
    
    # 生成报告
    report = generate_report(results, args.output)
    print("\n" + report)


if __name__ == "__main__":
    main()
