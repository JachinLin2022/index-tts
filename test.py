import os
os.environ['VLLM_USE_V1'] = '0'
from indextts.infer_v2_vllm import IndexTTS2
import asyncio






async def main():
    index_tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_cuda_kernel=False, gpu_memory_utilization=0.5)
    await index_tts.infer(spk_audio_prompt='data/test_real.wav', text="你好，我是小明，今天天气不错", output_path="test.wav")

if __name__ == '__main__':
    asyncio.run(main())