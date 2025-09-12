import os
os.environ['VLLM_USE_V1'] = '0'
from indextts.infer_v2_vllm import IndexTTS2
# from indextts.infer_v2 import IndexTTS2
import asyncio






async def main():
    index_tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_cuda_kernel=False, gpu_memory_utilization=0.5)
    # index_tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_cuda_kernel=False)
    # await index_tts.infer(spk_audio_prompt='data/test_real.wav', text="Mason has been working on his game, and a key area we've identified for him to focus on is his smash technique.\n\nDeveloping a strong smash is crucial for an attacking game in badminton. Mason should focus on getting a high point of contact and a powerful wrist snap to generate more speed and angle on the shuttle. He could try shadow drills at home, mimicking the full smash motion, paying close attention to the wrist action. Consistent practice will help him build muscle memory and increase the power and accuracy of his attacking shots, making him a more formidable player on the court.", output_path="test.wav", verbose=False, max_text_tokens_per_segment=1200)
    
    
    
    # await index_tts.infer(spk_audio_prompt='data/test_real.wav', text="Mason has been working on his game, and a key area we've identified for him to focus on is his smash technique.\n\nDeveloping a strong smash is crucial for an attacking game in badminton. Mason should focus on getting a high point of contact and a powerful wrist snap to generate more speed and angle on the shuttle. He could try shadow drills at home, mimicking the full smash motion, paying close attention to the wrist action. Consistent practice will help him build muscle memory and increase the power and accuracy of his attacking shots, making him a more formidable player on the court.", output_path="test.wav", verbose=False, max_text_tokens_per_segment=120, interval_silence=0)
    
    await index_tts.infer_fast(spk_audio_prompt='data/test_real.wav', text="Mason has been working on his game, and a key area we've identified for him to focus on is his smash technique.\n\nDeveloping a strong smash is crucial for an attacking game in badminton. Mason should focus on getting a high point of contact and a powerful wrist snap to generate more speed and angle on the shuttle. He could try shadow drills at home, mimicking the full smash motion, paying close attention to the wrist action. Consistent practice will help him build muscle memory and increase the power and accuracy of his attacking shots, making him a more formidable player on the court.", output_path="test.wav", verbose=False, max_text_tokens_per_segment=120, interval_silence=0, segments_bucket_max_size=4)

if __name__ == '__main__':
    asyncio.run(main())