import os
os.environ['VLLM_USE_V1'] = '0'
from indextts.infer_v2_vllm import IndexTTS2
# from indextts.infer_v2 import IndexTTS2
import asyncio






async def main():
    index_tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_cuda_kernel=False, gpu_memory_utilization=0.5)
    # index_tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_cuda_kernel=False)
    # await index_tts.infer(spk_audio_prompt='data/test_real.wav', text="Mason has been working on his game, and a key area we've identified for him to focus on is his smash technique.\n\nDeveloping a strong smash is crucial for an attacking game in badminton. Mason should focus on getting a high point of contact and a powerful wrist snap to generate more speed and angle on the shuttle. He could try shadow drills at home, mimicking the full smash motion, paying close attention to the wrist action. Consistent practice will help him build muscle memory and increase the power and accuracy of his attacking shots, making him a more formidable player on the court.", output_path="test.wav", verbose=False, max_text_tokens_per_segment=1200)
    
    
    
    # await index_tts.infer_fast(spk_audio_prompt='data/tina_test.wav', text="Mike is showing great potential in his Fall In-Person Chess Lessons. We've observed that chess is a wonderful activity for him, significantly helping to build his focus and concentration. To further support his development, we believe he can benefit from exploring additional resources. To help Mike continue advancing, we recommend a two-pronged approach. First, to deepen his understanding and tactical skills, Mike could explore beginner-friendly chess books like Chess for Children or Bobby Fischer Teaches Chess. These resources make learning fun and can significantly improve his strategic thinking. Second, engaging in online puzzles and supervised tournaments on platforms like ChessKid will provide valuable practice and expose him to different playing styles in a supportive environment. This consistent engagement will not only enhance his chess level but also reinforce his focus and problem-solving abilities.", output_path="test2_no_emo.wav", verbose=False, max_text_tokens_per_segment=120, interval_silence=0)
    
    await index_tts.infer_parallel(spk_audio_prompt='data/tina_test.wav', text="""Mike is showing great potential in his Fall In-Person Chess Lessons.""", output_path="test_parallel.wav", verbose=False, max_text_tokens_per_segment=120, interval_silence=0)

if __name__ == '__main__':
    asyncio.run(main())