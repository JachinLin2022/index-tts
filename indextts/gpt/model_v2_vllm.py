import patch_vllm  # ⚠️ Monkey Patch, do not delete this line
import uuid
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import transformers
from transformers import GPT2Config, LogitsProcessorList
from indextts.gpt.transformers_gpt2 import GPT2PreTrainedModel, GPT2Model

# from transformers import GPT2Config, GPT2PreTrainedModel, LogitsProcessorList
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.utils.model_parallel_utils import (assert_device_map,
                                                     get_device_map)

from indextts.gpt.conformer_encoder import ConformerEncoder
from indextts.gpt.perceiver import PerceiverResampler
from indextts.utils.arch_util import AttentionBlock
from indextts.utils.typical_sampling import TypicalLogitsWarper

from vllm import AsyncLLMEngine, SamplingParams, TokensPrompt
from vllm.engine.arg_utils import AsyncEngineArgs
import asyncio
from indextts.gpt.model_v2 import GPT2InferenceModel, ConditioningEncoder, MelEncoder, build_hf_gpt_transformer


def null_position_embeddings(range, dim):
    return torch.zeros((range.shape[0], range.shape[1], dim), device=range.device)




class UnifiedVoice(nn.Module):
    def __init__(self, layers=8, model_dim=512, heads=8, max_text_tokens=120, max_mel_tokens=250, max_conditioning_inputs=1,
                 mel_length_compression=1024, number_text_tokens=256,
                 start_text_token=0, stop_text_token=1, number_mel_codes=8194, start_mel_token=8192, stop_mel_token=8193,
                 train_solo_embeddings=False, use_mel_codes_as_input=True,
                 checkpointing=True, types=1, model_dir=None, gpu_memory_utilization=0.25,
                 condition_num_latent=32, condition_type="perceiver", condition_module=None, emo_condition_module=None):
        """
        Args:
            layers: Number of layers in transformer stack.
            model_dim: Operating dimensions of the transformer
            heads: Number of transformer heads. Must be divisible by model_dim. Recommend model_dim//64
            max_text_tokens: Maximum number of text tokens that will be encountered by model.
            max_mel_tokens: Maximum number of MEL tokens that will be encountered by model.
            max_conditioning_inputs: Maximum number of conditioning inputs provided to the model. If (1), conditioning input can be of format (b,80,s), otherwise (b,n,80,s).
            mel_length_compression: The factor between <number_input_samples> and <mel_tokens>. Used to compute MEL code padding given wav input length.
            number_text_tokens:
            start_text_token:
            stop_text_token:
            number_mel_codes:
            start_mel_token:
            stop_mel_token:
            train_solo_embeddings:
            use_mel_codes_as_input:
            checkpointing:
            condition_type: perceiver, gst or default encoder
        """
        super().__init__()
        self.number_text_tokens = number_text_tokens
        self.start_text_token = start_text_token
        self.stop_text_token = stop_text_token
        self.number_mel_codes = number_mel_codes
        self.start_mel_token = start_mel_token
        self.stop_mel_token = stop_mel_token
        self.layers = layers
        self.heads = heads
        self.max_mel_tokens = max_mel_tokens
        self.max_text_tokens = max_text_tokens
        self.model_dim = model_dim
        self.max_conditioning_inputs = max_conditioning_inputs
        self.mel_length_compression = mel_length_compression
        self.condition_type = condition_type
        self.cond_num = condition_num_latent
        self.cond_mask_pad = nn.ConstantPad1d((self.cond_num, 0), True)
        self.emo_cond_mask_pad = nn.ConstantPad1d((1, 0), True)
        if condition_type == "perceiver":
            self.conditioning_encoder = ConditioningEncoder(1024, model_dim, num_attn_heads=heads)
            self.perceiver_encoder = PerceiverResampler(model_dim, dim_context=model_dim, num_latents=self.cond_num)
        elif condition_type == "conformer_perceiver" or condition_type == "conformer_encoder":
            self.conditioning_encoder = ConformerEncoder(input_size=1024,
                                                         output_size=condition_module['output_size'],
                                                         linear_units=condition_module['linear_units'],
                                                         attention_heads=condition_module['attention_heads'],
                                                         num_blocks=condition_module['num_blocks'],
                                                         input_layer=condition_module['input_layer'])
            if condition_type == "conformer_perceiver":
                self.perceiver_encoder = PerceiverResampler(model_dim, dim_context=condition_module['output_size'],
                                                            ff_mult=condition_module['perceiver_mult'],
                                                            heads=condition_module['attention_heads'],
                                                            num_latents=self.cond_num)
        else:
            self.conditioning_encoder = ConditioningEncoder(1024, model_dim, num_attn_heads=heads, mean=True)

        self.emo_conditioning_encoder = ConformerEncoder(input_size=1024,
                                                         output_size=emo_condition_module['output_size'],
                                                         linear_units=emo_condition_module['linear_units'],
                                                         attention_heads=emo_condition_module['attention_heads'],
                                                         num_blocks=emo_condition_module['num_blocks'],
                                                         input_layer=emo_condition_module['input_layer'])
        self.emo_perceiver_encoder = PerceiverResampler(1024, dim_context=emo_condition_module['output_size'],
                                                            ff_mult=emo_condition_module['perceiver_mult'],
                                                            heads=emo_condition_module['attention_heads'],
                                                            num_latents=1)



        self.text_embedding = nn.Embedding(self.number_text_tokens * types + 1, model_dim)
        self.emo_layer = nn.Linear(model_dim, model_dim)
        self.emovec_layer = nn.Linear(1024, model_dim)

        if use_mel_codes_as_input:
            self.mel_embedding = nn.Embedding(self.number_mel_codes, model_dim)
        else:
            self.mel_embedding = MelEncoder(model_dim, resblocks_per_reduction=1)
        self.gpt, self.mel_pos_embedding, self.text_pos_embedding, self.mel_layer_pos_embedding, self.text_layer_pos_embedding = \
            build_hf_gpt_transformer(layers, model_dim, heads, self.max_mel_tokens + 2 + self.max_conditioning_inputs,
                                     self.max_text_tokens + 2, checkpointing)
        if train_solo_embeddings:
            self.mel_solo_embedding = nn.Parameter(torch.randn(1, 1, model_dim) * .02, requires_grad=True)
            self.text_solo_embedding = nn.Parameter(torch.randn(1, 1, model_dim) * .02, requires_grad=True)
        else:
            self.mel_solo_embedding = 0
            self.text_solo_embedding = 0

        self.final_norm = nn.LayerNorm(model_dim)
        self.text_head = nn.Linear(model_dim, self.number_text_tokens * types + 1)
        self.mel_head = nn.Linear(model_dim, self.number_mel_codes)

        self.speed_emb = nn.Embedding(2, model_dim)
        self.speed_emb.weight.data.normal_(mean=0.0, std=0.0)

        # Initialize the embeddings per the GPT-2 scheme
        embeddings = [self.text_embedding]
        if use_mel_codes_as_input:
            embeddings.append(self.mel_embedding)
        for module in embeddings:
            module.weight.data.normal_(mean=0.0, std=.02)
            
        # init vllm engine
        vllm_dir = os.path.join(model_dir, "vllm")
        engine_args = AsyncEngineArgs(
            model=vllm_dir,
            tensor_parallel_size=1,
            dtype="auto",
            gpu_memory_utilization=gpu_memory_utilization,
            # enforce_eager=True,
        )
        self.llm = AsyncLLMEngine.from_engine_args(engine_args)
        self.sampling_params = SamplingParams(
            temperature=1.0,
            top_p=0.8,
            top_k=30,  # 5, 30
            repetition_penalty=10.0,  # 8.0
            max_tokens=4000,  # 605
            stop_token_ids=[self.stop_mel_token],
            seed=42
        )

    def post_init_gpt2_config(self, use_deepspeed=False, kv_cache=False, half=False):
        seq_length = self.max_mel_tokens + self.max_text_tokens + 2
        gpt_config = GPT2Config(
            vocab_size=self.number_mel_codes,
            n_positions=seq_length,
            n_ctx=seq_length,
            n_embd=self.model_dim,
            n_layer=self.layers,
            n_head=self.heads,
            gradient_checkpointing=False,
            use_cache=True,
        )
        self.inference_model = GPT2InferenceModel(
            gpt_config,
            self.gpt,
            self.mel_pos_embedding,
            self.mel_embedding,
            self.final_norm,
            self.mel_head,
            kv_cache=kv_cache,
        )
        if use_deepspeed and half and torch.cuda.is_available():
            import deepspeed
            self.ds_engine = deepspeed.init_inference(model=self.inference_model,
                                                      mp_size=1,
                                                      replace_with_kernel_inject=True,
                                                      dtype=torch.float16)
            self.inference_model = self.ds_engine.module.eval()
        elif use_deepspeed and torch.cuda.is_available():
            import deepspeed
            self.ds_engine = deepspeed.init_inference(model=self.inference_model,
                                                      mp_size=1,
                                                      replace_with_kernel_inject=True,
                                                      dtype=torch.float32)
            self.inference_model = self.ds_engine.module.eval()
        else:
            self.inference_model = self.inference_model.eval()

        # self.inference_model = PrunedGPT2InferenceModel(gpt_config, self.gpt, self.mel_pos_embedding, self.mel_embedding, self.final_norm, self.mel_head)
        # self.gpt.wte = self.mel_embedding

    def build_aligned_inputs_and_targets(self, input, start_token, stop_token):
        inp = F.pad(input, (1, 0), value=start_token)
        tar = F.pad(input, (0, 1), value=stop_token)
        return inp, tar

    def set_mel_padding(self, mel_input_tokens, mel_lengths):
        """
        Given mel tokens that are derived from a padded audio clip and the actual lengths of each batch element in
        that audio clip, reformats the tokens with STOP_MEL_TOKEN in place of the zero padding. This is required
        preformatting to create a working TTS model.
        """
        for b in range(len(mel_lengths)):
            # Due to the convolutional nature of how these tokens are generated,
            # it would be best if the model predicts a token past the actual last token.
            actual_end = mel_lengths[b]
            if actual_end < mel_input_tokens.shape[-1]:
                mel_input_tokens[b, actual_end:] = self.stop_mel_token
        return mel_input_tokens

    def set_text_padding(self, text_input_tokens, text_lengths):
        """
        Given mel tokens that are derived from a padded audio clip and the actual lengths of each batch element in
        that audio clip, reformats the tokens with STOP_MEL_TOKEN in place of the zero padding. This is required
        preformatting to create a working TTS model.
        """
        for b in range(len(text_lengths)):
            # Due to the convolutional nature of how these tokens are generated,
            # it would be best if the model predicts a token past the actual last token.
            actual_end = text_lengths[b]
            if actual_end < text_input_tokens.shape[-1]:
                text_input_tokens[b, actual_end:] = self.stop_text_token
        return text_input_tokens

    def get_logits(self, speech_conditioning_inputs, first_inputs, first_head, second_inputs=None, second_head=None, get_attns=False, return_latent=False):
        if second_inputs is not None:
            emb = torch.cat([speech_conditioning_inputs, first_inputs, second_inputs], dim=1)
        else:
            emb = torch.cat([speech_conditioning_inputs, first_inputs], dim=1)

        gpt_out = self.gpt(inputs_embeds=emb, return_dict=True, output_attentions=get_attns)
        if get_attns:
            return gpt_out.attentions

        offset = speech_conditioning_inputs.shape[1]
        enc = gpt_out.last_hidden_state[:, offset:]
        enc = self.final_norm(enc)

        if return_latent:
            return enc[:, :first_inputs.shape[1]], enc[:, -second_inputs.shape[1]:]

        first_logits = enc[:, :first_inputs.shape[1]]
        first_logits = first_head(first_logits)
        first_logits = first_logits.permute(0, 2, 1)
        if second_inputs is not None:
            second_logits = enc[:, -second_inputs.shape[1]:]
            second_logits = second_head(second_logits)
            second_logits = second_logits.permute(0, 2, 1)
            return first_logits, second_logits
        else:
            return first_logits

    def get_conditioning(self, speech_conditioning_input, cond_mel_lengths=None):
        if self.condition_type == "perceiver":
            if speech_conditioning_input.ndim == 4:
                speech_conditioning_input = speech_conditioning_input.squeeze(1)
            speech_conditioning_input = self.conditioning_encoder(speech_conditioning_input)  # (b, d, s)
            conds = self.perceiver_encoder(speech_conditioning_input.transpose(1, 2))  # (b, 32, d)
        elif self.condition_type == "conformer_perceiver":
            speech_conditioning_input, mask = self.conditioning_encoder(speech_conditioning_input.transpose(1, 2),
                                                                        cond_mel_lengths)  # (b, s, d), (b, 1, s)
            if self.condition_type == "conformer_perceiver":
                # conds_mask = torch.cat([torch.ones((mask.shape[0], self.cond_num), dtype=torch.bool), mask.squeeze(1)], dim=1)
                conds_mask = self.cond_mask_pad(mask.squeeze(1))
                conds = self.perceiver_encoder(speech_conditioning_input, conds_mask)  # (b, 32, d)
        elif self.condition_type == "gst":
            if speech_conditioning_input.ndim == 4:
                speech_conditioning_input = speech_conditioning_input.squeeze(1)
            conds = self.gst_encoder(speech_conditioning_input.transpose(1, 2))  # (b, 1, d)
        else:
            speech_conditioning_input = (
                speech_conditioning_input.unsqueeze(1)
                if len(speech_conditioning_input.shape) == 3
                else speech_conditioning_input
            )
            conds = []
            for j in range(speech_conditioning_input.shape[1]):
                conds.append(self.conditioning_encoder(speech_conditioning_input[:, j]))
            conds = torch.stack(conds, dim=1)
            conds = conds.mean(dim=1)
            conds = conds.unsqueeze(1)
        return conds


    def get_emo_conditioning(self, speech_conditioning_input, cond_mel_lengths=None):
        speech_conditioning_input, mask = self.emo_conditioning_encoder(speech_conditioning_input.transpose(1, 2),
                                                                        cond_mel_lengths)  # (b, s, d), (b, 1, s)
        conds_mask = self.emo_cond_mask_pad(mask.squeeze(1))
        conds = self.emo_perceiver_encoder(speech_conditioning_input, conds_mask)  # (b, 1, d)
        return conds.squeeze(1)


    def forward(self, speech_conditioning_latent, text_inputs, text_lengths, mel_codes, mel_codes_lengths, emo_speech_conditioning_latent,
                cond_mel_lengths=None, emo_cond_mel_lengths=None, emo_vec=None, use_speed=None, do_spk_cond=False):
        """
        Forward pass that uses both text and voice in either text conditioning mode or voice conditioning mode

        speech_conditioning_input: MEL float tensor, (b,1024)
        text_inputs: long tensor, (b,t)
        text_lengths: long tensor, (b,)
        mel_inputs:  long tensor, (b,m)
        wav_lengths: long tensor, (b,)

        If return_attentions is specified, only logits are returned.
        If return_latent is specified, loss & logits are not computed or returned. Only the predicted latents are returned.
        """

        if do_spk_cond:
            speech_conditioning_latent = self.get_conditioning(speech_conditioning_latent.transpose(1,2), cond_mel_lengths)
        else:
            speech_conditioning_latent = speech_conditioning_latent

        if emo_vec is None:
            emo_vec_syn_ori = self.get_emo_conditioning(emo_speech_conditioning_latent.transpose(1,2), emo_cond_mel_lengths)
            emo_vec_syn = self.emovec_layer(emo_vec_syn_ori)
            emo_vec = self.emo_layer(emo_vec_syn)

        text_inputs = self.set_text_padding(text_inputs, text_lengths)
        text_inputs = F.pad(text_inputs, (0, 1), value=self.stop_text_token)

        mel_codes = self.set_mel_padding(mel_codes, mel_codes_lengths)
        mel_codes = F.pad(mel_codes, (0, 1), value=self.stop_mel_token)

        duration_emb = self.speed_emb(torch.zeros_like(use_speed))
        duration_emb_half = self.speed_emb(torch.ones_like(use_speed))
        conds = torch.cat((speech_conditioning_latent + emo_vec.unsqueeze(1), duration_emb_half.unsqueeze(1), duration_emb.unsqueeze(1)), 1)
        text_inputs, text_targets = self.build_aligned_inputs_and_targets(text_inputs, self.start_text_token, self.stop_text_token)
        text_emb = self.text_embedding(text_inputs) + self.text_pos_embedding(text_inputs)
        mel_codes, mel_targets = self.build_aligned_inputs_and_targets(mel_codes, self.start_mel_token, self.stop_mel_token)

        mel_emb = self.mel_embedding(mel_codes)
        mel_emb = mel_emb + self.mel_pos_embedding(mel_codes)

        text_logits, mel_logits = self.get_logits(conds, text_emb, self.text_head, mel_emb, self.mel_head, get_attns=False, return_latent=True)
        return mel_logits[:, :-2]  # Despite the name, these are not logits. Strip off the two tokens added by this forward pass.

    def prepare_gpt_inputs(
        self,
        conditional_latents: torch.Tensor,
        text_inputs: torch.Tensor,
    ):
        
        """
        Prepare the inputs for the GPT2InferenceModel to generate.
        Args:
            conds_latent: (b, 32, dim) audio conditioning embedding by `get_conditioning()`
            text_inputs: (b, L)
        Returns:
            input_ids: (b, s+1) the input ids for the GPT2InferenceModel.generate()
            inputs_embeds: (b, s+1, dim) the input embeddings for the GPT2InferenceModel.forward()
            attention_mask: (b, s+1) the attention mask for the GPT2InferenceModel.generate()
        """
        b, L = text_inputs.shape[:2]
        device = text_inputs.device
        single_cond = conditional_latents.ndim == 3 and conditional_latents.shape[0] == 1
        if not single_cond:
            assert conditional_latents.shape[0] == b, f"batch size mismatch: {conditional_latents.shape[0]} vs {b}"
        batched_mel_emb = []
        attention_masks = []
        target_len = conditional_latents.shape[1] + L + 2
        for i in range(b):
            valid_mask = (text_inputs[i] != self.stop_text_token) & (text_inputs[i] != self.start_text_token)
            text_input = text_inputs[i][valid_mask]
            text_input = F.pad(text_input, (1, 0), value=self.start_text_token)
            text_input = F.pad(text_input, (0, 1), value=self.stop_text_token)
            text_input_pos = torch.arange(0, text_input.size(-1), device=device)
            text_emb = self.text_embedding(text_input) + self.text_pos_embedding.emb(text_input_pos)
            # concatenate [conditional latents][text embeddings]
            conds_text_emb = [
                conditional_latents.squeeze(0) if single_cond else conditional_latents[i],
                text_emb,
            ]
            # +1 for the start_mel_token
            attention_mask = torch.ones(target_len+1, dtype=torch.long, device=device)
            # check this text input is padded
            padding: int = L + 2 - text_input.size(-1)
            # pad left of [cond][text] -> [pad][cond][text]
            if padding > 0:
                pad = torch.zeros((padding, conditional_latents.size(-1)), dtype=text_emb.dtype, device=device) # [p, dim]
                conds_text_emb.insert(0, pad)
                attention_mask[:padding] = 0
            mel_emb = torch.cat(conds_text_emb) #[s, dim]
            assert mel_emb.shape[0] == target_len, f"mel_emb.shape: {mel_emb.shape}, target_len: {target_len}"
            batched_mel_emb.append(mel_emb)
            attention_masks.append(attention_mask)
        # [b, s, dim]
        batched_mel_emb = torch.stack(batched_mel_emb, dim=0)
        # [b, s+1]
        attention_mask = torch.stack(attention_masks, dim=0)
        # [b, s+1]
        fake_inputs = torch.ones(
            (
                batched_mel_emb.shape[0],
                batched_mel_emb.shape[1] + 1,  # +1 for the start_mel_token
            ),
            dtype=torch.long,
            device=device,
        )
        fake_inputs[:, -1] = self.start_mel_token
        return fake_inputs, batched_mel_emb, attention_mask

    def inference_speech(self, speech_condition, text_inputs, emo_speech_condition=None, cond_lengths=None, emo_cond_lengths=None, emo_vec=None, use_speed=False, input_tokens=None, num_return_sequences=1,
                         max_generate_length=None, typical_sampling=False, typical_mass=.9, **hf_generate_kwargs):
        """
        Args:
            speech_condition: (b, d, frames) or (d, frames)
            text_inputs: (b, L)
            cond_mel_lengths: lengths of the conditioning mel spectrograms in shape (b,) or (1,)
            input_tokens: additional tokens for generation in shape (b, s) or (s,)
            max_generate_length: limit the number of generated tokens
            hf_generate_kwargs: kwargs for `GPT2InferenceModel.generate(**hf_generate_kwargs)`
        """

        if speech_condition.ndim == 2:
            speech_condition = speech_condition.unsqueeze(0)
        if emo_speech_condition is None:
            emo_speech_condition = speech_condition
        if cond_lengths is None:
            cond_lengths = torch.tensor([speech_condition.shape[-1]], device=speech_condition.device)
        if emo_cond_lengths is None:
            emo_cond_lengths = torch.tensor([emo_speech_condition.shape[-1]], device=speech_condition.device) 

        speech_conditioning_latent = self.get_conditioning(speech_condition.transpose(1,2), cond_lengths)
        if emo_vec is None:
            print('compute emo vec')
            emo_vec = self.get_emo_conditioning(emo_speech_condition.transpose(1,2), emo_cond_lengths)
            emo_vec = self.emovec_layer(emo_vec)
            emo_vec = self.emo_layer(emo_vec)
        else:
            print('Use the specified emotion vector')

        tmp = torch.zeros(text_inputs.size(0)).to(text_inputs.device)
        duration_emb =  self.speed_emb(torch.zeros_like(tmp).long())
        duration_emb_half = self.speed_emb(torch.ones_like(tmp).long())
        conds_latent = torch.cat((speech_conditioning_latent + emo_vec.unsqueeze(1), duration_emb_half.unsqueeze(1), duration_emb.unsqueeze(1)), 1)
        input_ids, inputs_embeds, attention_mask = self.prepare_gpt_inputs(conds_latent, text_inputs)
        self.inference_model.store_mel_emb(inputs_embeds)
        if input_tokens is None:
            inputs = input_ids
        else:
            if input_tokens.ndim == 1:
                input_tokens = input_tokens.unsqueeze(0)
            assert num_return_sequences % input_tokens.shape[0] == 0, \
                    "The num_return_sequences must be divisible by the batch number of input_tokens"
            assert num_return_sequences % text_inputs.shape[0] == 0, \
                    "The num_return_sequences must be divisible by the batch number of text_inputs"
            b = num_return_sequences // input_ids.shape[0]
            if b > 1:
                input_ids = input_ids.repeat(b, 1)
                attention_mask = attention_mask.repeat(b, 1)
            input_tokens = input_tokens.repeat(num_return_sequences // input_tokens.shape[0], 1)
            inputs = torch.cat([input_ids, input_tokens], dim=1)
            attention_mask = F.pad(attention_mask, (0, input_tokens.shape[1]), value=1)
        trunc_index = inputs.shape[1]
        logits_processor = LogitsProcessorList()
        if typical_sampling:
            # employ custom typical sampling
            if not (typical_mass > 0.0 and typical_mass < 1.0):
                raise ValueError(f"`typical_mass` has to be a float > 0 and < 1, but is {typical_mass}")
            min_tokens_to_keep = 2 if hf_generate_kwargs.get("num_beams", 1) > 1 else 1
            logits_processor.append(TypicalLogitsWarper(mass=typical_mass, min_tokens_to_keep=min_tokens_to_keep))
        max_length = (trunc_index + self.max_mel_tokens - 1) if max_generate_length is None else trunc_index + max_generate_length
        output = self.inference_model.generate(inputs, 
                                            bos_token_id=self.start_mel_token, pad_token_id=self.stop_mel_token,
                                            eos_token_id=self.stop_mel_token, attention_mask=attention_mask,
                                            max_length=max_length, logits_processor=logits_processor,
                                            num_return_sequences=num_return_sequences,
                                            **hf_generate_kwargs)
        if isinstance(output, torch.Tensor):
            return output[:, trunc_index:], speech_conditioning_latent
        # GenerateOutput
        output.sequences = output.sequences[:, trunc_index:]
        return output, speech_conditioning_latent

    async def inference_speech_vllm(self, speech_condition, text_inputs, emo_speech_condition=None, cond_lengths=None, emo_cond_lengths=None, emo_vec=None, num_return_sequences=1,
                                 typical_sampling=False):
        """
        使用 VLLM 加速的推理函数。

        Args:
            speech_condition: (b, d, frames) or (d, frames)
            text_inputs: (b, L) or List[(1, L)]
            cond_mel_lengths: conditioning mel spectrograms的长度，形状为 (b,) or (1,)
            emo_speech_condition: (b, d, frames) or (d, frames)
            emo_cond_lengths: emotion conditioning mel spectrograms的长度，形状为 (b,) or (1,)
            emo_vec: 预先计算好的emotion向量
            max_generate_length: 限制生成的token数量
            typical_sampling: 是否启用typical sampling
            typical_mass: typical sampling的质量参数
            hf_generate_kwargs: 其他HuggingFace generate参数将被转换为SamplingParams
        """

        # --- 1. 准备共享的Conditioning Latents ---
        if speech_condition.ndim == 2:
            speech_condition = speech_condition.unsqueeze(0)
        if emo_speech_condition is None:
            emo_speech_condition = speech_condition
        if cond_lengths is None:
            cond_lengths = torch.tensor([speech_condition.shape[-1]], device=speech_condition.device)
        if emo_cond_lengths is None:
            emo_cond_lengths = torch.tensor([emo_speech_condition.shape[-1]], device=speech_condition.device)

        speech_conditioning_latent = self.get_conditioning(speech_condition.transpose(1, 2), cond_lengths)

        if emo_vec is None:
            print('计算情感向量')
            emo_vec = self.get_emo_conditioning(emo_speech_condition.transpose(1, 2), emo_cond_lengths)
            emo_vec = self.emovec_layer(emo_vec)
            emo_vec = self.emo_layer(emo_vec)
        else:
            print('使用指定的情感向量')

        # --- 2. 设置 VLLM 采样参数 ---
        # 从hf_generate_kwargs转换参数
        # 注意: VLLM可能不支持所有HF参数，这里列出一些常见的
        # sampling_params = SamplingParams(
        #     n=num_return_sequences,
        #     temperature=hf_generate_kwargs.get("temperature", 1.0),
        #     top_p=hf_generate_kwargs.get("top_p", 1.0),
        #     top_k=hf_generate_kwargs.get("top_k", -1),
        #     use_beam_search=hf_generate_kwargs.get("num_beams", 1) > 1,
        #     stop_token_ids=[self.stop_mel_token],
        #     max_tokens=self.max_mel_tokens - 1 if max_generate_length is None else max_generate_length,
        # )
        
        
        # VLLM目前没有直接的typical_sampling参数，
        # 如果需要此功能，可能需要自定义LogitsProcessor或在VLLM层面实现
        if typical_sampling:
            print("警告: VLLM原生不支持typical_sampling，此设置将被忽略。")


        # --- 3. 定义单个请求的生成函数 ---
        async def _generate_one(text_input_tensor, speech_latent_single, emo_vec_single):
            # 将输入张量调整为batch size为1的形状
            if text_input_tensor.ndim == 1:
                text_input_tensor = text_input_tensor.unsqueeze(0)
            
            # --- a. 准备单个输入的Embeddings ---
            tmp = torch.zeros(text_input_tensor.size(0)).to(text_input_tensor.device)
            duration_emb = self.speed_emb(torch.zeros_like(tmp).long())
            duration_emb_half = self.speed_emb(torch.ones_like(tmp).long())
            
            # 合并latents
            # conds_latent = torch.cat((speech_latent_single + emo_vec_single.unsqueeze(1), duration_emb_half.unsqueeze(1), duration_emb.unsqueeze(1)), 1)
            conds_latent = torch.cat((speech_latent_single + emo_vec_single, duration_emb_half.unsqueeze(1), duration_emb.unsqueeze(1)), 1)

            # 准备GPT输入，得到inputs_embeds
            # 注意：这里的 `prepare_gpt_inputs` 应该被调整为只返回 `inputs_embeds`
            # 如果它还返回input_ids和attention_mask，我们需要忽略它们
            _, inputs_embeds, _ = self.prepare_gpt_inputs(conds_latent, text_input_tensor)
            
            # --- b. 为VLLM创建Prompt ---
            # VLLM需要知道输入的长度，但不需要真实的token_ids，因为我们提供了embeddings
            fake_inputs = list(range(inputs_embeds.shape[1]))
            multi_modal_data = {"image": inputs_embeds} # "image"是vllm多模态的通用键
            tokens_prompt = TokensPrompt(prompt_token_ids=fake_inputs, multi_modal_data=multi_modal_data)

            # --- c. 生成输出 ---
            request_id = f"req-{uuid.uuid4()}"
            output_generator = self.llm.generate(tokens_prompt, self.sampling_params, request_id)
            
            final_output = None
            async for output in output_generator:
                final_output = output
            
            # 提取生成的token ID
            # 我们不返回第一个pass的latent
            generated_codes = [seq.token_ids for seq in final_output.outputs]

            return generated_codes, speech_latent_single

        # --- 4. 并发处理所有输入 ---
        if not isinstance(text_inputs, list):
             # 如果输入不是列表（例如，单个张量），将其转换为列表以便统一处理
             text_inputs = [text_inputs[i:i+1] for i in range(text_inputs.shape[0])]

        # 为每个输入创建一个并发任务
        tasks = []
        for i, text_tensor in enumerate(text_inputs):
             # 每个任务获取其对应的latent切片
            speech_latent_slice = speech_conditioning_latent[i:i+1]
            emo_vec_slice = emo_vec[i:i+1]
            tasks.append(_generate_one(text_tensor, speech_latent_slice, emo_vec_slice))
        
        # 并发执行并等待所有任务完成
        all_outputs = await asyncio.gather(*tasks)
        
        # 根据num_return_sequences调整返回结构
        # all_outputs 的结构是 [(codes_list, latent), (codes_list, latent), ...]
        if num_return_sequences == 1:
            # 如果每个输入只返回一个序列，简化输出
            results = [item[0][0] for item in all_outputs] # [codes_tensor1, codes_tensor2, ...]
            latents = torch.cat([item[1] for item in all_outputs], dim=0)
            return results, latents
        else:
            # 否则，保持嵌套列表的结构
            return all_outputs

    def get_emovec(self, emo_speech_conditioning_latent, emo_cond_lengths):
        emo_vec_syn_ori = self.get_emo_conditioning(emo_speech_conditioning_latent.transpose(1,2), emo_cond_lengths)
        emo_vec_syn = self.emovec_layer(emo_vec_syn_ori)
        emo_vec = self.emo_layer(emo_vec_syn)
        return emo_vec

    def merge_emovec(self, speech_conditioning_latent, emo_speech_conditioning_latent, cond_lengths, emo_cond_lengths, alpha = 1.0):
        emo_vec = self.get_emovec(emo_speech_conditioning_latent, emo_cond_lengths)
        base_vec = self.get_emovec(speech_conditioning_latent, cond_lengths)

        out = base_vec + alpha * (emo_vec - base_vec)
        return out
