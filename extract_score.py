from typing import List
import os

import fire
import pandas as pd

from llama import Llama, Tokenizer


def reset_attention_cache(module, input, output):
    if hasattr(module, "cache_k"):
        module.cache_k.zero_()
    if hasattr(module, "cache_v"):
        module.cache_v.zero_()

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    max_seq_len: int = 128,
    max_gen_len: int = 64,
    max_batch_size: int = 4,
):
    
    print(f"Working directory: {os.getcwd()}")
    
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    EMOBANK_PATH = "experiment_data/emobank.csv"
    emobank = pd.read_csv(EMOBANK_PATH, na_values=[], keep_default_na=False)

    prompts: List[str] = emobank['text'].tolist()
    prompts_len = len(prompts)

    for layer in generator.model.layers:
        layer.attention.register_forward_hook(reset_attention_cache)

    final_attention_scores = []
    prompt_tokens = []
    for i in range(0, prompts_len, max_batch_size):
        batch_prompts = prompts[i:i+max_batch_size]

        batch_attention_scores = generator.extract_attention_scores(batch_prompts)
        batch_prompts_tokens = generator.prompt_tokens
        final_attention_scores.extend(batch_attention_scores)
        prompt_tokens.extend(batch_prompts_tokens)

    with open('final_attention_scores.txt', 'w') as f, open('softmax_attention_scores.txt', 'w') as f_softmax:
        for prompt, tokens, scores in zip(prompts, prompt_tokens, final_attention_scores):
            f.write(f'Prompt: {prompt}\n')
            f.write(f'Token Length: {len(tokens)}\n')
            f.write(f'Index of Tokens: {tokens}\n')
            f.write(f'Tokens: {Tokenizer.decode(tokens)}\n')

            f_softmax.write(f'Prompt: {prompt}\n')
            f_softmax.write(f'Token Length: {len(tokens)}\n')
            f_softmax.write(f'Index of Tokens: {tokens}\n')
            f_softmax.write(f'Tokens: {Tokenizer.decode(tokens)}\n')
            
            truncated_scores = scores[:, :len(tokens)]

            for head, score_vector in enumerate(truncated_scores):
                f.write(f'Head {head} Attention Scores: {score_vector.tolist()}\n')
            f.write('\n')

            for head, score_vector in enumerate(truncated_scores):
                softmax_score_vector = score_vector.softmax(dim=-1)
                f_softmax.write(f'Head {head} Attention Scores: {softmax_score_vector.tolist()}\n')
            f_softmax.write('\n')


if __name__ == "__main__":
    fire.Fire(main)