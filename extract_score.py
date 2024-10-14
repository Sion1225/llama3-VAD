from typing import List
import os

import fire
import pandas as pd

from llama import Llama


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

        batch_attention_scores = generator.extract_attention_metrics(batch_prompts)
        batch_prompts_tokens = generator.prompt_tokens
        final_attention_scores.extend(batch_attention_scores)
        prompt_tokens.extend(batch_prompts_tokens)

    with open('final_attention_scores.txt', 'w') as f:
        for promt, context_vector in zip(prompts, prompt_tokens, final_attention_scores):
            f.write(f'{promt}\n{len(prompt_tokens)}\n{prompt_tokens}\n{context_vector}\n\n')

if __name__ == "__main__":
    fire.Fire(main)