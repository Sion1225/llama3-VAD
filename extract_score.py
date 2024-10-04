from typing import List

import fire
import pandas as pd

from llama import Llama


def main(
    ckpt_path: str,
    tokenizer_path: str,
    max_seq_len: int = 128,
    max_gen_len: int = 64,
    max_batch_size: int = 4,
):
    
    generator = Llama.build(
        ckpt_path=ckpt_path,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    EMOBANK_PATH = "/experiment_data/emobank.csv"
    emobank = pd.read_csv(EMOBANK_PATH, na_values=[], keep_default_na=False)

    prompts: List[str] = emobank['text'].tolist()

    final_attention_scores = generator.extract_context_vector(prompts)

    with open('final_attention_scores.txt', 'w') as f:
        for promt, context_vector in zip(prompts, final_attention_scores):
            f.write(f'{promt}\n{context_vector}\n\n')

if __name__ == "__main__":
    fire.Fire(main)