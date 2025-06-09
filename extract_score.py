from typing import List
import os
import re
import json

import fire
import pandas as pd

from llama import Llama, Tokenizer

EMOBANK_PATH = "experiment_data/raw_data/emobank.csv"
#DAILYDAILOG_PATH = "experiment_data/raw_data/dialogues_text.txt"
EMPATHETIC_DIALOGUE_PATH = "experiment_data/raw_data/chat_empd.jsonl"


def reset_attention_cache(module, input, output):
    if hasattr(module, "cache_k"):
        module.cache_k.zero_()
    if hasattr(module, "cache_v"):
        module.cache_v.zero_()

# For Preprocessing
def remove_soft_hyphen(text):
    text = text.replace('\xad', '')
    text = re.sub(r'[\u00A0\u2000-\u200B\u202F\u2060]', ' ', text)
    return text

def output_attention_scores(output_path, tokenizer_path, prompts, prompt_tokens, final_attention_scores):
    def to_tokens(tokenizer, token_ids):
        return [tokenizer.decode([t]) for t in token_ids]
    
    tokenizer = Tokenizer(model_path=tokenizer_path)

    with open(os.path.join(output_path, 'final_attention_scores.jsonl'), 'a') as f_final:
        for prompt, tokens, scores in zip(prompts, prompt_tokens, final_attention_scores):
            tokens_list = to_tokens(tokenizer, tokens)
            token_len = len(tokens)

            # 기본 정보
            base_info = {
                "prompt": prompt,
                "token_length": token_len,
                "token_ids": tokens,
                "tokens": tokens_list,
            }

            # Final attention score 저장
            truncated_scores = scores[:, :token_len]
            base_info["attention_scores"] = {
                f"head_{i}": truncated_scores[i].tolist()
                for i in range(truncated_scores.size(0))
            }
            f_final.write(json.dumps(base_info, ensure_ascii=False) + '\n')

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    max_seq_len: int = 128,
    max_gen_len: int = 64,
    max_batch_size: int = 4,
    target_data: str = "emobank",
    output_path: str = "experiment_data/output_data",
):
    
    print(f"Working directory: {os.getcwd()}")
    
    # Model
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    for layer in generator.model.layers:
        layer.attention.register_forward_hook(reset_attention_cache)

    final_attention_scores = []
    prompt_tokens = []

    open(os.path.join(output_path, 'final_attention_scores.jsonl'), 'w').close()  # Clear the file before writing

    # Process the data
    if target_data == "emobank":
        emobank = pd.read_csv(EMOBANK_PATH, na_values=[], keep_default_na=False)

        prompts: List[str] = emobank['text'].tolist()
        prompts_len = len(prompts)
        batch_prompts_tokens = []

        for i in range(0, prompts_len, max_batch_size):
            batch_prompts = prompts[i:i+max_batch_size]
            batch_prompts = [remove_soft_hyphen(prompt) for prompt in batch_prompts]

            batch_attention_scores, prompt_tokens = generator.extract_from_text_prompts(batch_prompts)
            final_attention_scores.extend(batch_attention_scores)
            batch_prompts_tokens.extend(prompt_tokens)

        output_attention_scores(output_path, tokenizer_path, prompts, batch_prompts_tokens, final_attention_scores)

    elif target_data == "empathetic_dialogue":
        count = 0
        batch_prompts: List[str] = []
        batch_prompts_tokens = []
        with open(EMPATHETIC_DIALOGUE_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                batch_prompts.append(json.loads(line))
                count += 1

                if count >= max_batch_size:
                    for dialog in batch_prompts:
                        for utter in dialog:
                            utter["content"] = remove_soft_hyphen(utter["content"])

                    batch_attention_scores, prompt_tokens = generator.extract_from_dialog_prompts(batch_prompts)

                    output_attention_scores(output_path, tokenizer_path, batch_prompts, batch_prompts_tokens, batch_attention_scores)

                    batch_prompts = []
                    batch_prompts_tokens = []
                    count = 0
            
            # 파일을 다 읽어오고 난 시점에서도 batch_prompts가 남아있을것이기에 처리를하고 출력까지 해야함.
            batch_prompts = [remove_soft_hyphen(prompt) for prompt in batch_prompts]

            batch_attention_scores, prompt_tokens = generator.extract_from_dialog_prompts(batch_prompts)

            output_attention_scores(output_path, tokenizer_path, batch_prompts, batch_prompts_tokens, batch_attention_scores)
    else:
        raise("invalid data string")
 


if __name__ == "__main__":
    fire.Fire(main)