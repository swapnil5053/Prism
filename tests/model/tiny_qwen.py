"""Build a tiny, randomly initialised Qwen2.5-VL checkpoint on disk.

It produces gibberish, but it exercises the real loading, preprocessing,
generation and stopping code paths without downloading 7 GB of weights.
"""

from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    PreTrainedTokenizerFast,
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
    Qwen2VLImageProcessor,
    Qwen2VLVideoProcessor,
)

SPECIAL = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|image_pad|>",
    "<|video_pad|>",
]
WORDS = ["[", "]", "{", "}", ",", ":", '"', "bbox_2d", "label", "button", "text", "0", "1", "2"]

CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n"
    "{% for c in m['content'] %}{% if c['type'] == 'image' %}"
    "<|vision_start|><|image_pad|><|vision_end|>"
    "{% else %}{{ c['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def build(path: Path) -> Path:
    vocab = {tok: i for i, tok in enumerate(SPECIAL + WORDS + ["<unk>"])}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="<unk>",
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        additional_special_tokens=SPECIAL,
    )
    image_processor = Qwen2VLImageProcessor(min_pixels=56 * 56, max_pixels=112 * 112)
    processor = Qwen2_5_VLProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        video_processor=Qwen2VLVideoProcessor(),
        chat_template=CHAT_TEMPLATE,
    )

    config = Qwen2_5_VLConfig(
        text_config={
            "vocab_size": len(vocab),
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "rope_scaling": {"type": "mrope", "mrope_section": [1, 1, 2]},
            "bos_token_id": vocab["<|endoftext|>"],
            "eos_token_id": vocab["<|im_end|>"],
            "pad_token_id": vocab["<|endoftext|>"],
        },
        vision_config={
            "depth": 2,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "out_hidden_size": 32,
            "fullatt_block_indexes": [1],
            "window_size": 56,
        },
        image_token_id=vocab["<|image_pad|>"],
        video_token_id=vocab["<|video_pad|>"],
        vision_start_token_id=vocab["<|vision_start|>"],
        vision_end_token_id=vocab["<|vision_end|>"],
        eos_token_id=vocab["<|im_end|>"],
        pad_token_id=vocab["<|endoftext|>"],
    )
    model = Qwen2_5_VLForConditionalGeneration(config)
    model.save_pretrained(path)
    processor.save_pretrained(path)
    return path
