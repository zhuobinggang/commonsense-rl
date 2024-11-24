# 2024.11.24测试使用t5 large来压缩环境描述
from transformers import T5Tokenizer, T5ForConditionalGeneration
from functools import lru_cache

@lru_cache(maxsize=None)
def get_client():
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large", device_map="auto")
    return model, tokenizer


def ask(input_text = "translate English to German: How old are you?"):
    model, tokenizer = get_client()
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
    outputs = model.generate(input_ids)
    return tokenizer.decode(outputs[0])


