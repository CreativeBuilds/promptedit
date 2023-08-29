import torch
from transformers import AutoModelForCausalLM, AutoTokenizer



model = AutoModelForCausalLM.from_pretrained(
    'cerebras/btlm-3b-8k-base', 
    load_in_4bit=True, 
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

# check we are in 4bit
print(model.transformer.h[3].attn.c_attn)