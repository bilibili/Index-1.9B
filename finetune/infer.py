

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

mode_path = "your model path"
lora_path = "your lora model path"

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(mode_path, trust_remote_code=True)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(mode_path, 
                                             device_map="auto",
                                             torch_dtype=torch.bfloat16, 
                                             trust_remote_code=True).eval()

# 加载lora权重
model = PeftModel.from_pretrained(model, model_id=lora_path)

system = "your system_message"
prompt = "your input_text"
message = [
    {"role": "system", "content": system},
    {"role": "user", "content": prompt}
]
inputs = tokenizer.apply_chat_template(message,
                                       add_generation_prompt=True,
                                       tokenize=True,
                                       return_tensors="pt",
                                       return_dict=True
                                       ).to('cuda')

print(tokenizer.decode(inputs.input_ids[0]))


gen_kwargs = {"max_length": 32, "do_sample": True, "top_k": 1, "repetition_penalty": 1.1}
with torch.no_grad():
    outputs = model.generate(**inputs, **gen_kwargs)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

