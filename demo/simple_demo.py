import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

# 注意！目录不能含有"."，可以替换成"_"
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default="", type=str, help="")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map='auto')
model = model.eval()
print('model loaded', args.model_path, model.device)

system_message = "你是由哔哩哔哩人工智能平台部自主研发的大语言模型，名为“Index”。你能够根据用户传入的信息，帮助用户完成指定的任务，并生成恰当的、符合要求的回复。"
query = "续写 天不生我金坷垃"
model_input = []
model_input.append({"role": "system", "content": system_message})
model_input.append({"role": "user", "content": query})

inputs = tokenizer.apply_chat_template(model_input, tokenize=False, add_generation_prompt=False)
input_ids = tokenizer.encode(inputs, return_tensors="pt").to(model.device)
history_outputs = model.generate(input_ids, max_new_tokens=300, top_k=5, top_p=0.8, temperature=0.3, repetition_penalty=1.1, do_sample=True)

# 删除</s>
if history_outputs[0][-1] == 2:
    history_outputs = history_outputs[:, :-1]

outputs = history_outputs[0][len(input_ids[0]):]
print('User:', query)
print('\nModel:', tokenizer.decode(outputs))

