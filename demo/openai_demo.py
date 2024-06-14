from flask import Flask, request, jsonify
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

model_name = "Index-1.9B"
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default="IndexTeam/Index-1.9B-Chat", type=str, help="")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)

@app.route('/v1/chat/completions', methods=['POST'])#URL: http://127.0.0.1:8010/v1/chat/completions
def chat_completion():
    try:
    # Parse incoming JSON data
        data = request.get_json()
        messages = data.get('messages', [])
        is_streaming = data.get('stream', False)

        generate_config = {
            "max_new_tokens": data.get('max_new_tokens', 1024),
            "top_k": data.get('top_k', 5),
            "top_p": data.get('top_p', 0.8),
            "temperature": data.get('temperature', 0.1),
            "repetition_penalty": data.get('repetition_penalty', 1.1),
            "do_sample": data.get('do_sample', False),
            "num_beams": data.get('num_beams', 1)
        }

        # Check if streaming is enabled
        if is_streaming:#这里暂时只支持非流式输出
            return jsonify({"error": "Streaming is not supported."}), 400

        # Generate response using the model
        response_text = generate_response(messages, generate_config)

        # Calculate token counts
        prompt_tokens = sum(len(tokenizer.encode(msg['content'])) for msg in messages)
        # 暂时不支持调节其他参数
        completion_tokens = len(response_text)
        total_tokens = prompt_tokens + completion_tokens

        # Build the response
        response_data = {
            "object": "chat.completion",
            "model": model_name,
            "choices": [{"message": {"role": "assistant", "content": response_text}, "index": 0, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
        }

        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def generate_response(messages, generate_config):
    # Generate response using the model
    inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    enc = tokenizer.encode(inputs, return_tensors="pt").to(model.device)

    if generate_config['do_sample']:
        history_outputs = model.generate(enc, 
                        max_new_tokens=generate_config['max_new_tokens'], 
                        top_k=generate_config['top_k'], 
                        top_p=generate_config['top_p'], 
                        temperature=generate_config['temperature'], 
                        repetition_penalty=generate_config['repetition_penalty'], 
                        do_sample=generate_config['do_sample'])
    else:
        history_outputs = model.generate(enc, 
                        max_new_tokens=generate_config['max_new_tokens'], 
                        num_beams=generate_config['num_beams'], 
                        do_sample=generate_config['do_sample'],
                        num_return_sequences=1)    

    # 删除</s>
    if history_outputs[0][-1] == 2:
        history_outputs = history_outputs[:, :-1]

    response = history_outputs[0][len(enc[0]):]

    return tokenizer.decode(response)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8010)