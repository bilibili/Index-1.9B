from flask import Flask, request, jsonify
import torch
import argparse
from src.logger import LoggerFactory
from src.utils import decode_csv_to_json, load_json, save_to_json
from src.prompt_concat import GetManualTestSamples, CreateTestDataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import json
import os
import logging

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

model_name = "Index-1.9B-Character"
parser = argparse.ArgumentParser()
parser.add_argument('--role_name', type=str, required=True, help="")
parser.add_argument('--role_description', type=str, help="")
parser.add_argument('--role_dialog', type=str, help="")

args = parser.parse_args()

with open('config/config.json') as f:
    config_data = json.load(f)
    huggingface_local_path = config_data['huggingface_local_path']
model_path = huggingface_local_path

# load model
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)

# create new role
role_name = args.role_name
role_info = args.role_description
role_dialog_file = args.role_dialog
json_output_path = f"./character/{role_name}.json"
if role_name not in ['三三']:
    decode_csv_to_json(role_dialog_file, role_name, role_info, json_output_path)

class IndexRolePlay:
    def __init__(self, role_name, role_info=None, role_dialog_file=None):

        self.role_name = role_name
        self.role_info = role_info
        self.role_dialog_file = role_dialog_file
        self.role_processed_file = f"./character/{self.role_name}_测试问题.json"

        self.json_output_path = f"./character/{self.role_name}.json"

        self.save_samples_dir = "./character"
        self.save_samples_path = self.role_name + "_rag.json"
        self.prompt_path = "./prompt/dataset_character.txt"

    def generate_with_question(self, question):
        question_in = question

        g = GetManualTestSamples(
            role_name=self.role_name,
            role_data_path=self.json_output_path,
            save_samples_dir=self.save_samples_dir,
            save_samples_path=self.save_samples_path,
            prompt_path=self.prompt_path,
            max_seq_len=4000
        )
        g.get_qa_samples_by_query(
            questions_query=question_in,
            keep_retrieve_results_flag=True
        )

    def create_datasets(self):
        testset = []
        role_samples_path = os.path.join(self.save_samples_dir, self.save_samples_path)

        c = CreateTestDataset(role_name=self.role_name,
                              role_samples_path=role_samples_path,
                              role_data_path=role_samples_path,
                              prompt_path=self.prompt_path
                              )
        res = c.load_samples()
        testset.extend(res)
        save_to_json(testset, f"{self.save_samples_dir}/{self.role_name}_测试问题.json")

    def infer_with_question(self, question, generate_config):
        question_list = []
        for chat in question:
            if chat["role"] == "user":
                question_list.append(f"user:{chat['content']}")
            elif chat["role"] == "assistant":
                question_list.append(f"{self.role_name}:{chat['content']}")
        concat_question = "\n".join(question_list)

        self.generate_with_question(concat_question)
        self.create_datasets()

        json_data = load_json(f"{self.save_samples_dir}/{self.role_name}_测试问题.json")
        for i in json_data:
            text = i['input_text']
            # 使用分词器对文本进行编码
            inputs = tokenizer(text, return_tensors="pt").to("cuda:0")

            # 将输入数据传递给模型
            outputs = model.generate(**inputs, **generate_config)

            # 打印模型输出
            res = tokenizer.batch_decode(outputs)[0].replace("</s>","")
            return res[len(text):]

@app.route('/v1/character/chat/completions', methods=['POST'])#URL: http://127.0.0.1:8010/v1/character/chat/completions
def chat_completion():
    try:
    # Parse incoming JSON data
        data = request.get_json()
        messages = data.get('messages', [])
        is_streaming = data.get('stream', False)
        global role_name

        generate_config = {
            "max_new_tokens": data.get('max_new_tokens', 1024),
            "top_k": data.get('top_k', 5),
            "top_p": data.get('top_p', 0.8),
            "temperature": data.get('temperature', 0.1),
            "repetition_penalty": data.get('repetition_penalty', 1.1),
            "do_sample": data.get('do_sample', True),
            "num_beams": data.get('num_beams', 1)
        }

        # Check if streaming is enabled
        if is_streaming:#这里暂时只支持非流式输出
            return jsonify({"error": "Streaming is not supported."}), 400


        # Generate response using the model
        response_text = generate_response(role_name, messages, generate_config)

        response_data = {
            "object": "chat.completion",
            "model": model_name,
            "choices": [],
        }

        response_data["choices"].append({"message": {"role": "assistant", "content": response_text}, "finish_reason": "stop"})

        # Build the response
        prompt_tokens = sum(len(tokenizer.encode(msg['content'])) for msg in messages)
        completion_tokens = sum(len(msg) for msg in response_text)
        total_tokens = prompt_tokens + completion_tokens

        response_data["usage"]={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }

        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def generate_response(role_name, messages, generate_config):
    chatbox = IndexRolePlay(role_name=role_name)
    return chatbox.infer_with_question(messages,generate_config)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8010)