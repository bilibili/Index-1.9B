# coding=utf-8
from src.prompt_concat import CreateTestDataset, GetManualTestSamples
from src.utils import decode_csv_to_json, load_json, save_to_json
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from llama_cpp import Llama
import argparse
import os
import torch


class RealtimeChat:
    def __init__(self, role_name, role_info=None, role_dialog_file=None):
        self.available_role = ["三三"]
        if role_name not in self.available_role and (role_info is None or role_dialog_file is None):
            assert f"{role_name} not in list, provide role_desc and role_dialog_file"

        self.role_name = role_name
        self.role_info = role_info
        self.role_dialog_file = role_dialog_file
        self.role_data_path = f"./character/{self.role_name}.json"
        self.role_processed_file = f"./character/{self.role_name}_测试问题.json"

        self.save_samples_dir = "./character"
        self.save_samples_path = self.role_name + "_rag.json"
        self.prompt_path = "./prompt/dataset_character.txt"

        if self.role_name not in self.available_role:
            decode_csv_to_json(role_dialog_file, self.role_name, self.role_info, self.role_data_path)

        #  请先下载对应的index角色模型gguf版本，并修改为对应的路径
        config = load_json("config/config.json")
        self.gguf_model_path = config['gguf_model_local_path']

        self.history = []

    def generate_with_question(self, question):
        question_in = "\n".join(question)

        g = GetManualTestSamples(
            role_name=self.role_name,
            role_data_path=self.role_data_path,
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

    def run(self):
        # 启动 Llama 模型，n_gpu_layers=-1 表示所有层都使用 GPU，若无GPU可用或GPU显存小，设置为0或小的数值。
        llm = Llama(model_path = self.gguf_model_path, n_gpu_layers=-1, verbose=True, n_ctx=0)
        while True:
            query = input("user:")
            if query.strip() == "stop":
                break
            self.history.append(f"user:{query}")
            self.generate_with_question(self.history)
            self.create_datasets()

            json_data = load_json(f"{self.save_samples_dir}/{self.role_name}_测试问题.json")
            for i in json_data:
                text = i["input_text"]
                # 将输入数据传递给模型
                output = llm.create_chat_completion(
                    messages = [
                        {"role": "user","content": text}
                    ],
                    max_tokens = 2048
                )
                answer = f"{self.role_name}:{output['choices'][0]['message']['content']}"
                print(answer)
                self.history.append(answer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Realtime Chat Model")
    parser.add_argument("--role_name", type=str, required=True, help="rolename")
    parser.add_argument("--role_info", type=str, help="roleinfo")
    parser.add_argument("--role_dialog_file", type=str, help="rolepath")

    args = parser.parse_args()
    chat = RealtimeChat(role_name=args.role_name, role_info=args.role_info, role_dialog_file=args.role_dialog_file)
    chat.run()

