from src.prompt_concat import CreateTestDataset, GetManualTestSamples
from src.utils import decode_csv_to_json, load_json, save_to_json
import argparse
import os
import requests
import json
import subprocess
import time

class RealtimeChat:
    def __init__(self, role_name, role_info=None, role_dialog_file=None):
        self.available_role = ["三三"]
        if role_name not in self.available_role and (role_info is None or role_dialog_file is None):
            raise ValueError(f"{role_name} not in list, provide role_desc and role_dialog_file")

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

        self.history = []
        self.base_url = "http://localhost:11434"
        self.model_name = "Index-1.9B-Character"

        # 启动 Ollama 模型
        self.start_ollama_model()

    def start_ollama_model(self):
        try:
            # 启动 Ollama 模型并将输出重定向到文件
            self.model_process = subprocess.Popen(
                ["nohup", "ollama", "run", self.model_name, ">", "ollama_model_output.log", "2>&1", "&"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True
            )
            time.sleep(5)  # 等待模型启动
            print("Ollama 模型已启动。")
        except Exception as e:
            print(f"启动 Ollama 模型时出错: {e}")

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

    def chat_with_ollama(self, messages):
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model_name,
            "messages": messages
        }
        headers = {
            "Content-Type": "application/json"
        }
        response = requests.post(url, data=json.dumps(payload), headers=headers)
        try:
            response.raise_for_status()

            # 手动处理响应内容
            response_lines = response.content.decode('utf-8').strip().split("\n")
            responses = [json.loads(line) for line in response_lines]

            # 提取最终的完整消息
            complete_message = ""
            for res in responses:
                complete_message += res['message']['content']
                if res.get('done', False):
                    break

            return {"choices": [{"message": {"content": complete_message}}]}

        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
            print(f"Response content: {response.content.decode('utf-8')}")
        except requests.exceptions.RequestException as req_err:
            print(f"Request error occurred: {req_err}")
            print(f"Response content: {response.content.decode('utf-8')}")
        except json.JSONDecodeError as json_err:
            print(f"JSON decode error occurred: {json_err}")
            print(f"Response content: {response.content.decode('utf-8')}")

    def run(self):
        while True:
            query = input("user: ")
            if query.strip().lower() == "stop":
                break
            self.history.append(f"user: {query}")
            self.generate_with_question(self.history)
            self.create_datasets()

            json_data = load_json(f"{self.save_samples_dir}/{self.role_name}_测试问题.json")
            for i in json_data:
                text = i["input_text"]
                messages = [
                    {"role": "user", "content": text}
                ]
                response = self.chat_with_ollama(messages)
                if response:
                    answer = f"{self.role_name}: {response['choices'][0]['message']['content']}"
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
