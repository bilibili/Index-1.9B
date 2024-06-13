# coding=utf-8
from src.logger import LoggerFactory
from src.utils import decode_csv_to_json, load_json, save_to_json
from src.prompt_concat import GetManualTestSamples, CreateTestDataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import json
import logging
import os

logger = LoggerFactory.create_logger(name="test", level=logging.INFO)


class IndexRolePlay:
    def __init__(self, role_name, role_info=None, role_dialog_file=None):
        self.available_role = ['三三']
        if role_name not in self.available_role and (role_info is None or role_dialog_file is None):
            assert f'{role_name} not in list, provide role_desc and role_dialog_file'

        self.role_name = role_name
        self.role_info = role_info
        self.role_dialog_file = role_dialog_file
        self.role_processed_file = f"./character/{self.role_name}_测试问题.json"

        self.json_output_path = f"./character/{self.role_name}.json"

        self.save_samples_dir = "./character"
        self.save_samples_path = self.role_name + "_rag.json"
        self.prompt_path = "./prompt/dataset_character.txt"

        if self.role_name not in self.available_role:
            decode_csv_to_json(role_dialog_file, self.role_name, self.role_info, self.json_output_path)

        #  请先下载对应的index角色模型，并修改为对应的路径
        with open('config/config.json') as f:
            config_data = json.load(f)
            self.huggingface_local_path = config_data['huggingface_local_path']

    def generate_with_question_file(self, question_path):
        g = GetManualTestSamples(
            role_name=self.role_name,
            role_data_path=self.json_output_path,
            save_samples_dir=self.save_samples_dir,
            save_samples_path=self.save_samples_path,
            prompt_path=self.prompt_path,
            max_seq_len=4000
        )
        g.get_qa_samples_by_file(
            questions_path=question_path,
            keep_retrieve_results_flag=True
        )

    def generate_with_question(self, question):
        question_in = f"user:{question}"

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

    def infer_with_question_file(self, question_file):
        config = AutoConfig.from_pretrained(self.huggingface_local_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.huggingface_local_path,
            trust_remote_code=True,
            config=config)
        tokenizer = AutoTokenizer.from_pretrained(self.huggingface_local_path, trust_remote_code=True)

        self.generate_with_question_file(question_file)
        self.create_datasets()

        json_data = load_json(f"{self.save_samples_dir}/{self.role_name}_测试问题.json")
        for i in json_data:
            text = i["input_text"]
            # 使用分词器对文本进行编码
            inputs = tokenizer(text, return_tensors="pt")

            # 将输入数据传递给模型
            outputs = model.generate(**inputs, repetition_penalty=1.1, max_length=2048)

            # 打印模型输出
            res = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            print(text + res[len(text):])

    def infer_with_question(self, question):
        config = AutoConfig.from_pretrained(self.huggingface_local_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.huggingface_local_path,
            trust_remote_code=True,
            config=config)
        tokenizer = AutoTokenizer.from_pretrained(self.huggingface_local_path, trust_remote_code=True)

        self.generate_with_question(question)
        self.create_datasets()

        json_data = load_json(f"{self.save_samples_dir}/{self.role_name}_测试问题.json")
        for i in json_data:
            text = i['input_text']
            print(question)
            # 使用分词器对文本进行编码
            inputs = tokenizer(text, return_tensors="pt")

            # 将输入数据传递给模型
            outputs = model.generate(**inputs, repetition_penalty=1.1, max_length=2048)

            # 打印模型输出
            res = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            print(res[len(text):])


if __name__ == "__main__":
    chatbox = IndexRolePlay(role_name="三三")
    # chatbox.infer_with_question_file("question.txt")
    chatbox.infer_with_question("你叫什么名字")
