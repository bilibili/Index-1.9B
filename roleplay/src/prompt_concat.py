# coding=utf-8
from copy import deepcopy
from .get_dataset import CreateDataset
from .logger import LoggerFactory
from .retrieve_dialog import RetrieveDialog
from .utils import load_json, load_txt, save_to_json

import logging
import os

logger = LoggerFactory.create_logger(name="test", level=logging.INFO)


class GetManualTestSamples:
    def __init__(
        self,
        role_name,
        role_data_path,
        save_samples_dir,
        save_samples_path=None,
        prompt_path="dataset_character.txt",
        max_seq_len=4000,
        retrieve_num=20,
    ):
        self.role_name = role_name.strip()
        self.role_data = load_json(role_data_path)
        self.role_info = self.role_data[0]["role_info"].strip()

        self.prompt = load_txt(prompt_path)
        self.prompt = self.prompt.replace("${role_name}", self.role_name)
        self.prompt = self.prompt.replace("${role_info}",
                                          f"以下是{self.role_name}的人设：\n{self.role_info}\n").strip()

        self.retrieve_num = retrieve_num
        self.retrieve = RetrieveDialog(role_name=self.role_name,
                                       raw_dialog_list=[d["dialog"] for d in self.role_data],
                                       retrieve_num=retrieve_num)

        self.max_seq_len = max_seq_len
        if not save_samples_path:
            save_samples_path = f"{self.role_name}.json"
        self.save_samples_path = os.path.join(save_samples_dir, save_samples_path)

    def _add_simi_dialog(self, history: list, content_length):
        retrieve_results = self.retrieve.get_retrieve_res(history, self.retrieve_num)
        simi_dialogs = deepcopy(retrieve_results)

        if simi_dialogs:
            simi_dialogs = CreateDataset.choose_examples(simi_dialogs,
                                                         max_length=self.max_seq_len - content_length,
                                                         train_flag=False)
        logger.debug(f"retrieve_results: {retrieve_results}\nsimi_dialogs: {simi_dialogs}.")
        return simi_dialogs, retrieve_results

    def get_qa_samples_by_file(self,
                               questions_path,
                               user_name="user",
                               keep_retrieve_results_flag=False
                               ):
        questions = load_txt(questions_path).splitlines()
        samples = []
        for question in questions:
            question = question.replace('\\n', "\n")
            query = f"{user_name}:{question}" if ":" not in question else question
            content = self.prompt.replace("${dialog}", query)
            content = content.replace("${user_name}", user_name).strip()

            history = [query]
            simi_dialogs, retrieve_results = self._add_simi_dialog(history, len(content))

            sample = {
                "role_name": self.role_name,
                "role_info": self.role_info,
                "user_name": user_name,
                "dialog": history,
                "simi_dialogs": simi_dialogs,
            }
            if keep_retrieve_results_flag and retrieve_results:
                sample["retrieve_results"] = retrieve_results
            samples.append(sample)
        self._save_samples(samples)

    def get_qa_samples_by_query(self,
                                questions_query,
                                user_name="user",
                                keep_retrieve_results_flag=False
                                ):
        question = questions_query
        samples = []
        question = question.replace('\\n', "\n")
        query = f"{user_name}: {question}" if ":" not in question else question
        content = self.prompt.replace("${dialog}", query)
        content = content.replace("${user_name}", user_name).strip()

        history = [query]
        simi_dialogs, retrieve_results = self._add_simi_dialog(history, len(content))

        sample = {
            "role_name": self.role_name,
            "role_info": self.role_info,
            "user_name": user_name,
            "dialog": history,
            "simi_dialogs": simi_dialogs,
        }
        if keep_retrieve_results_flag and retrieve_results:
            sample["retrieve_results"] = retrieve_results
        samples.append(sample)
        self._save_samples(samples)

    def _save_samples(self, samples):
        data = samples
        save_to_json(data, self.save_samples_path)


class CreateTestDataset:
    def __init__(self,
                 role_name,
                 role_samples_path=None,
                 role_data_path=None,
                 prompt_path="dataset_character.txt",
                 max_seq_len=4000):
        self.max_seq_len = max_seq_len
        self.role_name = role_name

        self.prompt = load_txt(prompt_path)
        self.prompt = self.prompt.replace("${role_name}", role_name).strip()

        if not role_data_path:
            print("need role_data_path, check please!")
        self.default_simi_dialogs = None
        if os.path.exists(role_data_path):
            data = load_json(role_data_path)
            role_info = data[0]["role_info"]
        else:
            raise ValueError(f"{self.role_name} didn't find role_info.")
        self.role_info = role_info
        self.prompt = self.prompt.replace("${role_info}", f"以下是{self.role_name}的人设：\n{self.role_info}\n").strip()

        if role_samples_path:
            self.role_samples_path = role_samples_path
        else:
            print("check role_samples_path please!")

    def load_samples(self):
        samples = load_json(self.role_samples_path)
        results = []
        for sample in samples:
            input_text = self.prompt

            simi_dialogs = sample.get("simi_dialogs", None)
            if not simi_dialogs:
                simi_dialogs = self.default_simi_dialogs
            if not simi_dialogs:
                raise ValueError(f"didn't find simi_dialogs.")
            simi_dialogs = CreateDataset.choose_examples(simi_dialogs,
                                                         max_length=self.max_seq_len - len(input_text),
                                                         train_flag=False)

            input_text = input_text.replace("${simi_dialog}", simi_dialogs)
            user_name = sample.get("user_name", "user")
            input_text = input_text.replace("${user_name}", user_name)

            dialog = "\n".join(sample["dialog"]) if isinstance(sample["dialog"], list) else sample["dialog"]
            input_text = input_text.replace("${dialog}", dialog)

            assert len(input_text) < self.max_seq_len
            results.append({
                "input_text": input_text,
            })
        return results
