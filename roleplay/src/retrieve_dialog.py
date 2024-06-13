# coding=utf-8
from sentence_transformers import SentenceTransformer
from .utils import load_json

import faiss
import logging
import os
import re
import torch

logger = logging.getLogger(__name__)


class RetrieveDialog:
    def __init__(self,
                 role_name,
                 raw_dialog_list: list = None,
                 retrieve_num=20,
                 min_mean_role_utter_length=10):
        if torch.cuda.is_available():
            gpu_id = 0
            torch.cuda.set_device(gpu_id)

        assert raw_dialog_list

        self.role_name = role_name
        self.min_mean_role_utter_length = min_mean_role_utter_length
        self.retrieve_num = retrieve_num

        config = load_json("config/config.json")
        local_dir = config["bge_local_path"]

        if not os.path.exists(local_dir):
            print("Please download bge-large-zh-v1.5 first!")
        self.emb_model = SentenceTransformer(local_dir)

        self.dialogs, self.context_index = self._get_emb_base_by_list(raw_dialog_list)

        logger.info(f"dialog db num: {len(self.dialogs)}")
        logger.info(f"RetrieveDialog init success.")

    @staticmethod
    def dialog_preprocess(dialog: list, role_name):
        dialog_new = []
        # 把人名替换掉，减少对检索的影响
        user_names = []
        role_utter_length = []
        for num in range(len(dialog)):
            utter = dialog[num]
            try:
                user_name, utter_txt = re.split('[:：]', utter, maxsplit=1)
            except ValueError as e:
                logging.error(f"utter:{utter} can't find user_name.")
                return None, None

            if user_name != role_name:
                if user_name not in user_names:
                    user_names.append(user_name)
                index = user_names.index(user_name)
                utter = utter.replace(user_name, f"user{index}", 1)
            else:
                role_utter_length.append(len(utter_txt))
            dialog_new.append(utter)
        return dialog_new, user_names, role_utter_length

    def _get_emb_base_by_list(self, raw_dialog_list):
        logger.info(f"raw dialog db num: {len(raw_dialog_list)}")
        new_raw_dialog_list = []
        context_list = []

        # 为了兼容因为句长把所有对话都过滤掉的情况
        new_raw_dialog_list_total = []
        context_list_total = []
        for raw_dialog in raw_dialog_list:
            if not raw_dialog:
                continue

            end = 0
            for x in raw_dialog[::-1]:
                if x.startswith(self.role_name):
                    break
                end += 1

            raw_dialog = raw_dialog[:len(raw_dialog) - end]
            new_dialog, user_names, role_utter_length = self.dialog_preprocess(raw_dialog, self.role_name)
            if not new_dialog or not role_utter_length:
                continue

            if raw_dialog in new_raw_dialog_list_total:
                continue

            # 获得embedding时，不需要最后一句答案
            context = "\n".join(new_dialog) if len(new_dialog) < 2 else "\n".join(new_dialog[:-1])

            new_raw_dialog_list_total.append(raw_dialog)
            context_list_total.append(context)

            # 句长过滤
            role_length_mean = sum(role_utter_length) / len(role_utter_length)
            if role_length_mean < self.min_mean_role_utter_length:
                continue
            new_raw_dialog_list.append(raw_dialog)
            context_list.append(context)

        assert len(new_raw_dialog_list) == len(context_list)
        logger.debug(f"new_raw_dialog num: {len(new_raw_dialog_list)}")

        # 兼容样本过少的情况
        if len(new_raw_dialog_list) < self.retrieve_num:
            new_raw_dialog_list = new_raw_dialog_list_total
            context_list = context_list_total

        # 对话向量库
        context_vectors = self.emb_model.encode(context_list, normalize_embeddings=True)
        context_index = faiss.IndexFlatL2(context_vectors.shape[1])
        context_index.add(context_vectors)

        return new_raw_dialog_list, context_index

    def get_retrieve_res(self, dialog: list, retrieve_num: int):
        logger.debug(f"dialog: {dialog}")

        # 同样去掉user name影响
        dialog, _, _ = self.dialog_preprocess(dialog, self.role_name)
        dialog_vector = self.emb_model.encode(["\n".join(dialog)], normalize_embeddings=True)

        simi_dialog_distance, simi_dialog_index = self.context_index.search(
            dialog_vector, min(retrieve_num, len(self.dialogs)))
        simi_dialog_results = [
            (str(simi_dialog_distance[0][num]), self.dialogs[index]) for num, index in enumerate(simi_dialog_index[0])
        ]
        logger.debug(f"dialog retrieve res: {simi_dialog_results}")

        return simi_dialog_results
