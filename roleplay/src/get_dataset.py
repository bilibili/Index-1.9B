# coding=utf-8
import sys
sys.path.append("../")

from collections import defaultdict
from .utils import is_float, load_txt

import random

random.seed(1234)


class CreateDataset:
    def __init__(self, max_input_len=1500):
        self.prompt = load_txt("../prompt/dataset_character.txt")
        self.max_input_len = max_input_len  # 小于(seq-length)-(max-gen-length)
        self.example_split_flag = f"\n{'-' * 20}\n"

        self.dataset = defaultdict(list)
        self.manual_dataset = []

    @staticmethod
    def choose_examples(similar_examples,
                        max_length,
                        train_flag=False,
                        dialog=None,
                        example_split_flag=f"\n{'-' * 20}\n"):
        if isinstance(similar_examples, str):
            new_similar_examples = [x.strip() for x in similar_examples.split(example_split_flag)]
        else:
            # 去重
            new_similar_examples = []
            for example in similar_examples:
                if (isinstance(example, list) or isinstance(example, tuple)) and len(example) == 2 and is_float(
                        example[0]):
                    # 包含score
                    example = example[1]

                try:
                    example = "\n".join(example).strip()
                except TypeError:
                    raise TypeError(f"example: {example}")
                if train_flag and dialog and (example in dialog or dialog in example):
                    continue

                # example去重
                if train_flag:
                    # 部分相似也去掉
                    flag = False
                    for n_example in new_similar_examples:
                        if example in n_example or n_example in example:
                            flag = True
                            break
                    if not flag:
                        new_similar_examples.append(example)
                else:
                    if example not in new_similar_examples:
                        new_similar_examples.append(example)

        results = []
        total_length = 0
        for example in new_similar_examples:
            total_length += len(example) if not total_length else len(example_split_flag) + len(example)
            if total_length > max_length:
                break
            results.append(example)
        results = example_split_flag.join(results).strip()
        return results
