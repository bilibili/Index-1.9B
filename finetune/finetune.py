import logging
import torch
import os
import sys
import json
import transformers
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
    DataCollatorForSeq2Seq
)
from peft import get_peft_model, LoraConfig, TaskType
from utils.trainer import LoRATrainer
from utils.arguments import ModelArguments, DataTrainingArguments, ModelTrainingArguments
from datasets import Dataset
import pandas as pd

logger = logging.getLogger(__name__)

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, ModelTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.ddp_find_unused_parameters = False
    training_args.save_safetensors = False

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    set_seed(training_args.seed)

    logger.info('Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True, use_fast=False)
    print(f'tokenizer special_tokens_map: {tokenizer.special_tokens_map}')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    assert tokenizer.pad_token_id is not None, "pad_token_id should not be None"
    assert tokenizer.eos_token_id is not None, "eos_token_id should not be None"

    # 将JSON文件转换为CSV文件
    df = pd.read_json(data_args.train_data)
    ds = Dataset.from_pandas(df)

    def process_func(example):
        MAX_LENGTH = int(data_args.max_seq_length)
        input_ids, attention_mask, labels = [], [], []

        system_default = "你是由哔哩哔哩自主研发的大语言模型，名为“Index”。你能够根据用户传入的信息，帮助用户完成指定的任务，并生成恰当的、符合要求的回复。"
        system = example.get("system", system_default)
        human = example['human']
        assistant = example['assistant']

        instruction = tokenizer(f"<unk>{system}reserved_0{human}reserved_1", add_special_tokens=False)
        response = tokenizer(f"{assistant}", add_special_tokens=False)
        input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
        attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]  
        if len(input_ids) > MAX_LENGTH:
            input_ids = input_ids[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    train_dataset = ds.map(process_func, remove_columns=ds.column_names)

    torch_dtype = torch.float16 if training_args.fp16 else torch.bfloat16
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    config.use_cache = False
    
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, 
                                        torch_dtype=torch_dtype,
                                        config=config, 
                                        trust_remote_code=True)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=model_args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)

    trainer = LoRATrainer(
        model=model, 
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    logger.info("*** starting training ***")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload

if __name__ == "__main__":
    main()