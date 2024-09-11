# Description

In order to facilitate everyone to reproduce our experimental results, we will release the evaluation code. We used mainstream open source evaluation tasks（MMLU, CMMLU, CEVAL...） to measure the performance of our model. At the same time, we adopted [OpenCompass](https://github.com/open-compass/opencompass) as the main framework of the evaluation, and made adaptive modifications on this basis.


# Quick Start

1. Environment Setup

```bash
conda create --name benchmark_env python=3.10 pytorch torchvision pytorch-cuda -c nvidia -c pytorch -y
conda activate benchmark_env
git clone llm_benchmark_repo_url llm_benchmark
cd llm_benchmark
pip install -e .
```

2. Data Download

Please download from these URLs manually and place them in the correct directories as follows.

needlebench：
https://github.com/open-compass/opencompass/releases/download/0.2.4.rc1/OpenCompassData-complete-20240325.zip

LongBench:
https://huggingface.co/datasets/THUDM/LongBench/tree/main

LEval:
https://huggingface.co/datasets/L4NLP/LEval/tree/main

The placement directories and locations of the respective datasets are as follows:
```bash
data/
├── LongBench/
│   ├── LongBench.py
│   ├── README.md
│   └── data/
├── LEval/
│   ├── LEval.py
│   ├── README.md
│   ├── test_data.ipynb
│   └── LEval/
│       ├── Exam/
│       └── Generation/
└── needlebench/
    ├── PaulGrahamEssays.jsonl
    ├── multi_needle_reasoning_en.json
    ├── multi_needle_reasoning_zh.json
    ├── names.json
    ├── needles.jsonl
    ├── zh_finance.jsonl
    ├── zh_game.jsonl
    ├── zh_general.jsonl
    ├── zh_government.jsonl
    ├── zh_movie.jsonl
    └── zh_tech.jsonl
```

3. Evaluation

   Run with python command:

      ```bash
    # mmlu_gen ceval_gen cmmlu_gen hellaswag_gen gsm8k_gen humaneval_gen
    LLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=0 python run.py \
        --datasets mmlu_gen ceval_gen cmmlu_gen hellaswag_gen gsm8k_gen humaneval_gen \
        --hf-path your_model_path/model_name \
        --model-kwargs device_map='auto' trust_remote_code=True torch_dtype=torch.bfloat16 \
        --tokenizer-kwargs padding_side='left' truncation='left' use_fast=False trust_remote_code=True \
        --max-seq-len 4096 \
        --batch-size 32 \
        --hf-num-gpus 1 \
        --mode all

    # longbench
    LLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=0 python run.py \
        --datasets longbench \
        --summarizer longbench/summarizer \
        --hf-path your_model_path/model_name \
        --model-kwargs device_map='auto' trust_remote_code=True torch_dtype=torch.bfloat16 \
        --tokenizer-kwargs padding_side='left' truncation='left' use_fast=False trust_remote_code=True \
        --max-seq-len 32768 \
        --batch-size 1 \
        --hf-num-gpus 1 \
        --mode all 

    # leval
    LLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=0 python run.py \
        --datasets leval \
        --summarizer leval/summarizer \
        --hf-path your_model_path/model_name \
        --model-kwargs device_map='auto' trust_remote_code=True torch_dtype=torch.bfloat16 \
        --tokenizer-kwargs padding_side='left' truncation='left' use_fast=False trust_remote_code=True \
        --max-seq-len 32768 \
        --batch-size 1 \
        --hf-num-gpus 1 \
        --mode all

    # needlebench
    LLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=0 python run.py \
        --datasets needlebench_single_32k \
        --summarizer needlebench/needlebench_32k_summarizer \
        --hf-path your_model_path/model_name \
        --model-kwargs device_map='auto' trust_remote_code=True torch_dtype=torch.bfloat16 \
        --tokenizer-kwargs padding_side='left' truncation='left' use_fast=False trust_remote_code=True \
        --max-seq-len 32768 \
        --batch-size 1 \
        --hf-num-gpus 1 \
        --mode all
      ```

   - Run with config file:
     
     Define the task_file in [run_local_test.py](run_local_test.py_line_10_url), then run the following command:

      ```bash
      ./run_local_test.sh
      ```

4. Get dataset config file

   Use following python command to get dataset config
    ```bash
    # dataset name like mmlu or arc
    python ./tools/list_configs.py mmlu arc
    ```

# Acknowledgements
Thanks to the release of the following projects, which have provided great help in quickly building a comparable benchamrk：
- [OpenCompass](https://github.com/open-compass/opencompass)
- [HuggingFace](https://huggingface.co/)
- [OpenICL](https://github.com/Shark-NLP/OpenICL)
- [EvalPlus](https://github.com/evalplus/evalplus)
     