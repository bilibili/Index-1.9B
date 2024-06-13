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

```bash
# Download dataset to data/ folder
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip
unzip OpenCompassData-core-20240207.zip
```

3. Evaluation

   - Run with python command:

      ```bash
      python run.py --datasets mmlu_ppl ceval_ppl cmmlu_ppl ARC_c_ppl ARC_e_ppl hellaswag_ppl gsm8k_gen humaneval_gen \
          --hf-path ./models/model_name \ 
          --model-kwargs device_map='auto' trust_remote_code=True \
          --tokenizer-kwargs padding_side='left' truncation='left' use_fast=False trust_remote_code=True \
          --max-out-len 1 \
          --max-seq-len 4096 \
          --batch-size 64 \
          --no-batch-padding \
          --num-gpus 1
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
     