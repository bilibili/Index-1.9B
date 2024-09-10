# Wildbench

## Prepare the dataset

We support the [wildbench dataset](https://github.com/allenai/WildBench), developed by Lin et al. Please refer to their repo for more detail.

You have to download our preprocessed dataset. The format of dir should be like:

```
wildbench
---wildbench.jsonl
---gpt4
------wildbench.json
---claude
------wildbench.json
---llama2-70b
------wildbench.json
```

The wildbench.jsonl is the preprocessed dataset, and the other three are the reference, used for score.

Once you download the dataset, you have to modify the path defined in `configs/datasets/subjective/wildbench/wildbench_pair_judge.py` and `configs/datasets/subjective/wildbench/wildbench_single_judge.py`

## Run

We have provide the script for wildbench in `configs/eval_subjective_wildbench_pair.py` and `configs/eval_subjective_wildbench_single.py`.

Please modify the path for `give_pred` (line 171) in `configs/eval_subjective_wildbench_pair.py` to your path.

Note that if you test the wildbench with other models, please set the max_out_lens to 4096.
