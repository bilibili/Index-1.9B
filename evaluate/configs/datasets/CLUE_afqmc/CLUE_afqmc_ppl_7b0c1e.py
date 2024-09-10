from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import HFDataset

afqmc_reader_cfg = dict(
    input_columns=['sentence1', 'sentence2'],
    output_column='label',
    test_split='train')

afqmc_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            0: '{sentence1},{sentence2}不同。',
            1: '{sentence1},{sentence2}相似。'
        }),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer))

afqmc_eval_cfg = dict(evaluator=dict(type=AccEvaluator))

afqmc_datasets = [
    dict(
        type=HFDataset,
        abbr='afqmc-dev',
        path='json',
        data_files='./data/CLUE/AFQMC/dev.json',
        split='train',
        reader_cfg=afqmc_reader_cfg,
        infer_cfg=afqmc_infer_cfg,
        eval_cfg=afqmc_eval_cfg),
]
