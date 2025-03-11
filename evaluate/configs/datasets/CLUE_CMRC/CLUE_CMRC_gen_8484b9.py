from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import EMEvaluator
from opencompass.datasets import CMRCDataset

CMRC_reader_cfg = dict(
    input_columns=['question', 'context'], output_column='answers')

CMRC_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template='文章：{context}\n根据上文，回答如下问题： {question}\n答：'),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

CMRC_eval_cfg = dict(evaluator=dict(type=EMEvaluator), )

CMRC_datasets = [
    dict(
        type=CMRCDataset,
        abbr='CMRC_dev',
        path='./data/CLUE/CMRC/dev.json',
        reader_cfg=CMRC_reader_cfg,
        infer_cfg=CMRC_infer_cfg,
        eval_cfg=CMRC_eval_cfg),
]
