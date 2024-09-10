from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import LLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import winograndeDataset

winogrande_reader_cfg = dict(
    input_columns=['opt1', 'opt2'],
    output_column='answer',
)

winogrande_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            1: '{opt1}',
            2: '{opt2}',
        }
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=LLInferencer))

winogrande_eval_cfg = dict(evaluator=dict(type=AccEvaluator))

winogrande_datasets = [
    dict(
        abbr='winogrande',
        type=winograndeDataset,
        path='./data/winogrande',
        reader_cfg=winogrande_reader_cfg,
        infer_cfg=winogrande_infer_cfg,
        eval_cfg=winogrande_eval_cfg)
]
