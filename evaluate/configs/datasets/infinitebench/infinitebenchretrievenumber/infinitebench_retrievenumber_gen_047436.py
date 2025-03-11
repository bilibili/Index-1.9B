from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import InfiniteBenchretrievenumberDataset
from opencompass.datasets.infinitebench.utils import InfiniteBench_first_number_postprocess

InfiniteBench_retrievenumber_reader_cfg = dict(
    input_columns=['context', 'input'],
    output_column='answer',

)

InfiniteBench_retrievenumber_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin=[
                dict(role='SYSTEM', fallback_role='HUMAN', prompt='You are a helpful assistant.'),
            ],
            round=[
                dict(role='HUMAN', prompt='There is an important info hidden inside a lot of irrelevant text. Find it. I will quiz you about the important information there.\n\n{context}\n\n{input}'),
                dict(role='BOT', prompt=''),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=12)
)

InfiniteBench_retrievenumber_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_postprocessor=dict(type=InfiniteBench_first_number_postprocess),
    pred_role='BOT'
)

InfiniteBench_retrievenumber_datasets = [
    dict(
        type=InfiniteBenchretrievenumberDataset,
        abbr='InfiniteBench_retrievenumber',
        path='./data/InfiniteBench/number_string.jsonl',
        reader_cfg=InfiniteBench_retrievenumber_reader_cfg,
        infer_cfg=InfiniteBench_retrievenumber_infer_cfg,
        eval_cfg=InfiniteBench_retrievenumber_eval_cfg)
]
