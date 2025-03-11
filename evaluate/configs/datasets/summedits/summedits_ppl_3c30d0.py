from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import HFDataset

summedits_reader_cfg = dict(
    input_columns=['doc', 'summary'],
    output_column='label',
    test_split='train')

summedits_prompt1 = "Given the document below, you have to determine if 'Yes' or 'No', the summary is factually consistent with the document."
summedits_prompt2 = 'Document:\n{doc}\nSummary:\n{summary}\nIs the summary factually consistent with the document? '
summedits_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            0:
            dict(
                begin=[
                    dict(
                        role='SYSTEM',
                        fallback_role='HUMAN',
                        prompt=summedits_prompt1)
                ],
                round=[
                    dict(role='HUMAN', prompt=summedits_prompt2),
                    dict(role='BOT', prompt='No')
                ]),
            1:
            dict(
                begin=[
                    dict(
                        role='SYSTEM',
                        fallback_role='HUMAN',
                        prompt=summedits_prompt1)
                ],
                round=[
                    dict(role='HUMAN', prompt=summedits_prompt2),
                    dict(role='BOT', prompt='Yes')
                ]),
        }),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer))

summedits_eval_cfg = dict(evaluator=dict(type=AccEvaluator))

summedits_datasets = [
    dict(
        type=HFDataset,
        abbr='summedits',
        path='json',
        split='train',
        data_files='./data/summedits/summedits.jsonl',
        reader_cfg=summedits_reader_cfg,
        infer_cfg=summedits_infer_cfg,
        eval_cfg=summedits_eval_cfg)
]
