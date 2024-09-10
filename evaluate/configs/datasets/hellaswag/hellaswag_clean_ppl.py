from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccContaminationEvaluator
from opencompass.datasets import hellaswagDatasetClean as hellaswagDataset

hellaswag_reader_cfg = dict(
    input_columns=['ctx', 'A', 'B', 'C', 'D'],
    output_column='label')

hellaswag_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            i: dict(round=[
                dict(role='HUMAN', prompt='{ctx}'),
                dict(role='BOT', prompt=f"{{{chr(ord('A') + i)}}}"),
            ])
            for i in range(4)
        }),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer))

hellaswag_eval_cfg = dict(evaluator=dict(type=AccContaminationEvaluator),
                          analyze_contamination=True)

hellaswag_datasets = [
    dict(
        abbr='hellaswag',
        type=hellaswagDataset,
        path='./data/hellaswag/hellaswag.jsonl',
        reader_cfg=hellaswag_reader_cfg,
        infer_cfg=hellaswag_infer_cfg,
        eval_cfg=hellaswag_eval_cfg)
]
