from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccwithDetailsEvaluator
from opencompass.datasets import RaceDataset
from opencompass.utils.text_postprocessors import first_option_postprocess

race_reader_cfg = dict(
    input_columns=['article', 'question', 'A', 'B', 'C', 'D'],
    output_column='answer',
    train_split='validation',
    test_split='test'
)

race_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt=
                'Read the article, and answer the question by replying A, B, C or D.\n\nArticle:\n{article}\n\nQ: {question}\n\nA. {A}\nB. {B}\nC. {C}\nD. {D}'
            ),
        ])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

race_eval_cfg = dict(
    evaluator=dict(type=AccwithDetailsEvaluator),
    pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'),
    pred_role='BOT')

race_datasets = [
    dict(
        abbr='race-middle',
        type=RaceDataset,
        path='./data/race',
        name='middle',
        reader_cfg=race_reader_cfg,
        infer_cfg=race_infer_cfg,
        eval_cfg=race_eval_cfg),
    dict(
        abbr='race-high',
        type=RaceDataset,
        path='./data/race',
        name='high',
        reader_cfg=race_reader_cfg,
        infer_cfg=race_infer_cfg,
        eval_cfg=race_eval_cfg)
]
