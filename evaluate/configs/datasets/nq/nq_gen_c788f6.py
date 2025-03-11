from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import NaturalQuestionDataset, NQEvaluator

nq_reader_cfg = dict(
    input_columns=['question'], output_column='answer', train_split='test')

nq_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='Answer these questions, your answer should be as simple as possible, start your answer with the prompt \'The answer is \'.\nQ: {question}?'),
                dict(role='BOT', prompt='A:'),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

nq_eval_cfg = dict(evaluator=dict(type=NQEvaluator), pred_role='BOT')

nq_datasets = [
    dict(
        type=NaturalQuestionDataset,
        abbr='nq',
        path='./data/nq/',
        reader_cfg=nq_reader_cfg,
        infer_cfg=nq_infer_cfg,
        eval_cfg=nq_eval_cfg)
]
