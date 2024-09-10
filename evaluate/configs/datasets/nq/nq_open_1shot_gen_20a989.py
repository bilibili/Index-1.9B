from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever, FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import NQOpenDataset, NQEvaluator

nq_datasets = []
for k in [1]:
    nq_reader_cfg = dict(
        input_columns=['question'], output_column='answer', train_split='train', test_split='validation')

    if k == 0:
        nq_infer_cfg = dict(
            prompt_template=dict(
                type=PromptTemplate,
                template='Q: {question}\nA: ',
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer, max_out_len=50)
        )
    else:
        nq_infer_cfg = dict(
            ice_template=dict(
                type=PromptTemplate,
                template='Q: {question}\nA: {answer}.\n',
            ),
            prompt_template=dict(
                type=PromptTemplate,
                template='</E>Q: {question}\nA: ',
                ice_token='</E>',
            ),
            retriever=dict(type=FixKRetriever, fix_id_list=list(range(k))),
            inferencer=dict(type=GenInferencer, max_out_len=50, stopping_criteria=['Q:', '\n']),
        )

    nq_eval_cfg = dict(evaluator=dict(type=NQEvaluator), pred_role='BOT')

    nq_datasets.append(
        dict(
            type=NQOpenDataset,
            abbr=f'nq_open_{k}shot',
            path='./data/nq-open/',
            reader_cfg=nq_reader_cfg,
            infer_cfg=nq_infer_cfg,
            eval_cfg=nq_eval_cfg)
        )
