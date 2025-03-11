from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import HFDataset, HumanEvalEvaluator, humaneval_postprocess

apps_reader_cfg = dict(
    input_columns=['question'], output_column='problem_id', train_split='test')

# TODO: allow empty output-column
apps_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template='{question}'),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512))

apps_eval_cfg = dict(
    evaluator=dict(type=HumanEvalEvaluator),
    k=[1, 10, 100],
    pred_postprocessor=dict(type=humaneval_postprocess),
)

apps_datasets = [
    dict(
        type=HFDataset,
        path='codeparrot/apps',
        reader_cfg=apps_reader_cfg,
        infer_cfg=apps_infer_cfg,
        eval_cfg=apps_eval_cfg)
]
