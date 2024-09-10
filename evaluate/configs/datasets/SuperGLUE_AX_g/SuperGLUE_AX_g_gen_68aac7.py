from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import AXDataset_V2
from opencompass.utils.text_postprocessors import first_option_postprocess

AX_g_reader_cfg = dict(
    input_columns=['hypothesis', 'premise'],
    output_column='label',
)

AX_g_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt=
                '{premise}\n{hypothesis}\nIs the sentence below entailed by the sentence above?\nA. Yes\nB. No\nAnswer:'
            ),
        ]),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

AX_g_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=first_option_postprocess, options='AB'),
)

AX_g_datasets = [
    dict(
        abbr='AX_g',
        type=AXDataset_V2,
        path='./data/SuperGLUE/AX-g/AX-g.jsonl',
        reader_cfg=AX_g_reader_cfg,
        infer_cfg=AX_g_infer_cfg,
        eval_cfg=AX_g_eval_cfg,
    )
]
