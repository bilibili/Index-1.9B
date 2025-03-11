from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import ChatInferencer, GenInferencer
from opencompass.openicl.icl_evaluator import LMEvaluator
from opencompass.datasets import MultiroundDataset


subjective_reader_cfg = dict(
    input_columns=['dialogue', 'capability', 'gpt4_prefix', 'gpt4_suffix'],
    output_column='judge',
    )

subjective_all_sets = [
    'FunctionalMT',
]
data_path ='data/subjective/'

subjective_datasets = []

for _name in subjective_all_sets:
    subjective_infer_cfg = dict(
            prompt_template=dict(
                type=PromptTemplate,
                template="""{dialogue}""",
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=ChatInferencer, max_seq_len=4096, max_out_len=512, infer_mode='every'),
        )

    subjective_eval_cfg = dict(
        evaluator=dict(
            type=LMEvaluator,
            pack_all_predictions=True,
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(round=[
                    dict(
                        role='HUMAN',
                        prompt = '{gpt4_prefix}{prediction}{gpt4_suffix}'
                    ),
                ]),
            ),
        ),
        pred_role='BOT',
    )

    subjective_datasets.append(
        dict(
            abbr=f'{_name}',
            type=MultiroundDataset,
            path=data_path,
            name=_name,
            reader_cfg=subjective_reader_cfg,
            infer_cfg=subjective_infer_cfg,
            eval_cfg=subjective_eval_cfg
        ))
