from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import OBQADataset

_input_columns = [
    ['question_stem', 'A', 'B', 'C', 'D'],
    ['question_stem', 'A', 'B', 'C', 'D', 'fact1'],
]
_template = [{
    'A': '{question_stem} {A}',
    'B': '{question_stem} {B}',
    'C': '{question_stem} {C}',
    'D': '{question_stem} {D}',
}, {
    'A': 'Given the fact {fact1}, we know that {question_stem} {A}',
    'B': 'Given the fact {fact1}, we know that {question_stem} {B}',
    'C': 'Given the fact {fact1}, we know that {question_stem} {C}',
    'D': 'Given the fact {fact1}, we know that {question_stem} {D}',
}]

obqa_datasets = [
    dict(
        abbr='openbookqa',
        type=OBQADataset,
        path='./data/openbookqa/Main/test.jsonl',
    ),
    dict(
        abbr='openbookqa_fact',
        type=OBQADataset,
        path='./data/openbookqa/Additional/test_complete.jsonl',
    ),
]
for _i in range(2):
    obqa_reader_cfg = dict(
        input_columns=_input_columns[_i], output_column='answerKey')
    obqa_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=_template[_i]),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=PPLInferencer),
    )
    obqa_eval_cfg = dict(evaluator=dict(type=AccEvaluator))

    obqa_datasets[_i]['reader_cfg'] = obqa_reader_cfg
    obqa_datasets[_i]['infer_cfg'] = obqa_infer_cfg
    obqa_datasets[_i]['eval_cfg'] = obqa_eval_cfg
