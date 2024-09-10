from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import EMEvaluator, RougeEvaluator
from opencompass.datasets.leval import LEvalMultidocQADataset

LEval_multidocqa_reader_cfg = dict(
    input_columns=['context', 'question', 'length'],
    output_column='answer',
    train_split='test',
    test_split='test'
)

LEval_multidocqa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin=[
                dict(role='SYSTEM', fallback_role='HUMAN', prompt='Now you are given a very long document. Please follow the instruction after this document. These instructions may include summarizing a document, answering questions based on the document, or writing a required paragraph.'),
            ],
            round=[
                dict(role='HUMAN', prompt='Document is as follows. {context}\nInstruction: {question}\nAnswer this question with {length} words.'),
                dict(role='BOT', prompt=''),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=64)
)

LEval_multidocqa_eval_cfg = dict(
    evaluator=dict(type=RougeEvaluator),
    pred_role='BOT'
)

LEval_multidocqa_datasets = [
    dict(
        type=LEvalMultidocQADataset,
        abbr='LEval_multidocqa',
        path='./data/LEval/',
        name='multidoc_qa',
        reader_cfg=LEval_multidocqa_reader_cfg,
        infer_cfg=LEval_multidocqa_infer_cfg,
        eval_cfg=LEval_multidocqa_eval_cfg)
]
