from opencompass.models import Gemini


api_meta_template = dict(
    round=[
            dict(role='HUMAN', api_role='HUMAN'),
            dict(role='BOT', api_role='BOT', generate=True),
    ],
)

models = [
    dict(abbr='gemini',
    type=Gemini,
    path='gemini-pro',
    key='your keys',  # The key will be obtained from Environment, but you can write down your key here as well
    url = 'your url',
    meta_template=api_meta_template,
    query_per_second=16,
    max_out_len=100,
    max_seq_len=2048,
    batch_size=1,
    temperature=1,)
]
