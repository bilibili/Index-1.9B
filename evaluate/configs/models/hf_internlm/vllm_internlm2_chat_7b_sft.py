from opencompass.models import VLLMwithChatTemplate

models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr='internlm2-chat-7b-sft-vllm',
        path='internlm/internlm2-chat-7b-sft',
        model_kwargs=dict(tensor_parallel_size=1),
        max_out_len=1024,
        batch_size=32768,
        run_cfg=dict(num_gpus=1),
    )
]
