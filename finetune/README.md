# Index-1.9B-Chat Lora 微调

## **环境准备**
请确保你的torch版本高于2.0.0，并且支持gpu训练，其他主要安装包的版本
```bash
pip install transformers==4.39.2
pip install sentencepiece==0.1.99
pip install accelerate==0.27.0
pip install transformers_stream_generator==0.0.4
pip install datasets==2.8.0
pip install peft==0.10.0
```

## 模型下载  
模型下载链接：https://huggingface.co/IndexTeam/Index-1.9B-Chat

## 指令集构建
```json
[
    {
        "system":"回答以下用户问题，仅输出答案。",
        "human":"1+1等于几?",
        "assistant":"2"
    },
    ...
]
```
如果没有“system”字段，会默认使用官方原始的system message.

## 训练脚本

修改以下变量，运行 **bash train.sh**开始训练。
```bash
TRAIN_PATH=数据路径
BASE_MODEL_PATH=下载好的原始模型路径
```

## 推理脚本

修改以下变量，运行 **python infer.py**执行预测。
```python
mode_path = 下载好的原始模型路径
lora_path = 训练好的lora模型路径

system = 训练使用的system message
prompt = 输入问题
```
