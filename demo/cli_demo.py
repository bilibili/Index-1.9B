import os
import re
import json
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

# 注意！目录不能含有"."，可以替换成"_"
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default="", type=str, help="")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map='auto')
model = model.eval()
print('model loaded', args.model_path, model.device)


# 重置生成参数
def reset_para(generate_config, input_key, value):
    if input_key not in generate_config:
        print('input para key not legal')
        return generate_config

    value = value.strip()

    dtype = type(generate_config[input_key])
    if dtype == bool:
        if value in set(['0', 'False', 'false']):
            value = False
        elif value in set(['1', 'True', 'true']):
            value = True
        else:
            print('bool value error, you can set 0 or 1')
            return generate_config
    else:
        value = dtype(value)
    generate_config[input_key] = value
    print('para after reset:', generate_config)

    chat_template_changed = False
    if input_key in set(['sys_token_id', 'user_token_id', 'bot_token_id', 'system_message']):
        chat_template_changed = True
    return generate_config, chat_template_changed

def init_chat_template(generate_config):
    # Special Token 编号
    sys_token_id = generate_config['sys_token_id'] 
    user_token_id = generate_config['user_token_id']
    bot_token_id = generate_config['bot_token_id']

    system_start_ids = torch.tensor([[sys_token_id]], dtype=torch.int64, device=model.device)
    user_start_ids = torch.tensor([[user_token_id]], dtype=torch.int64, device=model.device)
    bot_start_ids = torch.tensor([[bot_token_id]], dtype=torch.int64, device=model.device)

    # System Message
    system = generate_config['system_message']
    system_ids = tokenizer.encode(system, return_tensors="pt").to(model.device)
    if len(system_ids) == 0 or system_ids.shape[-1] == 0:
        print('system_message is empty')
        system_ids = torch.tensor([[]], dtype=torch.int64).to(model.device)
    else:
        system_ids = torch.concat([system_start_ids,system_ids], dim=-1).long()
    return system_start_ids, user_start_ids, bot_start_ids, system_ids

def main():
    generate_config = {
        "max_new":300,
        "do_sample":True,
        "top_p":0.8,
        "top_k":5,
        "temperature": 0.3,
        "repetition_penalty":1.1,
        "system_message": "你是由哔哩哔哩人工智能平台部自主研发的大语言模型，名为“Index”。你能够根据用户传入的信息，帮助用户完成指定的任务，并生成恰当的、符合要求的回复。",
        "sys_token_id": 0,
        "user_token_id": 3,
        "bot_token_id": 4
    }
    system_start_ids, user_start_ids, bot_start_ids, system_ids = init_chat_template(generate_config)

    print("=====预设生成参数:", generate_config)
    print('')
    print("=====重置生成参数: 输入reset para key_value:value ")
    print("如reset para top_p:0.85, 重置top_p")
    print('')
    print("=====换行和json输入:")
    print("换行输入: json\"xxx\\nxxx\"")
    print("Json输入: json\"{\"key1\":\"aaa\\nbbb\"}")
    print('')
    print('=====注意:当前多轮对话, 历史对话会影响当前输出')
    print('输入"new session"开启新对话')
    print('输入"single mode"开启单轮模式')
    print('输入"multi mode"开启多轮模式')

    history_outputs = system_ids
    chat_mode = 'multi mode'
    while True:
        query = input("\nUser:")
        if len(query) == 0:
            continue

        # 处理Json格式
        if query.startswith('json"'):
            try:
              query = json.loads(query[4:])
            except:
              print('json format error !!')

        if query == 'show para':
            print(generate_config)
            continue

        # 开启新会话
        if query == 'new session':
            history_outputs = system_ids
            print('new session started')
            continue

        # 开启单轮模式
        if query == 'single mode':
            history_outputs = system_ids
            chat_mode = 'single mode'
            print('开启单轮对话模式模式, 不再记忆历史对话')
            continue

        # 开启对轮模式
        if query == 'multi mode':
            history_outputs = system_ids
            chat_mode = 'multi mode'
            print('开启多轮对话模式模式, 将记忆历史对话')
            continue

        if query.startswith('reset para '):
            try:
                input_key, value = re.findall('reset para (.*?):(.*?)$', query)[0]
                generate_config, chat_template_changed = reset_para(generate_config, input_key, value)
                if chat_template_changed:
                    system_start_ids, user_start_ids, bot_start_ids, system_ids = init_chat_template(generate_config)
                    print(system_start_ids, user_start_ids, bot_start_ids, system_ids)
                    history_outputs = system_ids
                    print('chat template changed, start new session!')
            except Exception as e:
                print('reset para format error !', e)
            continue

        if chat_mode == 'single mode':
            history_outputs = system_ids

        inputs = tokenizer.encode(query, return_tensors="pt").to(model.device)
        inputs = torch.concat([history_outputs, user_start_ids, inputs, bot_start_ids], dim=-1).long()
        history_outputs = model.generate(inputs, 
                        max_new_tokens=generate_config['max_new'], 
                        top_k=generate_config['top_k'], 
                        top_p=generate_config['top_p'], 
                        temperature=generate_config['temperature'], 
                        repetition_penalty=generate_config['repetition_penalty'], 
                        do_sample=generate_config['do_sample'])
        # 删除</s>
        if history_outputs[0][-1] == 2:
            history_outputs = history_outputs[:, :-1]

        outputs = history_outputs[0][len(inputs[0]):]
        print('\nModel:', tokenizer.decode(outputs))

        # 调试所有输出
        # print('\nModel DEBUG:', tokenizer.decode(history_outputs[0]))
        # print('\nModel DEBUG:', history_outputs[0])

if __name__ == "__main__":
    main()
