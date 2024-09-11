import os
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import warnings
import time
import threading

warnings.filterwarnings("ignore", category=UserWarning, module='transformers.generation.utils')
warnings.filterwarnings("ignore", category=UserWarning, module='transformers.generation.configuration_utils')

# Note: Directory names should not contain ".", you can replace it with "_" ( 注意！目录不能含有"."，可以替换成"_")
parser = argparse.ArgumentParser()

# Model path, specify the folder or name of the model to load
parser.add_argument('--model_path', required=True, type=str, help="Path to the model, e.g., /path/to/model or a model name from the Hugging Face repository.")
# Input file path, specify the text file to load, default path is data/user_long_text.txt
parser.add_argument('--input_file_path', default="data/user_long_text.txt", type=str, help="Path to the input text file, default is data/user_long_text.txt.")

args = parser.parse_args()

# Check if required parameters are missing and provide a friendly prompt
if not args.model_path:
    print("\033[91mError: The '--model_path' parameter is required. Please specify the path to the model. (错误：必须设置 '--model_path' 参数，请指定模型路径。)\033[0m")
    sys.exit(1)

# Load Tokenizer and model
try:
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map='auto')
    model = model.eval()
    print('\n\033[92mModel loaded successfully (模型加载成功)!\033[0m\nModel path: \033[96m{}\033[0m \nDevice (设备): \033[96m{}\033[0m'.format(args.model_path, model.device))
except Exception as e:
    print(f"\033[91mError loading model: {e} (加载模型时出错，请检查模型路径或文件完整性。)\033[0m")
    sys.exit(1)

def print_announcements():
    print("\033[93m===============  Announcements (公告) ===============\033[0m")
    print("\033[94m1. This model is only recommended for question answering based on long documents, not for chatting. (该模型仅建议用于根据长文档进行问答，不可用于对话聊天)\033[0m")
    print("\033[94m2. The model does not respond based on previous messages. (模型不会根据历史消息回答)\033[0m")
    print("\033[94m3. Do not use this model for any illegal/violating/infringing/malicious inducement activities, otherwise you will bear the legal responsibility. See the model's LICENSE for details. (不可用于任何违法/违规/侵权/恶意诱导等行为，否则自行承担法律责任，详见模型的LICENSE)\033[0m")
    print("\033[93m=====================================================\033[0m\n")
    time.sleep(3) 

def loading_animation(stop_event):
    chars = "/—\\|" 
    while not stop_event.is_set():
        for char in chars:
            sys.stdout.write(f'\r\033[96m Wait ... {char}\033[0m')
            sys.stdout.flush()
            time.sleep(0.2) 

def init_chat_template(generate_config):
    # Special Token IDs
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
        print('\033[91mSystem message is empty (系统消息为空)\033[0m')
        system_ids = torch.tensor([[]], dtype=torch.int64).to(model.device)
    else:
        system_ids = torch.cat([system_start_ids, system_ids], dim=-1).long()
    return system_start_ids, user_start_ids, bot_start_ids, system_ids

def main():
    generate_config = {
        "max_new": 500,
        "do_sample": False,
        "top_p": 1,
        "top_k": 1,
        "temperature": 0.9,
        "repetition_penalty": 1.1,
        "system_message": "你是由哔哩哔哩人工智能平台部自主研发的大语言模型，名为“Index”。你能够根据用户传入的信息，帮助用户完成指定的任务，并生成恰当的、符合要求的回复。",
        "sys_token_id": 0,
        "user_token_id": 3,
        "bot_token_id": 4
    }
    
    system_start_ids, user_start_ids, bot_start_ids, system_ids = init_chat_template(generate_config)

    print_announcements()

    previous_content = None 
    while True:
        ##########   step 1： Read the latest content from the input file   ##########  
        file_content = None
        if os.path.exists(args.input_file_path):
            with open(args.input_file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            # check file changes
            if content != previous_content:
                previous_content = content  # update

                # Print the total character count of the content
                content_length = len(content)
                if content_length > 36000:
                    print(f"\n>> \033[91mThe content of the file is too long (文件内容太长), please do not exceed 40,000 characters (请勿超过36000个字符)!\033[0m")

                # Only show the first and last 100 characters, with ellipses in the middle
                if content_length > 400:
                    part_content = f"{content[:200]}\n................(这里还有{content_length-400} 个字符)({content_length-400} characters)................\n{content[-200:]}"
                else:
                    part_content = content
                file_path = os.path.abspath(args.input_file_path)
                print(f">> \033[93mContent Loaded from file:(从这个文件读取了内容): \033[96m\n{file_path}\033[0m")
                print(f">> \033[93mTotal characters in the file (加载的文件总字符数): \033[96m{content_length}\033[0m")
                print(f"\033[93mPreview of content (部分内容预览):\033[0m\n{part_content}")
        else:
            print(f"\n\033[91mInput file {args.input_file_path} does not exist (输入文件 {args.input_file_path} 不存在), please check the file path (请检查文件路径)。\033[0m")

        ##########   step 2： concat the file with user Instruction   ##########  
        query = "请用中文总结以下内容：\n\n\n\n\n\n" + previous_content + "\n\n\n\n\n\n请用中文总结以上内容：" 
        inputs = tokenizer.encode(query, return_tensors="pt").to(model.device)
        inputs = torch.cat([system_ids, user_start_ids, inputs, bot_start_ids], dim=-1).long()
        #print(f"inputs.size:{inputs.size()}")
        print(f"\n>> \033[96mI will summarize the doc.(将对文件内容进行总结，请耐心等待...)\033[0m")

        # Start loading animation in a separate thread
        stop_event = threading.Event()
        loading_thread = threading.Thread(target=loading_animation, args=(stop_event,))
        loading_thread.start()

        # Generate output
        try:
            outputs = model.generate(inputs, 
                                     max_new_tokens=generate_config['max_new'], 
                                     top_k=generate_config['top_k'], 
                                     top_p=generate_config['top_p'], 
                                     temperature=generate_config['temperature'], 
                                     repetition_penalty=generate_config['repetition_penalty'], 
                                     do_sample=generate_config['do_sample'])
        finally:
            # Ensure loading animation stops
            stop_event.set()
            loading_thread.join()

        # Remove </s>
        if outputs[0][-1] == 2:
            outputs = outputs[:, :-1]

        generated_output = outputs[0][len(inputs[0]):]
        print(f"\n>> \033[92mThe summary of the doc (模型对文档内容的总结如下):\033[0m\n{tokenizer.decode(generated_output)}")
        
        continue_prompt = input(f"\n>> \033[96mContinue summarizing? (在其他窗口更新文件内容后，我可以继续为您总结，输入 y 继续)\033[0m \n(y / n): ")
        if continue_prompt.lower() != 'y':
            break


if __name__ == "__main__":
    main()
