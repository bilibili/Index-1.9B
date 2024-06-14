"""
Code is midified from miniCPM hf_based_demo.py(https://github.com/OpenBMB/MiniCPM/blob/main/demo/hf_based_demo.py)
"""
from typing import Dict
from typing import List
from typing import Tuple
import gradio as gr
import torch
import argparse
from threading import Thread
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    GenerationConfig
)
import warnings

warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default="", type=str, help="")
parser.add_argument('--port', default=8008, type=int, help="")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(args.model_path, 
                                             torch_dtype=torch.bfloat16, 
                                             device_map="auto", 
                                             trust_remote_code=True)

def hf_gen(dialog: List, top_k, top_p, temperature, repetition_penalty, max_dec_len):
    """generate model output with huggingface api

    Args:
        query (str): actual model input.
        top_p (float): only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
        temperature (float): Strictly positive float value used to modulate the logits distribution.
        max_dec_len (int): The maximum numbers of tokens to generate.

    Yields:
        str: real-time generation results of hf model
    """
    inputs = tokenizer.apply_chat_template(dialog, tokenize=False, add_generation_prompt=False)
    enc = tokenizer(inputs, return_tensors="pt").to("cuda")
    streamer = TextIteratorStreamer(tokenizer, **tokenizer.init_kwargs)
    generation_kwargs = dict(
        enc,
        do_sample=True,
        top_k=int(top_k),
        top_p=float(top_p),
        temperature=float(temperature),
        repetition_penalty=float(repetition_penalty),
        max_new_tokens=int(max_dec_len),
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    answer = ""
    for new_text in streamer:
        answer += new_text
        yield answer[len(inputs):]


def generate(chat_history: List, query, top_k, top_p, temperature, repetition_penalty, max_dec_len, system_message):
    """generate after hitting "submit" button

    Args:
        chat_history (List): [[q_1, a_1], [q_2, a_2], ..., [q_n, a_n]]. list that stores all QA records
        query (str): query of current round
        top_p (float): only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
        temperature (float): strictly positive float value used to modulate the logits distribution.
        max_dec_len (int): The maximum numbers of tokens to generate.

    Yields:
        List: [[q_1, a_1], [q_2, a_2], ..., [q_n, a_n], [q_n+1, a_n+1]]. chat_history + QA of current round.
    """
    assert query != "", "Input must not be empty!!!"
    # apply chat template
    model_input = []
    if system_message:
        model_input.append({
            "role": "system",
            "content": system_message
        })
    for q, a in chat_history:
        model_input.append({"role": "user", "content": q})
        model_input.append({"role": "assistant", "content": a})
    model_input.append({"role": "user", "content": query})
    # yield model generation
    chat_history.append([query, ""])
    for answer in hf_gen(model_input, top_k, top_p, temperature, repetition_penalty, max_dec_len):
        # chat_history[-1][1] = answer.strip("</s>")
        chat_history[-1][1] = answer.strip(tokenizer.eos_token)
        yield gr.update(value=""), chat_history


def regenerate(chat_history: List, top_k, top_p, temperature, repetition_penalty, max_dec_len, system_message):
    """re-generate the answer of last round's query

    Args:
        chat_history (List): [[q_1, a_1], [q_2, a_2], ..., [q_n, a_n]]. list that stores all QA records
        top_p (float): only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
        temperature (float): strictly positive float value used to modulate the logits distribution.
        max_dec_len (int): The maximum numbers of tokens to generate.

    Yields:
        List: [[q_1, a_1], [q_2, a_2], ..., [q_n, a_n]]. chat_history
    """
    assert len(chat_history) >= 1, "History is empty. Nothing to regenerate!!"
    # apply chat template
    model_input = []
    if system_message:
        model_input.append({
            "role": "system",
            "content": system_message
        })
    for q, a in chat_history[:-1]:
        model_input.append({"role": "user", "content": q})
        model_input.append({"role": "assistant", "content": a})
    model_input.append({"role": "user", "content": chat_history[-1][0]})
    # yield model generation
    for answer in hf_gen(model_input, top_k, top_p, temperature, repetition_penalty, max_dec_len):
        # chat_history[-1][1] = answer.strip("</s>")
        chat_history[-1][1] = answer.strip(tokenizer.eos_token)
        yield gr.update(value=""), chat_history


def clear_history():
    """clear all chat history

    Returns:
        List: empty chat history
    """
    torch.cuda.empty_cache()
    return []


def reverse_last_round(chat_history):
    """reverse last round QA and keep the chat history before

    Args:
        chat_history (List): [[q_1, a_1], [q_2, a_2], ..., [q_n, a_n]]. list that stores all QA records

    Returns:
        List: [[q_1, a_1], [q_2, a_2], ..., [q_n-1, a_n-1]]. chat_history without last round.
    """
    assert len(chat_history) >= 1, "History is empty. Nothing to reverse!!"
    return chat_history[:-1]

# launch gradio demo
with gr.Blocks(theme="soft") as demo:
    gr.Markdown("""# Index-1.9B Gradio Demo""")

    with gr.Row():
        with gr.Column(scale=1):
            top_k = gr.Slider(1, 10, value=5, step=1, label="top_k")
            top_p = gr.Slider(0, 1, value=0.8, step=0.1, label="top_p")
            temperature = gr.Slider(0.1, 2.0, value=0.3, step=0.1, label="temperature")
            repetition_penalty = gr.Slider(0.1, 2.0, value=1.1, step=0.1, label="repetition_penalty")
            max_dec_len = gr.Slider(1, 4096, value=1024, step=1, label="max_dec_len")
            with gr.Row():
                system_message = gr.Textbox(label="System Message", placeholder="Input your system message", value="你是由哔哩哔哩自主研发的大语言模型，名为“Index”。你能够根据用户传入的信息，帮助用户完成指定的任务，并生成恰当的、符合要求的回复。")
        with gr.Column(scale=10):
            chatbot = gr.Chatbot(bubble_full_width=False, height=500, label='Index-1.9B')
            user_input = gr.Textbox(label="User", placeholder="Input your query here!", lines=8)
            with gr.Row():
                submit = gr.Button("🚀 Submit")
                clear = gr.Button("🧹 Clear")
                regen = gr.Button("🔄 Regenerate")
                reverse = gr.Button("⬅️ Reverse")
    
    submit.click(generate, inputs=[chatbot, user_input, top_k, top_p, temperature, repetition_penalty, max_dec_len, system_message],
                 outputs=[user_input, chatbot])
    regen.click(regenerate, inputs=[chatbot, top_k, top_p, temperature, repetition_penalty, max_dec_len, system_message],
                outputs=[user_input, chatbot])
    clear.click(clear_history, inputs=[], outputs=[chatbot])
    reverse.click(reverse_last_round, inputs=[chatbot], outputs=[chatbot])

demo.queue()
demo.launch(server_name="0.0.0.0", 
            server_port=args.port,
            show_error=True, 
            share=False)
