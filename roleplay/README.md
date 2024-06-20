# Index 1.9B Character
本项目是一个支持Index-1.9B角色模型的推理框架，目前内置了`三三`的角色。

# 🌏️ 下载模型权重
下载以下模型到本地，并修改配置`config/config.json`
* [bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5)
* [Index-1.9B-Character](https://huggingface.co/IndexTeam/Index-1.9B-Character)

# 🥳 配置环境
1. 安装conda环境`conda create -n index python=3.10`
2. 激活对应的环境`conda activate index`
3. 安装torch，不要使用清华源 `pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cu116`
4. 对于Windows用户，安装faiss-gpu请用conda `conda install conda-forge::faiss-gpu`
5. 安装对应的依赖`pip install -r requirements.txt`

# 🤩 使用

## 欢迎使用我们的demo
* 请在命令行输入`python hf_based_demo.py`
    ![gradio demo](git_src/demo.png)

* 使用指南
    * 如果需要创建您自己的角色，请准备一个类似[character/三三.csv](character/三三.csv)的对话语料库（注意，文件名请与您要创建的角色名称保持一致）和对应角色的描述，点击`生成角色`即可创建成功。
    * 如果已经创建好对应的角色，请您直接在Role name里输入您想对话的角色，并输入query，点击submit，即可对话。

## 针对实时对话要求

* 针对已经支持的角色

你可以直接通过命令行的方式 `python realtime_chat.py --role_name your_role_name`

* 针对尚未支持的角色

你可以通过命令行`python realtime_chat.py --role_name your_rolename --role_info your_role_desc --role_dialog_file your_role_dialog_path`

* 如何结束对话

输入`stop`结束对话

## 针对非实时对话要求

* 针对已经支持的角色

```python
from index_play import IndexRolePlay

chatbox = IndexRolePlay(role_name="your_role_name")

# 以下两种方式都支持
chatbox.infer_with_question("your_question")
chatbox.infer_with_question_file("your_question_path")
```

* 针对尚未支持的角色

你需要先提供一个类似[character/三三.csv](character/三三.csv)的角色对话库以及对应的角色信息

```python
from index_play import IndexRolePlay
chatbox = IndexRolePlay(role_name="your_role_name", role_info="your description", role_dialog_file="your_dialog_path")

# 以下两种方式都支持
chatbox.infer_with_question("your_question")
chatbox.infer_with_question_file("your_question_path")
```

# 声明
我们在模型训练的过程中，利用合规性检测等方法，最大限度地确保使用数据的合规性。虽然我们已竭尽全力确保在模型训练中使用数据的合法性，但鉴于模型的复杂性和使用场景的多样性，仍然可能存在一些尚未预料到的潜在问题。因此，对于任何使用开源模型而导致的风险和问题，包括但不限于数据安全问题，因误导、滥用、传播或不当应用带来的风险和问题，我们将不承担任何责任。

用户应对其创建的角色和上传的语料负全责。用户需确保其角色设定和语料内容符合所有适用的法律法规，并不得含有任何违法或不当内容。对于因用户行为导致的任何法律后果，用户应独立承担所有责任。

我们强烈呼吁所有使用者，不要利用Index-1.9B-character进行任何危害国家安全或违法的活动。希望大家可以遵循以上的原则，共建健康的科技发展环境。
