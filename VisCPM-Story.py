from VisCPM import VisCPMChat
from PIL import Image
import os
import re

def sort_key_func(filename):
    numbers = re.findall(r'\d+', filename)
    return [int(num) for num in numbers]

model_path = './checkpoint/pytorch_model.v1.bin'
viscpm_chat = VisCPMChat(model_path, image_safety_checker=True)
file_name = 'VisCPM-Story.txt'
image_path = '../ChatGPT-Story/img1/'
files = os.listdir(image_path)
files.sort(key=sort_key_func)

for filename in files:
    image_file_path = os.path.join(image_path, filename)
    image = Image.open(image_file_path).convert("RGB")
    question = '请根据这幅图说一个儿童童话故事,篇幅长一点,包括几个段落。'
    answer, _, _ = viscpm_chat.chat(image, question)

    print(filename + "\n")
    char_count = len(answer)
    with open(file_name, 'a', encoding='utf-8') as file:
        file.write("Image filename: " + filename + "\n")
        file.write("Question: " + question + "\n")
        file.write("Response: " + answer + "\n")
        file.write("Character Count: " + str(char_count) + "\n\n")

# 如果您单卡显存不足40G，可以引入如下环境变量并将安全模块开关关闭。引入后显存占用约为5G，但推理所需时间会变长。此选项依赖BMInf，需要安装BMInf依赖库。
# export CUDA_MEMORY_CPMBEE_MAX=1g
