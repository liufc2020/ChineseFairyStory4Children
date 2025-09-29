from transformers import AutoTokenizer, AutoModel
import os
import re

def sort_key_func(filename):
    numbers = re.findall(r'\d+', filename)
    return [int(num) for num in numbers]

tokenizer = AutoTokenizer.from_pretrained("./visualglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("./visualglm-6b", trust_remote_code=True).half().cuda()

file_name = 'Visualglm-Story.txt'
image_path = '../ChatGPT-Story/imgs/'
files = os.listdir(image_path)
files.sort(key=sort_key_func)
for filename in files:
    text = '请根据这幅图说一个儿童童话故事,篇幅长一点,包括几个段落。'
    image_file_path = os.path.join(image_path, filename)
    response, history = model.chat(tokenizer, image_file_path, text, history=[])
    print(filename + "\n")
    char_count = len(response)
    with open(file_name, 'a', encoding='utf-8') as file:
        file.write("Image filename: " + filename + "\n")
        file.write("Question: " + text + "\n")
        file.write("Response: " + response + "\n")
        file.write("Character Count: " + str(char_count) + "\n\n")

