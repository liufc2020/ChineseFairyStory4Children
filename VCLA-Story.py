import transformers
import visualcla
from peft import PeftModel
import logging
import torch
import os
import re

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=logging.INFO)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--text_model', default=None, type=str,
        help="Path to the pre-trained text encoder")
parser.add_argument('--vision_model', default=None, type=str,
        help="Path to the pre-trained image encoder")
parser.add_argument('--lora_model', default=None, type=str,
        help="Path to the VisualCLA LorA model")
parser.add_argument('--visualcla_model', default='visualcla', type=str,
        help="Path to the merged VisualCLA model")
parser.add_argument('--image_file', default='../ChatGPT-Story/imgs/',required=False,type=str,
        help="The input image file")
parser.add_argument('--gpus', default="0", type=str,
        help="GPU(s) to use for inference")
parser.add_argument('--load_in_8bit',action='store_true',
        help="Whether to load the LLM in 8bit (only supports merged VisualCLA model)")
parser.add_argument('--only_cpu',action='store_true',
        help="Whether to use CPU for inference")
parser.add_argument('--seed', default=-1, type=int,
        help="Random seed, used in transformers.set_seed()")
args = parser.parse_args()
if args.only_cpu is True:
    args.gpus = ""
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

def sort_key_func(filename):
    numbers = re.findall(r'\d+', filename)
    return [int(num) for num in numbers]

def main():
    if args.seed != -1:
        transformers.set_seed(args.seed)

    # Preparation on logging and storing HPs
    logger.setLevel('INFO')
    transformers.utils.logging.set_verbosity('INFO')
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Load model
    load_type = torch.float16
    if torch.cuda.is_available():
        device = torch.device(1)
        device_map='cuda:1'
    else:
        device = torch.device('cpu')
        device_map={'':device}
    load_in_8bit = args.load_in_8bit
    base_model, tokenizer, image_processor = visualcla.get_model_and_tokenizer_and_processor(
        visualcla_model=args.visualcla_model,
        text_model=args.text_model,
        vision_model=args.vision_model,
        lora_model=args.lora_model,
        torch_dtype=load_type,
        default_device=device,
        device_map=device_map,
        load_in_8bit=load_in_8bit and (args.visualcla_model is not None)
    )

    if args.lora_model is not None:
        logger.info(f"Model embedding size is {base_model.text_model.get_input_embeddings().weight.size(0)}. "
                     f"Resize embedding size to {len(tokenizer)}")
        base_model.resize_token_embeddings(len(tokenizer))
        logger.info("Loading LoRA...")
        model = PeftModel.from_pretrained(
            base_model, 
            args.lora_model, 
        )
    else:
        model = base_model

    if device == torch.device('cpu'):
        model.float()
    model.eval()

    logger.info("*** Start Inference ***")

    file_name = 'VCLA-Story.txt'
    with torch.no_grad():
        history = []
        image_path = args.image_file
        if image_path is not None:
            print(f'Image: {image_path}')
        files = os.listdir(image_path)
        files.sort(key=sort_key_func)
        for filename in files:
            text = '请根据这幅图说一个儿童童话故事,篇幅长一点,包括几个段落。'
            image_file_path = os.path.join(image_path, filename)
            try:
                response, history = visualcla.chat(model, image=image_file_path, text=text, history=history)
                print(filename + "\n")
                history = []
                char_count = len(response)
                with open(file_name, 'a', encoding='utf-8') as file:
                    file.write("Image filename: " + filename + "\n")
                    file.write("Question: " + text + "\n")
                    file.write("Response: " + response + "\n")
                    file.write("Character Count: " + str(char_count) + "\n\n")
            except FileNotFoundError:
                print(f"Cannot find file {image_path}. Clear history")
                history = []

    logger.info("*** Exit Inference ***")


if __name__=='__main__':
    main() # usage: python ./scripts/inference/VCLA-testStory.py --load_in_8bit
