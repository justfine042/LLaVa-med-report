import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import csv

from llava import LlavaLlamaForCausalLM
from llava.conversation import conv_templates
from llava.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria

from PIL import Image
import random
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


detail_describe_instructions = [
    "Describe the following image in detail.",
    "Provide a detailed description of the given image.",
    "Give an elaborate explanation of the image you see.",
    "Share a comprehensive rundown of the presented image.",
    "Offer a thorough analysis of the image.",
    "Explain the various aspects of the image before you.",
    "Clarify the contents of the displayed image with great detail.",
    "Characterize the image using a well-detailed description.",
    "Break down the elements of the image in a detailed manner.",
    "Walk through the important details of the image.",
    "Portray the image with a rich, descriptive narrative.",
    "Narrate the contents of the image with precision.",
    "Analyze the image in a comprehensive and detailed manner.",
    "Illustrate the image through a descriptive explanation.",
    "Examine the image closely and share its details.",
    "Write an exhaustive depiction of the given image.",
]

concise_describe_instructions = [
    "Describe the following image concisely.",
    "Provide a brief description of the given image.",
    "Offer a succinct explanation of the picture presented.",
    "Summarize the visual content of the following image.",
    "Give a short and clear explanation of the subsequent image.",
    "Share a concise interpretation of the image provided.",
    "Present a compact description of the photo's key features.",
    "Relay a brief, clear account of the picture shown.",
    "Render a clear and concise summary of the photo below.",
    "Write a terse but informative summary of the following picture.",
    "Create a compact narrative representing the image presented.",
]

prompt_pool = detail_describe_instructions + concise_describe_instructions

prompt_pool = [ "Describe the following image in detail."]


def patch_config(config):
    patch_dict = {
        "use_mm_proj": True,
        "mm_vision_tower": "openai/clip-vit-large-patch14",
        "mm_hidden_size": 1024
    }

    cfg = AutoConfig.from_pretrained(config)
    if not hasattr(cfg, "mm_vision_tower"):
        print(f'`mm_vision_tower` not found in `{config}`, applying patch and save to disk.')
        for k, v in patch_dict.items():
            setattr(cfg, k, v)
        cfg.save_pretrained(config)


# new stopping implementation
class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        else:
            outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
        return False


def eval_model(args):
    # Model
    disable_torch_init() # 
    model_name = os.path.expanduser(args.model_name) # llava-med-v1.5-mistral-7b
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if args.mm_projector is None:
        patch_config(model_name)
        
        print(model_name)
        if "BiomedCLIP" in model_name or "biomed_clip" in model_name:
            model = LlavaLlamaForCausalLM.from_pretrained(model_name, use_cache=True).cuda()
            model = model.to(torch.float16)
            image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
            
            openai_vision_tower = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
            vision_config = openai_vision_tower.config
            vision_tower = model.model.vision_tower[0]
            vision_tower.to(device='cuda', dtype=torch.float16)
            setattr(vision_tower, 'config', vision_config)
        else:
            model = LlavaLlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, use_cache=True).cuda()
            image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower, torch_dtype=torch.float16)
            vision_tower = model.model.vision_tower[0]
            vision_tower.to(device='cuda', dtype=torch.float16)
            

        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

        vision_config = vision_tower.config
        vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
        vision_config.use_im_start_end = mm_use_im_start_end
        if mm_use_im_start_end:
            vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
        image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2
    else:
        # in case of using a pretrained model with only a MLP projector weights
        model = LlavaLlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, use_cache=True).cuda()

        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

        vision_tower = CLIPVisionModel.from_pretrained(args.vision_tower, torch_dtype=torch.float16).cuda()

        if "BiomedCLIP" in model.config.mm_vision_tower:
            image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
        else:
            image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower, torch_dtype=torch.float16)


        vision_config = vision_tower.config
        vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
        vision_config.use_im_start_end = mm_use_im_start_end
        if mm_use_im_start_end:
            vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])

        image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2

        mm_projector = torch.nn.Linear(vision_config.hidden_size, model.config.hidden_size)
        mm_projector_weights = torch.load(args.mm_projector, map_location='cpu')
        mm_projector.load_state_dict({k.split('.')[-1]: v for k, v in mm_projector_weights.items()})

        model.model.mm_projector = mm_projector.cuda().half()
        model.model.vision_tower = [vision_tower]

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file_gt = os.path.expanduser(args.answers_file_gt)
    answers_file_pred = os.path.expanduser(args.answers_file_pred)
    os.makedirs(os.path.dirname(answers_file_gt), exist_ok=True)
    os.makedirs(os.path.dirname(answers_file_pred), exist_ok=True)

    with open(answers_file_gt, "w", newline='') as gt_file, open(answers_file_pred, "w", newline='') as pred_file:
        gt_writer = csv.writer(gt_file)
        pred_writer = csv.writer(pred_file)
        gt_writer.writerow(["study_id", "report"])
        pred_writer.writerow(["study_id", "report"])

        batch_size = args.batch_size
        num_batches = math.ceil(len(questions) / batch_size)

        for batch_idx in tqdm(range(num_batches)):
            batch_questions = questions[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            batch_inputs = []
            batch_images = []
            batch_prompts = []
            batch_ids = []
            batch_gt_answers = []

            for line in batch_questions:
                idx = line["id"]
                try:
                    question = line["conversations"][0]
                    gt_ans = line["conversations"][1]
                except:
                    question = line["conversations"][0]
                    gt_ans = line["conversations"][1]

                qs = question['value']
                qs = qs.replace('<image>', '').strip()
                cur_prompt = qs

                if 'image' in line:
                    image_file = line["image"]
                    image = Image.open(os.path.join(args.image_folder, image_file))
                    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                    images = image_tensor.unsqueeze(0).half().cuda()
                    if getattr(model.config, 'mm_use_im_start_end', False):
                        qs = qs + '\n' + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IM_END_TOKEN
                    else:
                        qs = qs + '\n' + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
                    cur_prompt = cur_prompt + '\n' + '<image>'
                else:
                    images = None

                if args.conv_mode == 'simple_legacy':
                    qs += '\n\n### Response:'
                assert gt_ans['from'] == 'report'
                conv = conv_templates[args.conv_mode].copy()
                conv.append_message(conv.roles[0], qs)
                prompt = conv.get_prompt()
                inputs = tokenizer([prompt], return_tensors='pt')  

                batch_inputs.append(inputs['input_ids'])
                batch_images.append(images)
                batch_prompts.append(cur_prompt)
                batch_ids.append(idx)
                batch_gt_answers.append(gt_ans['value'])

            input_ids = torch.cat(batch_inputs, dim=0).cuda()
            print(input_ids.shape)
            images = torch.cat(batch_images, dim=0) if batch_images[0] is not None else None

            keywords = ['###']
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images,
                    do_sample=True,
                    temperature=0.7,
                    max_new_tokens=1024,
                    stopping_criteria=[stopping_criteria])

            input_token_len = input_ids.shape[1]
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)

            for i, (idx, cur_prompt, output, gt_ans) in enumerate(zip(batch_ids, batch_prompts, outputs, batch_gt_answers)):
                if args.conv_mode == 'simple_legacy':
                    while True:
                        cur_len = len(output)
                        output = output.strip()
                        for pattern in ['###', 'Assistant:', 'Response:']:
                            if output.startswith(pattern):
                                output = output[len(pattern):].strip()
                        if len(output) == cur_len:
                            break

                try:
                    index = output.index(conv.sep)
                except ValueError:
                    output += conv.sep
                    index = output.index(conv.sep)

                output = output[:index].strip()

                if args.answer_prompter:
                    outputs_reasoning = output
                    inputs = tokenizer([prompt + outputs_reasoning + ' ###\nANSWER:'], return_tensors='pt')  # 确保返回的是张量

                    input_ids = inputs['input_ids'].cuda()

                    keywords = ['###']
                    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

                    with torch.inference_mode():
                        output_ids = model.generate(
                            input_ids,
                            images=images,
                            do_sample=True,
                            temperature=0.7,
                            max_new_tokens=64,
                            stopping_criteria=[stopping_criteria])

                    input_token_len = input_ids.shape[1]
                    output = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]

                    try:
                        index = output.index(conv.sep)
                    except ValueError:
                        output += conv.sep
                        index = output.index(conv.sep)

                    output = output[:index].strip()
                    output = outputs_reasoning + '\n The answer is ' + output

                gt_writer.writerow([idx, gt_ans])
                pred_writer.writerow([idx, output])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--answers-file-gt", type=str, default="answer_gt.csv")
    parser.add_argument("--answers-file-pred", type=str, default="answer_pred.csv")
    parser.add_argument("--mm-projector", type=str, default=None)
    parser.add_argument("--vision-tower", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="simple")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)  # 添加batch_size参数
    args = parser.parse_args()

    eval_model(args)