import torch
import random
import argparse
import gc
import os
import PIL
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler, DPMSolverSinglestepScheduler, DiffusionPipeline
from compel import Compel, ReturnedEmbeddingsType


def get_txt_to_img_pipeline(
        path=os.getenv("SDXL_PATH"), safety = False, scheduler = DPMSolverSinglestepScheduler
        ) -> StableDiffusionXLPipeline:
    pipeline: StableDiffusionXLPipeline = StableDiffusionXLPipeline.from_single_file(
        path,
        torch_dtype=torch.float16,
    )
    if not safety:
        pipeline.safety_checker = None
    pipeline.scheduler = scheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to("cuda")
    return pipeline

def generate_image(pipeline: DiffusionPipeline, prompt, random_seed, negative_prompt = "", image_type = "square", num_inference_steps = 60) -> PIL.Image.Image: # random seed will always be filled

    match image_type:
        case "square":
            height, width = 1024, 1024
        case "landscape":
            height, width = 768, 1344
        case "portrait":
            height, width = 1344, 768
        case _:
            raise Exception("Dimensions incorrect!")
        
    with torch.no_grad():
        compel = Compel(
            tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2], 
            text_encoder=[pipeline.text_encoder, pipeline.text_encoder_2], 
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, 
            requires_pooled=[False, True],
            truncate_long_prompts=False
            )
        prompt_embeds, pooled_prompt_embeds = compel(prompt)
        negative_prompt_embeds, negative_pooled_prompt_embeds = compel(negative_prompt)
        prompt_embeds, negative_prompt_embeds = compel.pad_conditioning_tensors_to_same_length([prompt_embeds, negative_prompt_embeds])
    
    if random_seed != -1:
        generator = torch.Generator(device="cuda").manual_seed(random_seed)
    else:
        generator = torch.Generator(device="cuda").manual_seed(random.randint(0, 999999999))

    image: PIL.Image.Image = pipeline(
        prompt_embeds = prompt_embeds, 
        pooled_prompt_embeds = pooled_prompt_embeds, 
        negative_prompt_embeds = negative_prompt_embeds, 
        negative_pooled_prompt_embeds = negative_pooled_prompt_embeds, 
        height = height, 
        width = width, 
        num_inference_steps = num_inference_steps, 
        num_images_per_prompt = 1,
        generator=generator
        ).images[0]
    
    return image
        
def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=["/generate"])
    parser.add_argument('-p', '--prompt', type=str, required=True)
    parser.add_argument('-n', '--negprompt', type=str, required=False, default="")
    parser.add_argument(
        '-o', '--orientation', type=str, choices=['landscape', 'portrait', 'square'], required=False, default="square"
        )
    parser.add_argument('-s', '--steps', type=int, choices=[10 * i for i in range(3, 10)], required=False, default=60)
    parser.add_argument('-r', '--randomseed', type=int, required=False, default=-1)
    return parser

def replace_curly_quotes(input_string):
    replacements = {
        '‘': "'",
        '’': "'",
        '“': '"',
        '”': '"'
    }
    
    for curly_quote, straight_quote in replacements.items():
        input_string = input_string.replace(curly_quote, straight_quote)
    
    return input_string