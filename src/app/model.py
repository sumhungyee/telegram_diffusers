import torch
import gc
import PIL
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler, DPMSolverSinglestepScheduler
from compel import Compel, ReturnedEmbeddingsType

def get_txt_to_img_pipeline(
        path="./model/SDXLFaetastic_v24.safetensors", safety = False, scheduler = DPMSolverSinglestepScheduler
        ) -> StableDiffusionXLPipeline:
    pipeline: StableDiffusionXLPipeline = StableDiffusionXLPipeline.from_single_file(
        path,
        torch_dtype=torch.float16,
    )
    if not safety:
        pipeline.safety_checker = None
    pipeline.scheduler =  scheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to("cuda")
    return pipeline

def generate_image(pipeline, prompt, negative_prompt = "", image_type = "square", num_inference_steps = 60) -> PIL.Image.Image:

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
        
    image: PIL.Image.Image = pipeline(
        prompt_embeds = prompt_embeds, 
        pooled_prompt_embeds = pooled_prompt_embeds, 
        negative_prompt_embeds = negative_prompt_embeds, 
        negative_pooled_prompt_embeds = negative_pooled_prompt_embeds, 
        height = height, 
        width = width, 
        num_inference_steps = num_inference_steps, 
        num_images_per_prompt = 1,
        #guidance_scale=7.5
        ).images[0]
    
    return image
        
def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()

    