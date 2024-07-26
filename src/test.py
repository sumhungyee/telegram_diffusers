import torch
import gc
import PIL
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler, DPMSolverSinglestepScheduler
from compel import Compel, ReturnedEmbeddingsType

# references from https://github.com/huggingface/diffusers/issues/2136, https://github.com/damian0815/compel/blob/main/README.md
pipe: StableDiffusionXLPipeline = StableDiffusionXLPipeline.from_single_file(
    "./model/SDXLFaetastic_v24.safetensors",
    #safety_checker=None,
    torch_dtype=torch.float16,
)
pipe.safety_checker = None
pipe.scheduler =  DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config)

pipe = pipe.to("cuda")
while input("type something: "):
    prompt = """An image of a beautiful girl with pink hair watching beautiful fireworks in the night sky, semirealistic """ + ", HDR, extremely detailed, best quality, masterpiece, aesthetic"
    negative_prompt = ""

    with torch.no_grad():
        compel = Compel(
            tokenizer=[pipe.tokenizer, pipe.tokenizer_2], 
            text_encoder=[pipe.text_encoder, pipe.text_encoder_2], 
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, 
            requires_pooled=[False, True],
            truncate_long_prompts=False
            )
        prompt_embeds, pooled_prompt_embeds = compel(prompt)
        negative_prompt_embeds, negative_pooled_prompt_embeds = compel(negative_prompt)
        prompt_embeds, negative_prompt_embeds = compel.pad_conditioning_tensors_to_same_length([prompt_embeds, negative_prompt_embeds])
        
    image: PIL.Image.Image = pipe(
        prompt_embeds = prompt_embeds, 
        pooled_prompt_embeds=pooled_prompt_embeds, 
        negative_prompt_embeds = negative_prompt_embeds, 
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds, 
        height=768, 
        width=1344, 
        num_inference_steps=60, 
        num_images_per_prompt=1,
        #guidance_scale=7.5
        ).images[0]
    image.show()

    del prompt_embeds, pooled_prompt_embeds, negative_prompt_embeds, negative_pooled_prompt_embeds, image
    gc.collect()
    torch.cuda.empty_cache()
    