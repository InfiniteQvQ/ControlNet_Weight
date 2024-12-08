import argparse
from PIL import Image
import torch
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, UniPCMultistepScheduler

def gen(prompt):
    device = torch.device("cuda")

    init_img = Image.open("./input.jpg").convert("RGB")
    mask = Image.open("./binary_mask_preserve_size.png").convert("L")
    combine = Image.open("./combine.png").convert("L")

    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-seg", torch_dtype=torch.float16).to(device)

    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        controlnet=controlnet,
        torch_dtype=torch.float16,
    ).to(device)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    output = pipe(
        prompt=prompt,
        image=init_img,
        mask_image=mask,
        control_image=combine,
        num_inference_steps=100,
        guidance_scale=20,
    ).images[0]

    output.save("./output.png")
    output.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("prompt", type=str)
    args = parser.parse_args()

    gen(args.prompt)
