import argparse
from PIL import Image
import torch
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, UniPCMultistepScheduler

def generate_image(prompt):
    device = torch.device("cuda")

    init_image = Image.open("./input.jpg").convert("RGB")
    mask_image = Image.open("./binary_mask_preserve_size.png").convert("L")
    combined_control_image = Image.open("./combine.png").convert("L")

    assert init_image.size == mask_image.size == combined_control_image.size, "Image sizes must match!"

    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-seg", torch_dtype=torch.float16).to(device)

    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        controlnet=controlnet,
        torch_dtype=torch.float16,
    ).to(device)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    output = pipe(
        prompt=prompt,
        image=init_image,
        mask_image=mask_image,
        control_image=combined_control_image,
        num_inference_steps=100,
        guidance_scale=20,
    ).images[0]

    output.save("./output.png")
    output.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an inpainted image using Stable Diffusion with ControlNet.")
    parser.add_argument("prompt", type=str, help="Prompt describing the desired changes.")
    args = parser.parse_args()

    generate_image(args.prompt)
