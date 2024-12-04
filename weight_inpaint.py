from PIL import Image
import torch
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, UniPCMultistepScheduler

device = torch.device("cuda")

init_image_path = "../img/91.jpg"
mask_image_path = "binary_mask_preserve_size.png" 
combined_control_image_path = "../img/combine91.png" 

init_image = Image.open(init_image_path).convert("RGB")
mask_image = Image.open(mask_image_path).convert("L")
combined_control_image = Image.open(combined_control_image_path).convert("L")

assert init_image.size == mask_image.size == combined_control_image.size, "Image sizes must match!"

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-seg", torch_dtype=torch.float16).to(device)

pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    controlnet=controlnet,
    torch_dtype=torch.float16,
).to(device)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)


output = pipe(
    prompt="A polished wooden surface with natural grain patterns, replacing the carpet, adding a warm and rustic charm to the room.",
    image=init_image, 
    mask_image=mask_image,  
    control_image=combined_control_image,  
    num_inference_steps=100,
    guidance_scale=20,
).images[0]

output.save("../img/91out.jpg")
output.show()

print("Generated image saved")
