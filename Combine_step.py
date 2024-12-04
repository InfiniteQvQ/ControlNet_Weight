import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, StableDiffusionControlNetInpaintPipeline
from diffusers.utils import make_image_grid

device = torch.device("cuda")

init_image_path = "../img/13.jpg"  
mask_image_path = "binary_mask_preserve_size.png"  
canny_image_path = "canny_image.png" 
depth_image_path = "depth_image.png" 
segmentation_image_path = "segmentation_sam.png" 


init_image = Image.open(init_image_path).convert("RGB")
mask_image = Image.open(mask_image_path).convert("L")


init_image_resized = init_image.resize((768, 768))
mask_image_resized = mask_image.resize((768, 768))


control_image_canny = Image.open(canny_image_path).resize((768, 768)).convert("RGB")
control_image_depth = Image.open(depth_image_path).resize((768, 768)).convert("RGB")
control_image_segmentation = Image.open(segmentation_image_path).resize((768, 768)).convert("RGB")


def apply_mask_to_control_image(control_image, mask):
    control_image_np = np.array(control_image)
    mask_np = np.array(mask) / 255  
    control_image_np[mask_np == 0] = 0  
    return Image.fromarray(control_image_np)

control_image_canny = apply_mask_to_control_image(control_image_canny, mask_image_resized)
control_image_depth = apply_mask_to_control_image(control_image_depth, mask_image_resized)
control_image_segmentation = apply_mask_to_control_image(control_image_segmentation, mask_image_resized)


controlnet_canny = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16).to(device)
controlnet_depth = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16).to(device)
controlnet_segmentation = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-seg", torch_dtype=torch.float16).to(device)


def generate_with_controlnet(pipe, controlnet, prompt, control_image, init_image, mask_image, guidance_scale):
    pipe.controlnet = controlnet 
    output = pipe(
        prompt=prompt,
        image=init_image,
        mask_image=mask_image,
        control_image=control_image,
        num_inference_steps=50,
        guidance_scale=guidance_scale,
    ).images[0]
    return output

pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    controlnet=controlnet_canny,  
    torch_dtype=torch.float16,
).to(device)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)


output_canny = generate_with_controlnet(
    pipe, controlnet_canny, "A sofa made of colorful Plasticine in a realistic living room, detailed, soft texture, 8K",
    control_image_canny, init_image_resized, mask_image_resized,
    guidance_scale=15
).resize(init_image.size)

output_depth = generate_with_controlnet(
    pipe, controlnet_depth, "A sofa made of colorful Plasticine in a realistic living room, detailed, soft texture, 8K",
    control_image_depth, output_canny.resize((768, 768)), mask_image_resized,
    guidance_scale=15
).resize(init_image.size)

output_segmentation = generate_with_controlnet(
    pipe, controlnet_segmentation, "A sofa made of colorful Plasticine in a realistic living room, detailed, soft texture, 8K",
    control_image_segmentation, output_depth.resize((768, 768)), mask_image_resized,
    guidance_scale=15
).resize(init_image.size)


output_final = output_segmentation


def plot_steps(init_image, mask_image, output_canny, output_depth, output_segmentation, output_final):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(init_image)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(mask_image, cmap="gray")
    axes[0, 1].set_title("Mask Image")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(output_canny)
    axes[0, 2].set_title("Step 1: Canny")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(output_depth)
    axes[1, 0].set_title("Step 2: Depth")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(output_segmentation)
    axes[1, 1].set_title("Step 3: Segmentation")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(output_final)
    axes[1, 2].set_title("Final Output")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.show()


plot_steps(init_image, mask_image, output_canny, output_depth, output_segmentation, output_final)


output_final.save("output_multistep_inpaint.jpg")
print("Generated image saved as: output_multistep_inpaint.jpg")
