import argparse
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import torch
from PIL import Image
import numpy as np
import argparse
from PIL import Image
import torch
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, UniPCMultistepScheduler

processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-ade-semantic")
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-ade-semantic")

ADE20K_CLASSES = [
    "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed", "windowpane", "grass",
    "cabinet", "sidewalk", "person", "earth", "door", "table", "mountain", "plant", "curtain", "chair",
    "car", "water", "painting", "sofa", "shelf", "house", "sea", "mirror", "rug", "field", "oven", 
    "Range hoods", "cup", "button", "coat", "cloth", "mat", "furniture"
]

def generate_mask( target_class):
    if target_class not in ADE20K_CLASSES:
        raise ValueError(f"Class '{target_class}' not found in ADE20K_CLASSES. Available classes: {', '.join(ADE20K_CLASSES)}")
    
    image = Image.open("./input.jpg").convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    pred_semantic_map = processor.post_process_semantic_segmentation(
        outputs=outputs,
        target_sizes=[image.size[::-1]]
    )[0]
    
    pred_semantic_map = pred_semantic_map.cpu().numpy()
    
    target_class_id = ADE20K_CLASSES.index(target_class)
    print(f"Class '{target_class}' has ID: {target_class_id}")
    
    target_mask = (pred_semantic_map == target_class_id).astype(np.float32)
    binary_mask_np = (target_mask * 255).astype(np.uint8)
    
    binary_mask_img = Image.fromarray(binary_mask_np, mode="L")
    binary_mask_img.save("./binary_mask_preserve_size.png")
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a binary mask for a specific class from an image.")

    parser.add_argument("target_class", type=str,  help="Target class for which to generate the mask.")
   
    args = parser.parse_args()
    
    generate_mask( args.target_class)
