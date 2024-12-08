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
m = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-ade-semantic")

ADE20K_CLASSES = [
    "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed", "windowpane", "grass",
    "cabinet", "sidewalk", "person", "earth", "door", "table", "mountain", "plant", "curtain", "chair",
    "car", "water", "painting", "sofa", "shelf", "house", "sea", "mirror", "rug", "field", "oven", 
    "Range hoods", "cup", "button", "coat", "cloth", "mat", "furniture"
]

def gen(name):
    img = Image.open("./input.jpg").convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    
    with torch.no_grad():
        outputs = m(**inputs)
    
    pre = processor.post_process_semantic_segmentation(
        outputs=outputs,
        target_sizes=[img.size[::-1]]
    )[0]
    
    pre = pre.cpu().numpy()
    
    id = ADE20K_CLASSES.index(name)
    
    mask = (pre == name).astype(np.float32)
    mask_np = (mask * 255).astype(np.uint8)
    
    bmask = Image.fromarray(mask_np , mode="L")
    bmask.save("./mask.png")
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("name", type=str)
   
    args = parser.parse_args()
    
    gen( args.target_class)
