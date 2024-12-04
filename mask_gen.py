from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-ade-semantic")
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-ade-semantic")


image_path = "../img/91.jpg" 
image = Image.open(image_path).convert("RGB")

inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

pred_semantic_map = processor.post_process_semantic_segmentation(
    outputs=outputs,
    target_sizes=[image.size[::-1]] 
)[0]


pred_semantic_map = pred_semantic_map.cpu().numpy()

#could add more
ADE20K_CLASSES = [
    "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed", "windowpane", "grass",
    "cabinet", "sidewalk", "person", "earth", "door", "table", "mountain", "plant", "curtain", "chair",
    "car", "water", "painting", "sofa", "shelf", "house", "sea", "mirror", "rug", "field", "oven", "Range hoods", "cup", "button", "coat" , "cloth", "mat"
 
]


target_class = "rug" 
if target_class in ADE20K_CLASSES:
    target_class_id = ADE20K_CLASSES.index(target_class)
    print(f"Class '{target_class}' has ID: {target_class_id}")
else:
    raise ValueError(f"Class '{target_class}' not found in ADE20K_CLASSES.")

target_mask = (pred_semantic_map == target_class_id).astype(np.float32)


def prepare_binary_mask(target_mask):
    binary_mask = target_mask 
    binary_mask = torch.from_numpy(binary_mask).float()
    return binary_mask

binary_mask = prepare_binary_mask(target_mask)


binary_mask_np = (binary_mask.numpy() * 255).astype(np.uint8)
binary_mask_img = Image.fromarray(binary_mask_np, mode="L")
binary_mask_img.save("binary_mask_preserve_size.png")



print("Binary Mask Shape:", binary_mask_np.shape)
