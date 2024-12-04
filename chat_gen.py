import os
import numpy as np
import torch
import cv2
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import sys

# ============================
# Function to Generate Canny Image
# ============================
def generate_canny_image(image):
    image_np = np.array(image.convert("L")) 
    canny_edges = cv2.Canny(image_np, 100, 200)
    canny_edges = np.repeat(canny_edges[:, :, None], 3, axis=2)  
    return Image.fromarray(canny_edges)

# ============================
# Functions for Depth Map Generation
# ============================
def load_depth_model(path, device):
    sys.path.append(path)
    from depth_anything_v2.dpt import DepthAnythingV2
    model = DepthAnythingV2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024])
    model.load_state_dict(torch.load(f'{path}/checkpoints/depth_anything_v2_vitl.pth', map_location=device))
    model.to(device)
    model.eval()
    return model

def get_depth_image(model, image_path, device):
    raw_img = cv2.imread(image_path)
    depth = model.infer_image(raw_img)
    depth = depth.astype(np.uint8)
    return Image.fromarray(depth)

# ============================
# Functions for Segmentation (SAM)
# ============================
def generate_segmentation_sam(image_path, sam_checkpoint, model_type="vit_h"):
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    mask_generator = SamAutomaticMaskGenerator(sam)

    image = np.array(Image.open(image_path).convert("RGB"))

    masks = mask_generator.generate(image)

    combined_mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)
    for idx, mask in enumerate(masks):
        combined_mask[mask['segmentation']] = idx + 1  # Assign unique ID to each mask

    return Image.fromarray(combined_mask)

# ============================
# Main Function
# ============================
def main():

    image_path = "./input.jpg" 
    sam_checkpoint = "../sam_vit_h.pth" 
    depth_model_path = "../../progress/Depth-Anything-V2"  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    original_image = Image.open(image_path).convert("RGB")

    canny_image = generate_canny_image(original_image)
    canny_image.save("canny.png")

    depth_model = load_depth_model(depth_model_path, device)
    depth_image = get_depth_image(depth_model, image_path, device)
    depth_image.save("depth.png")

    segmentation_sam = generate_segmentation_sam(image_path, sam_checkpoint)
    segmentation_sam.save("seg.png")

if __name__ == "__main__":
    main()
