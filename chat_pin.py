import argparse
from PIL import Image
import numpy as np

def combine_control_images(weights):
    control_image_canny = Image.open("./canny.png").convert("L")
    control_image_depth = Image.open("./depth.png").convert("L")
    control_image_segmentation = Image.open("./seg.png").convert("L")

    control_image_canny_np = np.array(control_image_canny)
    control_image_depth_np = np.array(control_image_depth)
    control_image_segmentation_np = np.array(control_image_segmentation)

    combined_control_image_np = (
        weights[0] * control_image_canny_np +
        weights[1] * control_image_depth_np +
        weights[2] * control_image_segmentation_np
    ).astype(np.uint8)

    combined_control_image = Image.fromarray(combined_control_image_np)
    combined_control_image.save("./combine.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine control images with weights.")
    parser.add_argument("--weights", nargs=3, type=float, default=[0.4, 0.9, 0.4],
                        help="Weights for Canny, Depth, and Segmentation images (default: 0.4, 0.9, 0.4).")

    args = parser.parse_args()

    combine_control_images(tuple(args.weights))
