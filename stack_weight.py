from PIL import Image
import numpy as np

canny_image_path = "canny_image.png"
depth_image_path = "depth_image.png"
segmentation_image_path = "segmentation_sam.png"

control_image_canny = Image.open(canny_image_path).convert("L")
control_image_depth = Image.open(depth_image_path).convert("L")
control_image_segmentation = Image.open(segmentation_image_path).convert("L")

control_image_canny_np = np.array(control_image_canny)
control_image_depth_np = np.array(control_image_depth)
control_image_segmentation_np = np.array(control_image_segmentation)

combined_control_image_np = np.maximum(
    control_image_canny_np,
    np.maximum(control_image_depth_np, control_image_segmentation_np)
)

weight_canny =0.4
weight_depth = 0.9
weight_segmentation = 0.4
combined_control_image_np = (
    weight_canny * control_image_canny_np +
    weight_depth * control_image_depth_np +
    weight_segmentation * control_image_segmentation_np
).astype(np.uint8)

combined_control_image = Image.fromarray(combined_control_image_np)

combined_control_image.save("../img/combine91.png")
combined_control_image.show()
