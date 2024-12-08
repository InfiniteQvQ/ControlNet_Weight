import argparse
from PIL import Image
import numpy as np

def combine(weights):
    c = Image.open("./canny.png").convert("L")
    depth = Image.open("./depth.png").convert("L")
    seg = Image.open("./seg.png").convert("L")

    canny = np.array(c)
    dep = np.array(depth)
    segp = np.array(seg)

    combined = (weights[0] * canny + weights[1] * dep + weights[2] * segp).astype(np.uint8)

    img = Image.fromarray(combined)
    img.save("./combine.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--weights", nargs=3, type=float, default=[0.4, 0.9, 0.4])

    args = parser.parse_args()

    combine(tuple(args.weights))
