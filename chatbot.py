import openai
import os
import subprocess
import json
import matplotlib.pyplot as plt
import argparse
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import torch
from PIL import Image
import numpy as np
import argparse
from PIL import Image
import torch
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, UniPCMultistepScheduler

openai.api_key = os.getenv("OPENAI_API_KEY")


conversation_history = [
    {"role": "system", "content": (
        "You are an intelligent assistant specialized in image editing and model weighting. "
        "When a user describes their desired image modification,  you will return a JSON-like string in the following format:"
        "{"
        "\"things_to_change\": <string>, "
        "\"segmentation_weight\": <float>, "
        "\"depth_map_weight\": <float>, "
        "\"canny_edge_weight\": <float>, "
        "\"generated_prompt\": <string>"
        "}."
        "The weights should between 0 and 1 and reflect the importance of each control modality to achieve the desired change. "
        "in the first parameter, return only one word, that refers to the thing to change"
        "In the last parameter, return only the prompt refers to the things going to change, not contain the original one, and generate prompt to make the new things looks good"
        "The prompt should clearly describe the desired result, ensuring high-quality output."
    )}
]

def query_openai_with_memory(user_input):
  
    conversation_history.append({"role": "user", "content": user_input})

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=conversation_history
    )

    assistant_response = response['choices'][0]['message']['content']
    conversation_history.append({"role": "assistant", "content": assistant_response})
    
    return assistant_response

def call_chat_mask(target_class):
    result = subprocess.run(
        ["python", "chat_mask.py", target_class], capture_output=True, text=True)
  

def gen_3map():
    subprocess.run(["python", "chat_gen.py"], capture_output=True, text=True)

def pin(w):
    subprocess.run(
        ["python", "chat_pin.py", "--weights", str(w[0]), str(w[1]), str(w[2])],
        capture_output=True,
        text=True
    )

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

def inpaint(p):
    generate_image(p)

if __name__ == "__main__":
    print("Hi! please input your text: ")
    while True:
        user_message = input("\nYou: ")
        if user_message.lower() == "exit":
            print("Goodbye!")
            break
        response = query_openai_with_memory(user_message)
        print(f"{response}")
        response_list = json.loads(response)
        call_chat_mask(response_list["things_to_change"])
        print("mask done")
        gen_3map()
        print("gen done")
        w = [
            float(response_list["canny_edge_weight"]),
            float(response_list["depth_map_weight"]),
            float(response_list["segmentation_weight"])
        ]
        pin(w)
        print("pin done")
        inpaint(response_list["generated_prompt"])



