from typing import  Dict, List, Any
import torch
import os
import PIL
from PIL import Image

from torch import autocast
from diffusers import StableDiffusionPipeline,EulerDiscreteScheduler
import base64
from io import BytesIO


# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if device.type != 'cuda':
    raise ValueError("need to run on GPU")

class EndpointHandler():
    def __init__(self, path=""):
        # load the optimized model
        self.pipe = StableDiffusionPipeline.from_pretrained(path, torch_dtype=torch.float16,low_cpu_mem_usage=False)
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to(device)


    def __call__(self, data: Any) -> Dict[str, str]:
        """
        Args:
            data (Any): Includes the input data and the parameters for the inference.

        Returns:
            Dict[str, str]: Dictionary with the base64 encoded image.
        """
        inputs = data.pop("inputs", data)
        # positive_prompt = data.pop("positive_prompt", None)
        negative_prompt = data.pop("negative_prompt", None)
        height = data.pop("height", 512)
        width = data.pop("width", 512)
        inference_steps = data.pop("inference_steps", 25)
        guidance_scale = data.pop("guidance_scale", 7.5)

        # Run inference pipeline
        with autocast(device.type):
            if negative_prompt is None:
                print(str(inputs), str(height), str(width), str(guidance_scale))
                image = self.pipe(prompt=inputs, height=height, width=width, guidance_scale=float(guidance_scale),num_inference_steps=inference_steps)
                image = image.images[0]
            else:
                print(str(inputs), str(height), str(negative_prompt), str(width), str(guidance_scale))
                image = self.pipe(prompt=inputs, negative_prompt=negative_prompt, height=height, width=width, guidance_scale=float(guidance_scale),num_inference_steps=inference_steps)
                image = image.images[0]

        # Encode image as base64
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue())

        # Postprocess the prediction
        return {"image": img_str.decode()}

    def decode_base64_image(self, image_string):
        base64_image = base64.b64decode(image_string)
        buffer = BytesIO(base64_image)
        image = Image.open(buffer)
        return image
