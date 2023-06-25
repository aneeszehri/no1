from typing import  Dict, List, Any
import torch
import os
import PIL
from PIL import Image

from torch import autocast
from diffusers import StableDiffusionPipeline
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
        self.pipe = self.pipe.to(device)


    def __call__(self, data: Any) -> List[List[Dict[str, float]]]:
        """
        Args:
            data (:obj:):
                includes the input data and the parameters for the inference.
        Return:
            A :obj:`dict`:. base64 encoded image
        """
        postive_prompt = data.pop("postive_prompt", data)
        negative_prompt = data.pop("negative_prompt", None)
        height = data.pop("height", 512)
        width = data.pop("width", 512)
        guidance_scale = data.pop("guidance_scale", 7.5)

        # run inference pipeline
        with autocast(device.type):
            if negative_prompt is None:
                image = self.pipe(prompt = postive_prompt ,height = height ,width = width ,guidance_scale=float(guidance_scale))["sample"][0]
            else:
                image = self.pipe(prompt = postive_prompt ,negative_prompt = negative_prompt,height = height ,width = width ,guidance_scale=float(guidance_scale))["sample"][0]

        # encode image as base 64
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue())

        # postprocess the prediction
        return {"image": img_str.decode()}