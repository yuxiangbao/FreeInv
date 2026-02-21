
import abc
from typing import Optional, Union, Tuple, List, Callable, Dict
from diffusers import LCMScheduler
from models.ddpm_edit.inversion_utils import inversion_forward_process, inversion_reverse_process
from models.ddpm_edit.ptp_classes import AttentionControl, AttentionStore, load_512
from models.ddpm_edit.ptp_utils import register_attention_control
import math
import torch
import torch.nn.functional as nnf
from torch import autocast, inference_mode
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, StableDiffusionPipeline
import numpy as np
from PIL import Image
import os
import json
import random
import argparse
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from utils.utils import txt_draw,latent2image

device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
    'cpu')

NUM_STEPS = 100

def setup_seed(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

cfg_scale_src = 3.5
cfg_scale_tar = 15
eta = 1
skip = 36

model_id = "./SD2.1"
ldm_stable = StableDiffusionPipeline.from_pretrained(model_id).to(device)
ldm_stable.scheduler = DDIMScheduler.from_config(model_id, subfolder = "scheduler")

@torch.no_grad()
def edit_image_ddpm_edit(source_prompt, target_prompt,
                        width=512, height=512, seed=0, img=None, strength=0.7,
                        cross_replace_steps=0.8, self_replace_steps=0.4, eta=0.1, thresh_e=0.3, thresh_m=0.3, denoise=True):
    ldm_stable.scheduler.set_timesteps(NUM_STEPS)

    # load image
    offsets=(0,0,0,0)
    x0 = load_512(img, *offsets, device)
    
    # vae encode image
    with autocast("cuda"), inference_mode():
        w0 = (ldm_stable.vae.encode(x0).latent_dist.mode() * 0.18215).float()

    # find Zs and wts - forward process
    wt, zs, wts = inversion_forward_process(ldm_stable, w0, etas=eta, prompt=source_prompt, cfg_scale=cfg_scale_src, prog_bar=True, num_inference_steps=NUM_STEPS)

    controller = AttentionStore()
    register_attention_control(ldm_stable, controller)
    r0, _ = inversion_reverse_process(ldm_stable, xT=wts[NUM_STEPS-skip], etas=eta, prompts=[target_prompt], cfg_scales=[cfg_scale_tar], prog_bar=True, zs=zs[:(NUM_STEPS-skip)], controller=controller)

    image_instruct = txt_draw(f"source prompt: {source_prompt}\ntarget prompt: {target_prompt}")
    img_gt = x0 *0.5 + 0.5
    img_recon = ldm_stable.vae.decode(1 / 0.18215 * w0).sample *0.5 + 0.5
    img_edit = ldm_stable.vae.decode(1 / 0.18215 * r0).sample *0.5 + 0.5
    
    return Image.fromarray(np.concatenate((
        image_instruct,
        np.uint8(np.clip(img_gt[0].permute(1,2,0).cpu().numpy()*255, 0, 255)),
        np.uint8(np.clip(img_recon[0].permute(1,2,0).cpu().numpy()*255, 0, 255)),
        np.uint8(np.clip(img_edit[0].permute(1,2,0).cpu().numpy()*255, 0, 255)),
        ),1))


def replace_nsfw_images(results):
    for i in range(len(results.images)):
        if results.nsfw_content_detected[i]:
            results.images[i] = Image.open("nsfw.png")
    return results.images[0]

def mask_decode(encoded_mask,image_shape=[512,512]):
    length=image_shape[0]*image_shape[1]
    mask_array=np.zeros((length,))
    
    for i in range(0,len(encoded_mask),2):
        splice_len=min(encoded_mask[i+1],length-encoded_mask[i])
        for j in range(splice_len):
            mask_array[encoded_mask[i]+j]=1
            
    mask_array=mask_array.reshape(image_shape[0], image_shape[1])
    # to avoid annotation errors in boundary
    mask_array[0,:]=1
    mask_array[-1,:]=1
    mask_array[:,0]=1
    mask_array[:,-1]=1
            
    return mask_array

image_save_paths={
    "ddpm_edit":"ddpm_edit",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rerun_exist_images', action= "store_true") # rerun existing images
    parser.add_argument('--data_path', type=str, default="data") # the editing category that needed to run
    parser.add_argument('--output_path', type=str, default="output") # the editing category that needed to run
    parser.add_argument('--edit_category_list', nargs = '+', type=str, default=["0","1","2","3","4","5","6","7","8","9"]) # the editing category that needed to run
    parser.add_argument('--edit_method_list', nargs = '+', type=str, default=["ddpm_edit"]) # the editing methods that needed to run
    args = parser.parse_args()
    
    rerun_exist_images=args.rerun_exist_images
    data_path=args.data_path
    output_path=args.output_path
    edit_category_list=args.edit_category_list
    edit_method_list=args.edit_method_list
    
    with open(f"{data_path}/mapping_file.json", "r") as f:
        editing_instruction = json.load(f)

    for key, item in editing_instruction.items():
        
        if item["editing_type_id"] not in edit_category_list:
            continue
        
        original_prompt = item["original_prompt"].replace("[", "").replace("]", "")
        editing_prompt = item["editing_prompt"].replace("[", "").replace("]", "")
        image_path = os.path.join(f"{data_path}/annotation_images", item["image_path"])
        editing_instruction = item["editing_instruction"]
        blended_word = item["blended_word"].split(" ") if item["blended_word"] != "" else []
        if item["blended_word"]!="":
            local = item["blended_word"].split(" ")[1]
        else:
            local = ""
        mask = Image.fromarray(np.uint8(mask_decode(item["mask"])[:,:,np.newaxis].repeat(3,2))).convert("L")

        for edit_method in edit_method_list:
            present_image_save_path=image_path.replace(data_path, os.path.join(output_path,image_save_paths[edit_method]))
            if ((not os.path.exists(present_image_save_path)) or rerun_exist_images):
                print(f"editing image [{image_path}] with [{edit_method}]")
                setup_seed()
                torch.cuda.empty_cache()
                if edit_method=="ddpm_edit":
                    edited_image = edit_image_ddpm_edit(
                        source_prompt=original_prompt,
                        target_prompt=editing_prompt,
                        img=image_path,
                        width=512, height=512,)


                else:
                    raise NotImplementedError(f"No edit method named {edit_method}")
                
                
                if not os.path.exists(os.path.dirname(present_image_save_path)):
                    os.makedirs(os.path.dirname(present_image_save_path))
                edited_image.save(present_image_save_path)
                
                print(f"finish")
                
            else:
                print(f"skip image [{image_path}] with [{edit_method}]")
        
        
        