
import torch
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
from models.flux.flux.sampling import denoise, denoise_vanilla, get_schedule, prepare, unpack
from models.flux.flux.util import (configs, load_ae, load_clip,
                       load_flow_model, load_t5)

from utils.utils import txt_draw,load_512,latent2image
from einops import rearrange
import pdb


device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
    'cpu')

NUM_DDIM_STEPS = 50

def setup_seed(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def patch_shuffle(img_tensor, patch_size, indices=None):
    
    # Assuming the image is square and the patch size divides the image dimensions evenly
    H, W = img_tensor.shape[2], img_tensor.shape[3]  # B, C, H, W
    patch_H, patch_W = patch_size, patch_size
    
    # Calculate the number of patches
    num_patches_H, num_patches_W = H // patch_H, W // patch_W
    
    # Create an index map for shuffling
    indices = torch.randperm(num_patches_H * num_patches_W).to(torch.int64) if indices is None else indices
    
    # Extract patches and shuffle them
    patches = []
    for r in range(num_patches_H):
        for j in range(num_patches_W):
            patches.append(img_tensor[:, :, r*patch_H:(r+1)*patch_H, j*patch_W:(j+1)*patch_W])
    patches = torch.stack(patches, 0)

    shuffled_patches = patches[indices]
    reverse_indices = torch.empty_like(indices)
    reverse_indices[indices] = torch.arange(len(indices))
    # for i in range(len(indices)):
    #     reverse_indices[indices[i]] = i
    
    shuffled_patches = shuffled_patches[indices]
    new_img_tensor = torch.empty_like(img_tensor)
    for r in range(num_patches_H):
        for j in range(num_patches_W):
            new_img_tensor[:, :, r*patch_H:(r+1)*patch_H, j*patch_W:(j+1)*patch_W] = shuffled_patches[r*num_patches_W+j]

    
    return new_img_tensor, indices, reverse_indices


torch_device = torch.device("cuda")
name = "flux-dev"
offload = False
t5 = load_t5(torch_device, max_length=256 if name == "flux-schnell" else 512)
clip = load_clip(torch_device)
model = load_flow_model(name, device="cpu" if offload else torch_device)
ae = load_ae(name, device="cpu" if offload else torch_device)


@torch.inference_mode()
def encode(init_image, torch_device, ae):
    init_image = torch.from_numpy(init_image).permute(2, 0, 1).float() / 127.5 - 1
    init_image = init_image.unsqueeze(0) 
    init_image = init_image.to(torch_device)
    init_image = ae.encode(init_image.to()).to(torch.bfloat16)
    return init_image

@torch.inference_mode()
@torch.autocast("cuda", torch.bfloat16)
def edit_image(
    image_path,
    prompt_src,
    prompt_tar,
    guidance_scale=7.5,
    image_shape=[512,512],
    num_steps=25,
    freeinv=False,
    second_order=False,
):
    torch.cuda.empty_cache()
    image_gt = load_512(image_path)
    init_image = encode(image_gt, torch_device, ae)

    inp = prepare(t5, clip, init_image, prompt=prompt_src)
    # inp_target = prepare(t5, clip, init_image, prompt=prompt_tar)
    inp_target = inp
    timesteps = get_schedule(num_steps, inp["img"].shape[1], shift=(name != "flux-schnell"))
    angle_lst = [random.choice([0, 90, 180, 270]) for _ in range(num_steps)] if freeinv else [0] * num_steps

    # inversion initial noise
    z, info = denoise_vanilla(model, **inp, timesteps=timesteps, guidance=1, inverse=True, info={}, angle_lst=angle_lst, second_order=second_order)

    inp_target["img"] = z

    timesteps = get_schedule(num_steps, inp_target["img"].shape[1], shift=(name != "flux-schnell"))

    # denoise initial noise
    x, _ = denoise_vanilla(model, **inp_target, timesteps=timesteps, guidance=1, inverse=False, info=info, angle_lst=angle_lst, second_order=second_order)

    x = unpack(x.float(), 512, 512)

    with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
        x = ae.decode(x)
    x = x.clamp(-1, 1)
    x = rearrange(x[0], "c h w -> h w c")
    rgb_reconstruction = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

    edited_image = rgb_reconstruction
    
    image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")

    return Image.fromarray(np.concatenate((
        image_instruct,
        image_gt,
        np.uint8(np.array(rgb_reconstruction)),
        np.uint8(np.array(edited_image)),
        ),1))


@torch.inference_mode()
@torch.autocast("cuda", torch.bfloat16)
def edit_image_RF_solver(
    image_path,
    prompt_src,
    prompt_tar,
    guidance_scale=7.5,
    image_shape=[512,512],
    num_steps=25,
    freeinv=False,
):
    torch.cuda.empty_cache()
    image_gt = load_512(image_path)
    init_image = encode(image_gt, torch_device, ae)

    inp = prepare(t5, clip, init_image, prompt=prompt_src)
    # inp_target = prepare(t5, clip, init_image, prompt=prompt_tar)
    inp_target = inp
    timesteps = get_schedule(num_steps, inp["img"].shape[1], shift=(name != "flux-schnell"))
    angle_lst = [random.choice([0, 90, 180, 270]) for _ in range(num_steps)] if freeinv else [0] * num_steps

    info = {}
    info['feature_path'] = './tmp'
    info['feature'] = {}
    info['inject_step'] = 3

    # inversion initial noise
    z, info = denoise(model, **inp, timesteps=timesteps, guidance=1, inverse=True, info=info)

    inp_target["img"] = z

    timesteps = get_schedule(num_steps, inp_target["img"].shape[1], shift=(name != "flux-schnell"))

    # denoise initial noise
    x, _ = denoise(model, **inp_target, timesteps=timesteps, guidance=1, inverse=False, info=info)

    x = unpack(x.float(), 512, 512)

    with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
        x = ae.decode(x)
    x = x.clamp(-1, 1)
    x = rearrange(x[0], "c h w -> h w c")
    rgb_reconstruction = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

    edited_image = rgb_reconstruction
    
    image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")

    return Image.fromarray(np.concatenate((
        image_instruct,
        image_gt,
        np.uint8(np.array(rgb_reconstruction)),
        np.uint8(np.array(edited_image)),
        ),1))


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
    "flux":"flux-RF",
    "flux+freeinv":"flux-RF+freeinv",
    "flux+rfsolver": "flux-RF+rfsolver",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rerun_exist_images', action= "store_true") # rerun existing images
    parser.add_argument('--data_path', type=str, default="data") # the editing category that needed to run
    parser.add_argument('--output_path', type=str, default="output") # the editing category that needed to run
    parser.add_argument('--edit_category_list', nargs = '+', type=str, default=["0","1","2","3","4","5","6","7","8","9"]) # the editing category that needed to run
    parser.add_argument('--edit_method_list', nargs = '+', type=str, default=["ddim+pnp","directinversion+pnp", "freeinv+pnp"]) # the editing methods that needed to run
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
        mask = Image.fromarray(np.uint8(mask_decode(item["mask"])[:,:,np.newaxis].repeat(3,2))).convert("L")

        for edit_method in edit_method_list:
            present_image_save_path=image_path.replace(data_path, os.path.join(output_path,image_save_paths[edit_method]))
            if ((not os.path.exists(present_image_save_path)) or rerun_exist_images):
                print(f"editing image [{image_path}] with [{edit_method}]")
                setup_seed()
                torch.cuda.empty_cache()
                if edit_method=="flux":
                    edited_image = edit_image(
                        image_path=image_path,
                        prompt_src=original_prompt,
                        prompt_tar=editing_prompt,
                        guidance_scale=7.5,
                    )
                elif edit_method=="flux+freeinv":
                    edited_image = edit_image(
                        image_path=image_path,
                        prompt_src=original_prompt,
                        prompt_tar=editing_prompt,
                        guidance_scale=7.5,
                        freeinv=True,
                    )
                elif edit_method=="flux+rfsolver":
                    edited_image = edit_image_RF_solver(
                        image_path=image_path,
                        prompt_src=original_prompt,
                        prompt_tar=editing_prompt,
                        guidance_scale=7.5,
                        freeinv=False,
                    )
                else:
                    raise NotImplementedError(f"No edit method named {edit_method}")
                
                
                if not os.path.exists(os.path.dirname(present_image_save_path)):
                    os.makedirs(os.path.dirname(present_image_save_path))
                edited_image.save(present_image_save_path)
                
                print(f"finish")
                
            else:
                print(f"skip image [{image_path}] with [{edit_method}]")
        
        
        