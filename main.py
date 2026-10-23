import os 
import json
from PIL import Image 
import numpy as np 
from tqdm import tqdm
import math 
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from guidance.sd import StableDiffusion
from utils import *


def seed_everything(seed=2024):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles: float = 0.5):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, -1)


def init_model(args):
    model = StableDiffusion(args, t_range=[0.02, 0.98])
    return model
    
    
def run(args):
    args.precision = torch.float16 if args.precision == "fp16" else torch.float32
    
    # Initialize model and optimizing parameters
    model = init_model(args)
    device = model.device
    guidance_scale = args.guidance_scale
    steps = args.steps

    # Get text embeddings
    cond_embeddings = model.get_text_embeds(args.prompt)
    uncond_embeddings = model.get_text_embeds(args.negative_prompt)
    text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])
  
    latents = nn.Parameter(
        torch.randn(1, 4, 64, 64, device=device, dtype=args.precision)
    )
    
    # Setup optimizer
    if args.loss_type == "vsd":
        # VSD needs to optimize both latents and LoRA parameters
        optimizer = torch.optim.AdamW(
            [
                {'params': [latents], 'lr': args.lr},
                {'params': model.lora_layers, 'lr': args.lora_lr}
            ],
            weight_decay=0
        )
    else:
        optimizer = torch.optim.AdamW([latents], lr=args.lr, weight_decay=0)

    scheduler = get_cosine_schedule_with_warmup(optimizer, 100, int(steps*1.5))

    # Run optimization
    for step in tqdm(range(steps)):
        optimizer.zero_grad()
        
        if args.loss_type == "sds":
            loss = model.get_sds_loss(
                latents=latents,
                text_embeddings=text_embeddings, 
                guidance_scale=guidance_scale,
            )
        elif args.loss_type == "diff":
            loss = model.get_diffusion_loss(
                latents=latents,
                text_embeddings=text_embeddings,
            )
            
        elif args.loss_type == "sdi":
            loss = model.get_sdi_loss(
                latents=latents,
                text_embeddings=text_embeddings,
                guidance_scale=guidance_scale,
                current_iter=step,
                total_iters=steps,
                inversion_guidance_scale=args.inversion_guidance_scale,
                inversion_n_steps=args.inversion_n_steps,
                inversion_eta=args.inversion_eta,
                update_interval=args.sdi_update_interval,
            )
            
        elif args.loss_type == "vsd":
            loss = model.get_vsd_loss(
                latents=latents,
                text_embeddings=text_embeddings,
                guidance_scale=guidance_scale,
                lora_loss_weight=args.lora_loss_weight,
            )
            
        else:
            raise ValueError("Invalid loss type")
        
        # Backward
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_([latents], max_norm=1.0)

        # Update optimizers
        optimizer.step()
        scheduler.step()
        
        # Logging
        if step % args.log_step == 0:
            with torch.no_grad():
                img = model.decode_latents(latents)
                img_save_path = os.path.join(args.save_dir, f"{step}.png")
                torch_to_pil(img).save(img_save_path)
                print(f"Step: {step}, Loss: {loss.item():.6f}")
            
    # Final save
    img = model.decode_latents(latents)
    prompt_key = args.prompt.replace(" ", "_")
    img_save_path = os.path.join(args.save_dir, f"{prompt_key}.png")
    torch_to_pil(img).save(img_save_path)
    print(f"Save path: {img_save_path}")
        
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative_prompt", type=str, default="low quality")
    parser.add_argument("--save_dir", type=str, default="./outputs")
    
    parser.add_argument("--loss_type", type=str, default="sds", choices=["sds", "sdi", "vsd", "diff"])
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--steps", type=int, default=500)

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for latents (default: 0.01)")
    
    # SDI specific parameters
    parser.add_argument("--inversion_n_steps", type=int, default=10,
                        help="Number of DDIM inversion steps for SDI (default: 10)")
    parser.add_argument("--inversion_guidance_scale", type=float, default=-7.5,
                        help="Guidance scale for inversion in SDI (negative, default: -7.5)")
    parser.add_argument("--inversion_eta", type=float, default=0.3,
                        help="Noise perturbation strength in SDI (default: 0.3)")
    parser.add_argument("--sdi_update_interval", type=int, default=25,
                        help="Update target every N steps in SDI (default: 25)")
    
    # VSD specific parameters
    parser.add_argument("--lora_lr", type=float, default=1e-4, 
                        help="Learning rate for LoRA in VSD")
    parser.add_argument("--lora_loss_weight", type=float, default=1.0, 
                        help="Weight for LoRA denoising loss in VSD")
    parser.add_argument("--lora_rank", type=int, default=4, 
                    help="LoRA rank for VSD (default: 4)")
    
    parser.add_argument("--log_step", type=int, default=25)
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16"])
    
    return parser.parse_args()

    
def main():
    args = parse_args()
    
    if os.path.exists(args.save_dir):
        print("[*] Save directory already exists. Overwriting...")
    else:
        os.makedirs(args.save_dir)
        
    log_opt = vars(args)
    config_path = os.path.join(args.save_dir, "run_config.json")
    with open(config_path, "w") as f:
        json.dump(log_opt, f, indent=4)
    
    print(f"[*] Running {args.loss_type}")
    print(f"[*] Prompt: {args.prompt}")
    print(f"[*] Guidance scale: {args.guidance_scale}")
    print(f"[*] Learning rate: {args.lr}")
    print(f"[*] Steps: {args.steps}")

    if args.loss_type == "sdi":
        print(f"[*] SDI Parameters:")
        print(f"    - Update interval: every {args.sdi_update_interval} steps")
        print(f"    - Inversion n_steps: {args.inversion_n_steps}")
        print(f"    - Inversion guidance scale: {args.inversion_guidance_scale}")
        print(f"    - Inversion eta: {args.inversion_eta}")

    run(args)
    

if __name__ == "__main__":
    main()
