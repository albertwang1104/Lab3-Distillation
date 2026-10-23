from diffusers import DDIMScheduler, StableDiffusionPipeline
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from peft import LoraConfig


class StableDiffusion(nn.Module):
    def __init__(self, args, t_range=[0.02, 0.98]):
        super().__init__()

        self.device = args.device
        self.dtype = args.precision
        print(f'[INFO] Loading Stable Diffusion...')

        model_key = "sd2-community/stable-diffusion-2-1-base"
        pipe = StableDiffusionPipeline.from_pretrained(
            model_key, torch_dtype=self.dtype,
        )

        pipe.to(self.device)
        self.vae = pipe.vae.eval()
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet.eval()
        
        # Freeze models
        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)
        
        # Schedulers
        self.scheduler = DDIMScheduler.from_pretrained(
            model_key, subfolder="scheduler", torch_dtype=self.dtype,
        )
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)

        del pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod
        
        print(f'[INFO] Loaded Stable Diffusion!')
        
        # Initialize VSD components if needed
        if args.loss_type == "vsd":
            self._init_vsd_components(args.lora_rank)
    
    def _init_vsd_components(self, lora_rank=4):
        """Initialize LoRA for VSD"""
        print(f"[INFO] Initializing VSD with LoRA rank={lora_rank}")
        
        self.unet.requires_grad_(False)
        
        unet_lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        
        self.unet.add_adapter(unet_lora_config)
        self.lora_layers = list(filter(lambda p: p.requires_grad, self.unet.parameters()))

    @torch.no_grad()
    def get_text_embeds(self, prompt):
        """Get text embeddings from prompt"""
        inputs = self.tokenizer(
            prompt, 
            padding='max_length', 
            max_length=self.tokenizer.model_max_length, 
            return_tensors='pt'
        )
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]
        return embeddings
    
    def get_noise_preds(self, latents_noisy, t, text_embeddings, guidance_scale=7.5):
        """
        Predict noise with Classifier-Free Guidance (CFG)
        
        CFG formula: noise_pred = uncond + guidance_scale * (cond - uncond)
        """
        latent_model_input = torch.cat([latents_noisy] * 2)
        tt = torch.cat([t] * 2)
        
        noise_pred = self.unet(latent_model_input, tt, encoder_hidden_states=text_embeddings).sample
        noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
        
        # Classifier-Free Guidance
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)
        
        return noise_pred
    
    def get_sds_loss(self, latents, text_embeddings, guidance_scale=7.5):
        """
        Score Distillation Sampling (SDS) loss.

        Reference: DreamFusion (https://arxiv.org/abs/2209.14988)

        Adds random noise to the latents at a random timestep and lets the
        diffusion model predict it. The difference between predicted and added
        noise is used as the gradient on the latents.

        Args:
            latents: Current optimized latents (B, 4, H, W)
            text_embeddings: Concatenated [uncond, cond] embeddings (2*B, seq_len, dim)
            guidance_scale: CFG scale

        Returns:
            loss: Scalar SDS loss
        """
        # TODO: Implement SDS loss
        # raise NotImplementedError("TODO: Implement SDS loss")
        batch_size = latents.shape[0]

        # 1. 在設定的區間 [min_step, max_step) 內隨機採樣時間步 t
        t = torch.randint(
            self.min_step,
            self.max_step,
            (batch_size,),
            dtype=torch.long,
            device=self.device
        )

        # 2. 隨機採樣高斯噪聲，並對 latents 加噪生成 z_t
        noise = torch.randn_like(latents)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)

        # 3. 使用預訓練 UNet 預測噪聲 (無梯度計算)
        with torch.no_grad():
            noise_pred = self.get_noise_preds(
                latents_noisy,
                t,
                text_embeddings,
                guidance_scale=guidance_scale
            )

        # 4. 計算權重項 w(t) 與 SDS 梯度方向
        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        grad = w * (noise_pred - noise)

        # 5. 構建虛擬 Loss，使得反向傳播時剛好產生 SDS 梯度
        target = (latents - grad).detach()
        loss = 0.5 * torch.nn.functional.mse_loss(latents, target, reduction='sum') / batch_size
        return loss

    def get_vsd_loss(self, latents, text_embeddings, guidance_scale=7.5, lora_loss_weight=1.0):
        """
        Variational Score Distillation (VSD) loss.

        Reference: ProlificDreamer (https://arxiv.org/abs/2305.16213)

        Same as SDS, but the added noise is replaced by the prediction of a
        LoRA-finetuned UNet that models the distribution of the current latents.
        The LoRA is trained jointly with a standard denoising loss.

        Args:
            lora_loss_weight: Weight of the LoRA training loss

        Returns:
            loss: Scalar VSD loss + lora_loss_weight * LoRA loss
        """
        # TODO: Implement VSD loss
        # raise NotImplementedError("TODO: Implement VSD loss")
        batch_size = latents.shape[0]

        # 1. 在 [min_step, max_step) 區間內隨機採樣時間步 t
        t = torch.randint(
            self.min_step,
            self.max_step,
            (batch_size,),
            dtype=torch.long,
            device=self.device
        )

        # 2. 隨機採樣高斯噪聲並加噪生成 z_t
        noise = torch.randn_like(latents)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)

        # 3. 凍結 LoRA 權重，使用預訓練 UNet 預測目標資料分布的噪聲 (CFG)
        with torch.no_grad():
            self.unet.disable_adapters()
            noise_pred_pretrained = self.get_noise_preds(
                latents_noisy,
                t,
                text_embeddings,
                guidance_scale=guidance_scale
            )

        # 4. 啟用 LoRA 權重，預測當前優化表示分布的噪聲
        self.unet.enable_adapters()
        
        # 訓練 LoRA 的 Denoising Loss (將 latents_noisy 切斷梯度，只用來訓練 LoRA)
        noisy_latents_lora = latents_noisy.detach()
        latent_model_input = torch.cat([noisy_latents_lora] * 2)
        tt = torch.cat([t] * 2)
        noise_pred_lora_raw = self.unet(
            latent_model_input,
            tt,
            encoder_hidden_states=text_embeddings
        ).sample

        # 計算 LoRA 自身的去噪均方誤差 (以無條件或有條件分支對齊實際噪聲)
        noise_pred_lora_uncond, noise_pred_lora_pos = noise_pred_lora_raw.chunk(2)
        lora_loss = torch.nn.functional.mse_loss(noise_pred_lora_raw, torch.cat([noise] * 2))

        # LoRA 的 CFG 噪聲預測 (計算 VSD 梯度方向時不需要對 LoRA 求導)
        noise_pred_lora = (noise_pred_lora_uncond + guidance_scale * (noise_pred_lora_pos - noise_pred_lora_uncond)).detach()

        # 5. 計算 VSD 梯度 (noise_pred_pretrained - noise_pred_lora)
        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        grad = w * (noise_pred_pretrained - noise_pred_lora)

        # 6. 建構虛擬 Loss 以將 VSD 梯度傳遞回 latents
        target = (latents - grad).detach()
        vsd_loss = 0.5 * torch.nn.functional.mse_loss(latents, target, reduction='sum') / batch_size

        # 7. 合併 VSD Loss 與 LoRA 訓練 Loss
        total_loss = vsd_loss + lora_loss_weight * lora_loss

        return total_loss

    @torch.no_grad()
    def invert_noise(self, latents, target_t, text_embeddings, guidance_scale=-7.5, n_steps=10, eta=0.3):
        """
        DDIM inversion: x0 -> x_t.

        Reference: Score Distillation via Reparametrized DDIM (https://arxiv.org/abs/2405.15891)

        Runs the DDIM update in reverse, from t=0 up to `target_t`, to find noisy
        latents consistent with x0. Write the loop yourself: you must NOT call any
        "invert" / "ddim_invert" / "invert_step" utilities from diffusers or other libraries.

        Args:
            target_t: Target timestep to invert to
            guidance_scale: CFG scale (typically negative for inversion)
            n_steps: Number of inversion steps
            eta: Stochasticity level (0 = deterministic)

        Returns:
            latents_noisy: Inverted noisy latents x_t (B, 4, H, W)
        """
        # TODO: Implement DDIM inversion
        # Hint: You may assume eta=0 (deterministic inversion) and ignore `eta`;
        # no points will be deducted for this.
        # raise NotImplementedError("TODO: Implement DDIM inversion")
        
        # 1. 建立從 0 到 target_t 的離散時間步序列 (包含端點)
        timesteps = torch.linspace(0, target_t, n_steps + 1, device=self.device, dtype=self.dtype)
        
        # 複製起始的 x_0 latents
        xt = latents.clone()
        batch_size = latents.shape[0]

        # 2. 依序從 t 前進到 t_next (t -> target_t)
        for i in range(n_steps):
            t_curr = int(timesteps[i])
            t_next = int(timesteps[i + 1])

            # 準備時間步輸入 (batch 維度)
            t_input = torch.full((batch_size,), t_curr, device=self.device, dtype=self.dtype)

            # 3. 預測當前時間步的噪聲 (使用傳入的 guidance_scale，通常為負值)
            noise_pred = self.get_noise_preds(
                xt,
                t_input,
                text_embeddings,
                guidance_scale=guidance_scale
            )

            # 4. 取得相應的 alpha_cumprod (bar_alpha)
            # t=0 時若 alpha_cumprod 未對齊，可視為 1.0；其餘索引 self.alphas[t]
            alpha_curr = self.alphas[t_curr] if t_curr > 0 else torch.tensor(1.0, device=self.device)
            alpha_next = self.alphas[t_next]

            # 5. 根據當前 xt 與噪聲估計 x_0
            x0_pred = (xt - torch.sqrt(1 - alpha_curr) * noise_pred) / torch.sqrt(alpha_curr)

            # 6. 利用估計的 x0 與噪聲，外推計算下一步 x_{next} (假設 eta=0)
            xt = torch.sqrt(alpha_next) * x0_pred + torch.sqrt(1 - alpha_next) * noise_pred

        return xt

    def get_sdi_loss(
        self,
        latents,
        text_embeddings,
        guidance_scale=7.5,
        current_iter=0,
        total_iters=500,
        inversion_guidance_scale=-7.5,
        inversion_n_steps=10,
        inversion_eta=0.3,
        update_interval=25,
    ):
        """
        Score Distillation via Inversion (SDI) loss.

        Reference: Score Distillation via Reparametrized DDIM (https://arxiv.org/abs/2405.15891)

        Instead of random noise as in SDS, SDI gets the noise by DDIM inversion of
        the current latents. The timestep is annealed from max_step to min_step, and
        the denoised target is cached and refreshed every `update_interval` iterations.

        Args:
            current_iter: Current training iteration
            total_iters: Total number of training iterations
            inversion_guidance_scale, inversion_n_steps, inversion_eta:
                Passed to `invert_noise` as guidance_scale, n_steps, eta
            update_interval: Refresh the cached target every N iterations

        Returns:
            loss: Scalar MSE loss between current latents and the cached target
        """
        B = latents.shape[0]

        # TODO: Compute the annealed timestep for the current iteration
        # 1. 計算當前 iteration 的退火時間步 (隨著訓練進度從 max_step 遞減至 min_step)
        progress = current_iter / total_iters
        t = int(self.max_step - progress * (self.max_step - self.min_step))
        t = max(self.min_step, min(self.max_step, t))
        t_tensor = torch.full((B,), t, device=self.device, dtype=torch.long)

        # Check if we need to update target
        should_update = (current_iter % update_interval == 0) or not hasattr(self, 'sdi_target')

        if should_update:
            with torch.no_grad():
                # Perform DDIM inversion: x0 -> x_t
                latents_noisy = self.invert_noise(
                    latents, t, text_embeddings,
                    guidance_scale=inversion_guidance_scale,
                    n_steps=inversion_n_steps,
                    eta=inversion_eta
                )

                # TODO: Predict noise from the inverted noisy latents
                # 2. 使用凍結的 UNet 與 CFG 預測加噪潛在變量的噪聲
                noise_pred = self.get_noise_preds(
                    latents_noisy,
                    t_tensor,
                    text_embeddings,
                    guidance_scale=guidance_scale
                )

                # TODO: Denoise to get the target x0 from the predicted noise
                # 3. 透過 DDIM 公式將預測噪聲反推回去，得到乾淨的目標圖像 x0
                alpha_curr = self.alphas[t] if t > 0 else torch.tensor(1.0, device=self.device)
                target = (latents_noisy - torch.sqrt(1 - alpha_curr) * noise_pred) / torch.sqrt(alpha_curr)

                # Cache the target
                self.sdi_target = target.detach()

        # TODO: Compute the MSE loss between current latents and the cached target
        # 4. 計算目前 latents 與快取目標之間的 MSE 損失
        loss = 0.5 * torch.nn.functional.mse_loss(latents, self.sdi_target, reduction='sum') / B

        return loss
        
    @torch.no_grad()
    def decode_latents(self, latents):
        """Decode latents to RGB images"""
        latents = 1 / self.vae.config.scaling_factor * latents
        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs
