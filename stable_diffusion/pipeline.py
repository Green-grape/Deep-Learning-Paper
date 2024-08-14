import torch
import numpy as np
from tqdm import tqdm  # for progress bar
from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH//8  # VAE latent width
LATENTS_HEIGHT = HEIGHT//8  # VAE latent height

# negative prompt: 원하지 않는 것을 제거하는 prompt
# strength: input image 집중도 or 얼마나 많은 noise를 추가할 것인가? (0~1)
# do_cfg: classifer free guidance 활용 여부
# cfg_scale: conditioned된 ouput에 얼마나 집중할 것인가? (1~14)
# n_inference_steps: inference step 수(50이 일반적)
# models: pretrained model
# seed: random sampling seed
# device: 사용할 device
# idle_device: idle device(cpu)
# tokenizer: tokenizer


def generate(prompt: str,  ucond_prompt: str, input_image=None,
             strength=0.8, do_cfg=True, cfg_scale=7.5, sampler_name="ddpm", n_inference_steps=50, models={}, seed=None,
             device=None,
             idle_device=None,
             tokenizer=None):
    with torch.no_grad():
        if not (0 < strength <= 1):
            raise ValueError("strength should be in (0, 1]")

        if idle_device:
            def to_idle(x): return x.to(idle_device)
        else:
            def to_idle(x): return x

        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        clip = models["clip"]
        clip.to(device)

        # UNET에 conditioned된 prompt를 넣어줄 것인가/안넣어줄 것인가
        if do_cfg:
            # convert into a list of tokens
            cond_tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77)["input_ids"]
            cond_tokens = torch.tensor(
                cond_tokens, dtype=torch.long, device=device)
            cond_context = clip(cond_tokens)

            # convert into a list of tokens
            uncond_tokens = tokenizer.batch_encode_plus(
                [ucond_prompt], padding="max_length", max_length=77)["input_ids"]
            uncond_tokens = torch.tensor(
                uncond_tokens, dtype=torch.long, device=device)
            uncond_context = clip(uncond_tokens)

            # (Batch_Size, Seq_Len, Dim)+ (Batch_Size, Dim) -> (Batch_Size*2, Seq_Len, Dim)
            context = torch.cat([cond_context, uncond_context])
        else:
            # convert into a list of tokens
            tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77)["input_ids"]
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            context = clip(tokens)
        to_idle(clip)  # move clip to idle device

        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
        else:
            raise ValueError(f"Unknown sampler {sampler_name}")

        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        # Image To Image
        if input_image:
            encoder = models["encoder"]
            encoder.to(device)

            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            input_image_tensor = np.array(input_image_tensor)
            input_image_tensor = torch.tensor(
                input_image_tensor, dtype=torch.float32, device=device)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            # (Height, Width, Channel) -> (Batch_Size, Channel, Height, Width)
            input_image_tensor = input_image_tensor.permute(
                2, 0, 1).unsqueeze(0)

            encoder_noise = torch.randn(
                latents_shape, generator=generator, device=device)
            latents = encoder(input_image_tensor, encoder_noise)

            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            to_idle(encoder)  # move encoder to idle device
        # Text To Image
        else:
            # Just random noise
            latents = torch.randn(
                latents_shape, generator=generator, device=device)
        diffusion = models["diffusion"]
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            time_embedding = get_time_embedding(timestep).to(device)

            model_input = latents
            if do_cfg:
                # (Batch_Size, 4, Lantents_Height, Latents_Width) -> (Batch_Size*2,4, Lantents_Height, Latents_Width)
                model_input = model_input.repeat(2, 1, 1, 1)

            # predicted noise
            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * \
                    (output_cond-output_uncond)+output_uncond
            # remove noise from latents
            latents = sampler.step(
                timestep=timestep, latents=latents, model_output=model_output)

        to_idle(diffusion)  # move diffusion to idle device

        decoder = models["decoder"]
        decoder.to(device)

        images = decoder(latents)
        to_idle(decoder)  # move decoder to idle device

        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        images = images.permute(0, 2, 3, 1).to("cpu").numpy().astype(np.uint8)
        return images[0]


def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    y = (x-old_min)/(old_max-old_min) * (new_max-new_min) + new_min
    if clamp:
        y = y.clamp(new_min, new_max)
    return y


'''
# transformer에서 사용하는 time embedding
# return (1,320) time embedding
'''


def get_time_embedding(timesteps):
    # shape (160,)
    freqs = torch.pow(10000, -torch.arange(start=0,
                      end=160, dtype=torch.float32)/160)
    # shape (1, 160)
    x = torch.tensor([timesteps], dtype=torch.float32)[:, None]*freqs[None, :]
    # shape (1, 160)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
