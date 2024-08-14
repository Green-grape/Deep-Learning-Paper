import torch
import numpy as np


class DDPMSampler:
    # beta: 논문에서는 0.00085~0.0120 사이의 값을 사용
    def __init__(self, generator: torch.Generator, num_training_steps=1000, beta_start: float = 0.00085, beta_end: float = 0.0120):
        self.betas = torch.linspace(
            beta_start**0.5, beta_end**0.5, num_training_steps, dtype=torch.float32)**2
        self.alphas = 1.0-self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, 0)
        self.one = torch.tensor(1.0)

        self.generator = generator
        self.num_training_steps = num_training_steps
        # [::-1]: reverse
        self.timesteps = torch.from_numpy(
            np.arange(0, num_training_steps)[::-1].copy())

    def set_inference_timesteps(self, num_inference_steps=50):
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_training_steps//self.num_inference_steps
        self.timesteps = torch.from_numpy(
            np.arange(0, self.num_training_steps, step_ratio)[::-1].copy())
    '''
    noise를 이미지에 얼마나 넣을 것인가? 
    강도를 수정하고 해당 강도에 따라 timestep을 수정한다.
    '''

    def set_strength(self, strength=1):
        start_step = self.num_inference_steps - \
            int(self.num_inference_steps*strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step

    def _get_previous_timestep(self, timestep: int) -> int:
        prev_t = timestep-self.num_training_steps//self.num_inference_steps
        return prev_t

    def add_noise(self, original_samples: torch.FloatTensor, timestep: torch.IntTensor) -> torch.FloatTensor:
        alphas_cumpord = self.alpha_cumprod.to(
            device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timestep.to(device=original_samples.device)

        # calculate mean of the gaussian distribution
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumpord[timesteps])
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.flatten()

        # 차원을 맞추기 위해 unsqueeze
        while len(sqrt_alphas_cumprod.shape) < len(original_samples.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod.unsqueeze(-1)

        # calcute std of the gaussian distribution
        sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1-alphas_cumpord[timesteps])
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.flatten()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(original_samples.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.unsqueeze(
                -1)

        noise = torch.randn(original_samples.shape, generator=self.generator,
                            device=original_samples.device, dtype=original_samples.dtype)

        return sqrt_alphas_cumprod*original_samples+sqrt_one_minus_alphas_cumprod*noise

    def step(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor):
        t = timestep
        prev_t = self._get_previous_timestep(t)  # t-1

        # alphas, betas
        alpha_prod_t = self.alpha_cumprod[t]
        alpha_prod_prev_t = self.alpha_cumprod[prev_t] if prev_t >= 0 else self.one
        alpha_t = alpha_prod_t/alpha_prod_prev_t
        beta_t = 1-alpha_t

        # predicted x0
        pred_x0 = (latents-torch.sqrt(1-alpha_prod_t) *
                   model_output)/torch.sqrt(alpha_prod_t)

        # mean and std of q(x_{t-1}|x_t, x_0)
        mean = (torch.sqrt(alpha_prod_prev_t)*beta_t*pred_x0 +
                torch.sqrt(alpha_t)*(1-alpha_prod_prev_t)*latents)/(1-alpha_prod_t)
        std = 0
        # add noise to std
        if t > 0:
            device = model_output.device
            noise = torch.randn(model_output.shape, generator=self.generator,
                                device=device, dtype=model_output.dtype)
            std = torch.sqrt(torch.clamp((1-alpha_prod_prev_t) *
                                         beta_t/(1-alpha_prod_t), min=1e-20))
            std = std*noise
        return mean+std
