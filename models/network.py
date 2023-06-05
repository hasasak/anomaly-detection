import math
import torch
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
from core.base_network import BaseNetwork
from torchvision import utils as vutils
import core.util as Util
from PIL import Image
class Network(BaseNetwork):
    def __init__(self, unet, beta_schedule, module_name='sr3', **kwargs):
        super(Network, self).__init__(**kwargs)
        if module_name == 'sr3':
            from .sr3_modules.unet import UNet
        elif module_name == 'guided_diffusion':
            from .guided_diffusion_modules.unet import UNet
        
        self.denoise_fn = UNet(**unet)
        self.beta_schedule = beta_schedule

    def set_loss(self, loss_fn):
        self.loss_fn = loss_fn

    def set_new_noise_schedule(self, device=torch.device('cuda'), phase='train'):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        betas = make_beta_schedule(**self.beta_schedule[phase])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        
        gammas = np.cumprod(alphas, axis=0)
        gammas_prev = np.append(1., gammas[:-1])

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('gammas', to_torch(gammas))
        self.register_buffer('sqrt_recip_gammas', to_torch(np.sqrt(1. / gammas)))
        self.register_buffer('sqrt_recipm1_gammas', to_torch(np.sqrt(1. / gammas - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - gammas_prev) / (1. - gammas)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(gammas_prev) / (1. - gammas)))
        self.register_buffer('posterior_mean_coef2', to_torch((1. - gammas_prev) * np.sqrt(alphas) / (1. - gammas)))

    def predict_start_from_noise(self, y_t, t, noise):
        return (
            extract(self.sqrt_recip_gammas, t, y_t.shape) * y_t -
            extract(self.sqrt_recipm1_gammas, t, y_t.shape) * noise
        )

    def q_posterior(self, y_0_hat, y_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, y_t.shape) * y_0_hat +
            extract(self.posterior_mean_coef2, t, y_t.shape) * y_t
        )
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, y_t.shape)
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, y_t, t, clip_denoised: bool, y_cond=None):
        noise_level = extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device)
        y_0_hat = self.predict_start_from_noise(
                y_t, t=t, noise=self.denoise_fn(torch.cat([y_cond, y_t], dim=1), noise_level))

        if clip_denoised:
            y_0_hat.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            y_0_hat=y_0_hat, y_t=y_t, t=t)
        return model_mean, posterior_log_variance

    def q_sample(self, y_0, sample_gammas, noise=None):
        noise = default(noise, lambda: torch.randn_like(y_0))
        return (
            sample_gammas.sqrt() * y_0 +
            (1 - sample_gammas).sqrt() * noise
        )
    @torch.no_grad()
    def q_sample_test(self, y_0, sample_gammas, noise=None):
        noise = default(noise, lambda: torch.randn_like(y_0))
        return (
            sample_gammas.sqrt() * y_0 +
            (1 - sample_gammas).sqrt() * noise
        )
    @torch.no_grad()
    def p_sample(self, y_t, t, clip_denoised=True, y_cond=None):
        model_mean, model_log_variance = self.p_mean_variance(
            y_t=y_t, t=t, clip_denoised=clip_denoised, y_cond=y_cond)
        noise = torch.randn_like(y_t) if any(t>0) else torch.zeros_like(y_t)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def restoration(self, y_cond, y_t=None, y_0=None, mask=None, sample_num=8,restoration_type = 'stripe'):
        if restoration_type == 'global':
            b, *_ = y_cond.shape
    
            assert self.num_timesteps > sample_num, 'num_timesteps must greater than sample_num'
            sample_inter = (self.num_timesteps//sample_num)
            y_t = default(y_t, lambda: torch.randn_like(y_cond))
            #t_test = torch.randint(200, 201, (b,), device=y_0.device).long()
            t_test = torch.ones(size=(b,),device=y_0.device).long()
            for i in range(t_test.shape[0]):
                t_test[i] = 200
            t_test.long()
            gamma_t1 = extract(self.gammas, t_test-1, x_shape=(1, 1))
            sqrt_gamma_t2 = extract(self.gammas, t_test, x_shape=(1, 1))
            sample_gammas = (sqrt_gamma_t2-gamma_t1) * torch.rand((b, 1), device=y_0.device) + gamma_t1
            sample_gammas = sample_gammas.view(b, -1)
            
            noise_test = torch.randn_like(y_0)
            y_n = self.q_sample_test(
                y_0=y_0, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise_test)
            ret_arr = y_t
            y_T = y_n
            y_T_1 = y_T_2 = y_T_3 =y_T
            for i in range(y_T.shape[0]):
                y_re = y_T[i,:,:,:]
                y_re_fortensor=Util.tensor2img(y_re)
                #Image.fromarray(y_re_fortensor).save('y_IN'+str(i)+'.jpg')
                
            for i in range(y_0.shape[0]):
                y_0_re = y_0[i,:,:,:]
                y_re_fortensor=Util.tensor2img(y_0_re)
               # Image.fromarray(y_re_fortensor).save('y_0'+str(i)+'.jpg')
            for i in tqdm(reversed(range(0, int(200))), desc='sampling loop time step', total=self.num_timesteps):
                t = torch.full((b,), i, device=y_cond.device, dtype=torch.long)
                y_T = self.p_sample(y_T, t, y_cond=y_cond)
                #y_T_1 = self.p_sample(y_T_1, t, y_cond=y_cond)
                #y_T_2 = self.p_sample(y_T_2, t, y_cond=y_cond)
                #y_T_3 = self.p_sample(y_T_3, t, y_cond=y_cond)
                if i !=0:
                    t_test = torch.ones(size=(b,),device=y_0.device).long()
                    for k in range(t_test.shape[0]):
                        t_test[k] = i
                    t_test.long()
                    gamma_t1 = extract(self.gammas, t_test-1, x_shape=(1, 1))
                    sqrt_gamma_t2 = extract(self.gammas, t_test, x_shape=(1, 1))
                    sample_gammas = (sqrt_gamma_t2-gamma_t1) * torch.rand((b, 1), device=y_0.device) + gamma_t1
                    sample_gammas = sample_gammas.view(b, -1)
                
                    y_n = self.q_sample_test(
                        y_0=y_0, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise_test)
                    if mask is not None:
                        #print(mask)
                        mask = mask.to(y_0.device)
                        #mask_1 = mask[0].to(y_0.device)
                        #mask_2 = mask[1].to(y_0.device)
                        #mask_3 = mask[2].to(y_0.device)
                        #mask_4 = mask[3].to(y_0.device)
                        #y_T = y_n*(1.-mask_1)+mask_1*y_T
                        #y_T_1 = y_n*(1.-mask_2)+mask_2*y_T_1
                        #y_T_2 = y_n*(1.-mask_3)+mask_3*y_T_2
                        #y_T_3 = y_n*(1.-mask_4)+mask_4*y_T_3
                        y_T = y_n * (1. - mask) + mask * y_T
                if i == 0:
                    #y_T = y_0*(1.-mask_1)+mask_1*y_T
                    y_T = y_0 * (1. - mask) + mask * y_T
                    #y_T_1 = y_0*(1.-mask_2)+mask_2*y_T_1
                    #y_T_2 = y_0*(1.-mask_3)+mask_3*y_T_2
                    #y_T_3 = y_0*(1.-mask_4)+mask_4*y_T_3
                if i % sample_inter == 0:
                    ret_arr = torch.cat([ret_arr, y_t], dim=0)
           
            #return y_T,y_T_1,y_T_2,y_T_3,mask_1,mask_2,mask_3,mask_4,y_n,y_0, ret_arr
            return y_T, y_n, y_0, ret_arr
        if restoration_type == 'random':
            b, *_ = y_cond.shape
    
            assert self.num_timesteps > sample_num, 'num_timesteps must greater than sample_num'
            sample_inter = (self.num_timesteps//sample_num)
            y_t = default(y_t, lambda: torch.randn_like(y_cond))
            #t_test = torch.randint(200, 201, (b,), device=y_0.device).long()
            t_test = torch.ones(size=(b,),device=y_0.device).long()
            for i in range(t_test.shape[0]):
                t_test[i] = 999
            t_test.long()
            gamma_t1 = extract(self.gammas, t_test-1, x_shape=(1, 1))
            sqrt_gamma_t2 = extract(self.gammas, t_test, x_shape=(1, 1))
            sample_gammas = (sqrt_gamma_t2-gamma_t1) * torch.rand((b, 1), device=y_0.device) + gamma_t1
            sample_gammas = sample_gammas.view(b, -1)
            
            noise_test = torch.randn_like(y_0)
            y_n = self.q_sample_test(
                y_0=y_0, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise_test)
            ret_arr = y_t
            y_T = y_n
            y_T_1 =self.q_sample_test(
                y_0=y_0, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise_test)
            y_T_2 = self.q_sample_test(
                y_0=y_0, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise_test)
            y_T_3 =self.q_sample_test(
                y_0=y_0, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise_test)
            for i in tqdm(reversed(range(0, int(999))), desc='sampling loop time step', total=self.num_timesteps):
                t = torch.full((b,), i, device=y_cond.device, dtype=torch.long)
                y_T = self.p_sample(y_T, t, y_cond=y_cond)
                y_T_1 = self.p_sample(y_T_1, t, y_cond=y_cond)
                y_T_2 = self.p_sample(y_T_2, t, y_cond=y_cond)
                y_T_3 = self.p_sample(y_T_3, t, y_cond=y_cond)
                if i !=0:
                    t_test = torch.ones(size=(b,),device=y_0.device).long()
                    for k in range(t_test.shape[0]):
                        t_test[k] = i
                    t_test.long()
                    gamma_t1 = extract(self.gammas, t_test-1, x_shape=(1, 1))
                    sqrt_gamma_t2 = extract(self.gammas, t_test, x_shape=(1, 1))
                    sample_gammas = (sqrt_gamma_t2-gamma_t1) * torch.rand((b, 1), device=y_0.device) + gamma_t1
                    sample_gammas = sample_gammas.view(b, -1)
                
                    y_n = self.q_sample_test(
                        y_0=y_0, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise_test)
                    if mask is not None:
                        #print(mask)
                        mask_1 = mask[0].to(y_0.device)
                        mask_2 = mask[1].to(y_0.device)
                        mask_3 = mask[2].to(y_0.device)
                        mask_4 = mask[3].to(y_0.device)
                        y_T = y_n*(1.-mask_1)+mask_1*y_T
                        y_T_1 = y_n*(1.-mask_2)+mask_2*y_T_1
                        y_T_2 = y_n*(1.-mask_3)+mask_3*y_T_2
                        y_T_3 = y_n*(1.-mask_4)+mask_4*y_T_3
                if i == 0:
                    y_T = y_0*(1.-mask_1)+mask_1*y_T
                    y_T_1 = y_0*(1.-mask_2)+mask_2*y_T_1
                    y_T_2 = y_0*(1.-mask_3)+mask_3*y_T_2
                    y_T_3 = y_0*(1.-mask_4)+mask_4*y_T_3
                if i % sample_inter == 0:
                    ret_arr = torch.cat([ret_arr, y_t], dim=0)
           
            return y_T,y_T_1,y_T_2,y_T_3,mask_1,mask_2,mask_3,mask_4,y_n,y_0, ret_arr
        if restoration_type == 'stripe':
            b, *_ = y_cond.shape
    
            assert self.num_timesteps > sample_num, 'num_timesteps must greater than sample_num'
            sample_inter = (self.num_timesteps//sample_num)
            y_t = default(y_t, lambda: torch.randn_like(y_cond))
            #t_test = torch.randint(200, 201, (b,), device=y_0.device).long()
            t_test = torch.ones(size=(b,),device=y_0.device).long()
            for i in range(t_test.shape[0]):
                t_test[i] = 999
            t_test.long()
            gamma_t1 = extract(self.gammas, t_test-1, x_shape=(1, 1))
            sqrt_gamma_t2 = extract(self.gammas, t_test, x_shape=(1, 1))
            sample_gammas = (sqrt_gamma_t2-gamma_t1) * torch.rand((b, 1), device=y_0.device) + gamma_t1
            sample_gammas = sample_gammas.view(b, -1)
            
            noise_test = torch.randn_like(y_0)
            y_n = self.q_sample_test(
                y_0=y_0, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise_test)
            ret_arr = y_t
            y_T = y_n
            y_T_1 =self.q_sample_test(
                y_0=y_0, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise_test)
            #y_T_2 = self.q_sample_test(
                #y_0=y_0, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise_test)
            #y_T_3 =self.q_sample_test(
                #y_0=y_0, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise_test)
            for i in tqdm(reversed(range(0, int(999))), desc='sampling loop time step', total=self.num_timesteps):
                t = torch.full((b,), i, device=y_cond.device, dtype=torch.long)
                y_T = self.p_sample(y_T, t, y_cond=y_cond)
                y_T_1 = self.p_sample(y_T_1, t, y_cond=y_cond)
                #y_T_2 = self.p_sample(y_T_2, t, y_cond=y_cond)
                #y_T_3 = self.p_sample(y_T_3, t, y_cond=y_cond)
                if i !=0:
                    t_test = torch.ones(size=(b,),device=y_0.device).long()
                    for k in range(t_test.shape[0]):
                        t_test[k] = i
                    t_test.long()
                    gamma_t1 = extract(self.gammas, t_test-1, x_shape=(1, 1))
                    sqrt_gamma_t2 = extract(self.gammas, t_test, x_shape=(1, 1))
                    sample_gammas = (sqrt_gamma_t2-gamma_t1) * torch.rand((b, 1), device=y_0.device) + gamma_t1
                    sample_gammas = sample_gammas.view(b, -1)
                
                    y_n = self.q_sample_test(
                        y_0=y_0, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise_test)
                    if mask is not None:
                        mask_1 = mask[0].to(y_0.device)
                        mask_2 = mask[1].to(y_0.device)
                        #mask_3 = mask[2].to(y_0.device)
                        #mask_4 = mask[3].to(y_0.device)
                        y_T = y_n*(1.-mask_1)+mask_1*y_T
                        y_T_1 = y_n*(1.-mask_2)+mask_2*y_T_1
                        #y_T_2 = y_n*(1.-mask_3)+mask_3*y_T_2
                        #y_T_3 = y_n*(1.-mask_4)+mask_4*y_T_3
                if i == 0:
                    y_T = y_0*(1.-mask_1)+mask_1*y_T
                    y_T_1 = y_0*(1.-mask_2)+mask_2*y_T_1
                    #y_T_2 = y_0*(1.-mask_3)+mask_3*y_T_2
                    #y_T_3 = y_0*(1.-mask_4)+mask_4*y_T_3
                if i % sample_inter == 0:
                    ret_arr = torch.cat([ret_arr, y_t], dim=0)
           
            return y_T,y_T_1,mask_1,mask_2,y_n,y_0, ret_arr
    def forward(self, y_0, y_cond=None, mask=None, noise=None):
        # sampling from p(gammas)
        b, *_ = y_0.shape
        t = torch.randint(1, self.num_timesteps, (b,), device=y_0.device).long()
        gamma_t1 = extract(self.gammas, t-1, x_shape=(1, 1))
        sqrt_gamma_t2 = extract(self.gammas, t, x_shape=(1, 1))
        sample_gammas = (sqrt_gamma_t2-gamma_t1) * torch.rand((b, 1), device=y_0.device) + gamma_t1
        sample_gammas = sample_gammas.view(b, -1)

        noise = default(noise, lambda: torch.randn_like(y_0))
        y_noisy = self.q_sample(
            y_0=y_0, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise)

        if mask is not None:
            noise_hat = self.denoise_fn(torch.cat([y_cond, y_noisy*mask+(1.-mask)*y_0], dim=1), sample_gammas)
            loss = self.loss_fn(mask*noise, mask*noise_hat)
        else:
            noise_hat = self.denoise_fn(torch.cat([y_cond, y_noisy], dim=1), sample_gammas)
            loss = self.loss_fn(noise, noise_hat)
        return loss


# gaussian diffusion trainer class
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def extract(a, t, x_shape=(1,1,1,1)):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

# beta_schedule function
def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas

def make_beta_schedule(schedule, n_timestep, linear_start=1e-6, linear_end=1e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


