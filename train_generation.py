import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7,8, 9"

from collections import OrderedDict
import re
import gc

import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

import argparse
from torch.distributions import Normal

from utils.file_utils import *
from utils.visualize import *
from model.pvcnn_generation import PVCNN2Base
import torch.distributed as dist
from datasets.dataset_generation import ToothDataset, FDIS
from datasets.augmentation_techniques import *
from torchvision import transforms
import json
from tqdm import tqdm


'''
----- Some utilities -----
'''

def norm(v, f):
    v = (v - v.min()) / (v.max() - v.min()) - 0.5
    return v, f

def getGradNorm(net):
    pNorm = torch.sqrt(sum(torch.sum(p ** 2) for p in net.parameters()))
    gradNorm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in net.parameters()))
    return pNorm, gradNorm

def weights_init(m):
    """
    xavier initialization
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and m.weight is not None:
        torch.nn.init.xavier_normal_(m.weight)

    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_()
        m.bias.data.fill_(0)

''' 
----- Models ----- 
'''

def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2)
                  + (mean1 - mean2) ** 2 * torch.exp(-logvar2))

def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    # Assumes data is integers [0, 1]
    assert x.shape == means.shape == log_scales.shape
    px0 = Normal(torch.zeros_like(means), torch.ones_like(log_scales))

    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 0.5)
    cdf_plus = px0.cdf(plus_in)
    min_in = inv_stdv * (centered_x - .5)
    cdf_min = px0.cdf(min_in)
    log_cdf_plus = torch.log(torch.max(cdf_plus, torch.ones_like(cdf_plus) * 1e-12))
    log_one_minus_cdf_min = torch.log(torch.max(1. - cdf_min, torch.ones_like(cdf_min) * 1e-12))
    cdf_delta = cdf_plus - cdf_min

    log_probs = torch.where(
        x < 0.001, log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min,
                    torch.log(torch.max(cdf_delta, torch.ones_like(cdf_delta) * 1e-12))))
    assert log_probs.shape == x.shape
    return log_probs

class GaussianDiffusion:
    def __init__(self, betas, loss_type, model_mean_type, model_var_type):
        self.loss_type = loss_type
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        assert isinstance(betas, np.ndarray)
        self.np_betas = betas = betas.astype(np.float64)  # computations here in float64 for accuracy
        assert (betas > 0).all() and (betas <= 1).all()
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # No orientation loss - orientation is handled as pure conditioning input (like CrownGen's bounding box)
        # initialize twice the actual length so we can keep running for eval
        # betas = np.concatenate([betas, np.full_like(betas[:int(0.2*len(betas))], betas[-1])])

        alphas = 1. - betas
        alphas_cumprod = torch.from_numpy(np.cumprod(alphas, axis=0)).float()
        alphas_cumprod_prev = torch.from_numpy(np.append(1., alphas_cumprod[:-1])).float()

        self.betas = torch.from_numpy(betas).float()
        self.alphas_cumprod = alphas_cumprod.float()
        self.alphas_cumprod_prev = alphas_cumprod_prev.float()

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).float()
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod).float()
        self.log_one_minus_alphas_cumprod = torch.log(1. - alphas_cumprod).float()
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod).float()
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1).float()

        betas = torch.from_numpy(betas).float()
        alphas = torch.from_numpy(alphas).float()
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.posterior_variance = posterior_variance
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(
            torch.max(posterior_variance, 1e-20 * torch.ones_like(posterior_variance)))
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)
        self.posterior_mean_coef3 = (
            1.0 + ((torch.sqrt(self.alphas_cumprod) - 1.) * ( torch.sqrt(self.alphas_cumprod_prev) + torch.sqrt(alphas)))
            / (1.0 - self.alphas_cumprod))

    @staticmethod
    def _extract(a, t, x_shape):
        """
        Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
        """
        bs, = t.shape
        assert x_shape[0] == bs
        out = torch.gather(a, 0, t)
        assert out.shape == torch.Size([bs])

        return torch.reshape(out, [bs] + ((len(x_shape) - 1) * [1]))

    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start
        variance = self._extract(1. - self.alphas_cumprod.to(x_start.device), t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape)

        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """ Diffuse the data (t == 0 means diffused for 1 step) """
        if noise is None:
            noise = torch.randn(x_start.shape, device=x_start.device)

        assert noise.shape == x_start.shape

        return (self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start +
                self._extract(self.sqrt_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape) * noise)

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """ Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0) """
        assert x_start.shape == x_t.shape
        posterior_mean = (self._extract(self.posterior_mean_coef1.to(x_start.device), t, x_t.shape) * x_start +
                          self._extract(self.posterior_mean_coef2.to(x_start.device), t, x_t.shape) * x_t)
        posterior_variance = self._extract(self.posterior_variance.to(x_start.device), t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped.to(x_start.device), t,
                                                       x_t.shape)
        assert (posterior_mean.shape[0] == posterior_variance.shape[0] == posterior_log_variance_clipped.shape[0] ==
                x_start.shape[0])

        return posterior_mean, posterior_variance, posterior_log_variance_clipped
                                                                        
    def p_mean_variance(self, denoise_fn, xt, model_kwargs, t, return_pred_xstart: bool):

        model_output, saved_attns  = denoise_fn(xt, 
                                    t, 
                                    model_kwargs)

        device = xt.device
        shape = xt.shape

        if self.model_var_type in ['fixedsmall', 'fixedlarge']:  
            # below: only log_variance is used in the KL computations
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so to get a better decoder log likelihood
                'fixedlarge': (self.betas.to(device),
                               torch.log(torch.cat([self.posterior_variance[1:2], self.betas[1:]])).to(device)),
                'fixedsmall': (self.posterior_variance.to(device),
                               self.posterior_log_variance_clipped.to(device))}[self.model_var_type]

            model_variance = self._extract(model_variance, t, shape) * torch.ones_like(model_output)
            model_log_variance = self._extract(model_log_variance, t, shape) * torch.ones_like(model_output)

        else:
            raise NotImplementedError(self.model_var_type)



        if self.model_mean_type == 'eps':
            x_recon = self._predict_xstart_from_eps(xt, t=t, eps=model_output)
            x_recon = x_recon.clamp(-1.5, 1.5)  # Prevent amplification of poor high-t predictions

            model_mean, _, _ = self.q_posterior_mean_variance(x_start=x_recon, x_t=xt, t=t)

        else:
            raise NotImplementedError(self.loss_type)

        assert model_mean.shape == x_recon.shape
        assert model_variance.shape == model_log_variance.shape

        if return_pred_xstart:
            return model_mean, model_variance, model_log_variance, x_recon, saved_attns
        else:
            return model_mean, model_variance, model_log_variance, saved_attns

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (self._extract(self.sqrt_recip_alphas_cumprod.to(x_t.device), t, x_t.shape) * x_t -
                self._extract(self.sqrt_recipm1_alphas_cumprod.to(x_t.device), t, x_t.shape) * eps)
    
    ''' 
    ----- Sampling ----- 
    '''

    def p_sample(self, denoise_fn, xt, model_kwargs, t, noise_fn, return_pred_xstart=False):
        """ Sample from the model """

        model_mean, _, model_log_variance, pred_xstart, saved_attns = self.p_mean_variance(denoise_fn, xt=xt, model_kwargs=model_kwargs, t=t,
                                                                              return_pred_xstart=True)
    

        noise = noise_fn(size=model_mean.shape, dtype=model_mean.dtype, device=model_mean.device)

        # no noise when t == 0 
        nonzero_mask = torch.reshape(1 - (t == 0).float(), [xt.shape[0]] + [1] * (len(model_mean.shape) - 1))

        sample = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise

        return (sample, pred_xstart, saved_attns) if return_pred_xstart else (sample, saved_attns)

    def p_sample_loop(self, model_kwargs, denoise_fn,  noise_fn=torch.randn, keep_running=False):
        """
        Generate samples
        keep_running: True if we run 2 x num_timesteps, False if we just run num_timesteps
        """
        x0 = model_kwargs['x0']
        B = x0.shape[0]
        device = x0.device
        img = noise_fn(size=x0.shape, dtype=torch.float, device=device)

        for t in tqdm(reversed(range(0, self.num_timesteps if not keep_running else len(self.betas)))):

            t_ = torch.empty(B, dtype=torch.int64, device=device).fill_(t)

            img, _ = self.p_sample(denoise_fn=denoise_fn, xt=img, model_kwargs = model_kwargs, t=t_, noise_fn=noise_fn, return_pred_xstart=False)

            # Clear GPU cache every 100 steps during sampling to prevent memory buildup
            if t % 100 == 0:
                torch.cuda.empty_cache()

        return img

    def ddim_sample_loop(self, model_kwargs, denoise_fn, ddim_steps=100, eta=0.0, noise_fn=torch.randn):
        """
        DDIM deterministic sampling using the same trained model weights.
        Much faster (ddim_steps << num_timesteps) and avoids error accumulation.
        eta=0.0: fully deterministic; eta=1.0: equivalent to DDPM stochastic.
        """
        x0 = model_kwargs['x0']
        B = x0.shape[0]
        device = x0.device

        # Select evenly spaced subset of timesteps
        step_size = self.num_timesteps // ddim_steps
        timesteps = list(reversed(range(0, self.num_timesteps, step_size)))  # e.g. [999, 989, ..., 9]

        img = noise_fn(size=x0.shape, dtype=torch.float, device=device)

        for i, t in enumerate(tqdm(timesteps)):
            t_ = torch.empty(B, dtype=torch.int64, device=device).fill_(t)
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1

            with torch.no_grad():
                eps, _ = denoise_fn(img, t_, model_kwargs)

            # Predict x0 and clamp
            alpha_t = self.alphas_cumprod[t].to(device)
            x0_pred = (img - (1 - alpha_t).sqrt() * eps) / alpha_t.sqrt()
            x0_pred = x0_pred.clamp(-1.5, 1.5)

            if t_prev < 0:
                img = x0_pred
            else:
                alpha_prev = self.alphas_cumprod[t_prev].to(device)
                sigma = eta * ((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)).sqrt()
                noise = noise_fn(size=img.shape, dtype=img.dtype, device=device)
                img = alpha_prev.sqrt() * x0_pred \
                    + (1 - alpha_prev - sigma ** 2).clamp(min=0).sqrt() * eps \
                    + sigma * noise

            if t % 100 == 0:
                torch.cuda.empty_cache()

        return img

    ''' 
    ----- Losses ----- 
    '''

    def mse_mean_flat(self, B, noise, eps_recon, mask, debug=False):
        """
        Take the mean over all non-batch dimensions, considering only unmasked elements.
        Apply higher loss weight (1.5x) to 2nd molar teeth (indices 0, 1, 26, 27).

        Args:
            noise: (B, 28, 3, 1024) - ground truth noise
            eps_recon: (B, 28, 3, 1024) - predicted noise
            mask: (B, 28, 1, 1) - latent mask indicating which teeth to reconstruct
            debug: If True, print detailed debugging information
        """
        # Define tooth weights: 1.5x higher for 2nd molars (indices 0, 1, 26, 27)
        # These correspond to FDI: 17 (upper right), 47 (lower right), 27 (upper left), 37 (lower left)
        tooth_weights = torch.ones(28, device=noise.device)
        molar_weight = 1.5  # 1.5x weight for 2nd molars
        tooth_weights[0] = molar_weight   # Index 0 = FDI 17 (upper right 2nd molar)
        tooth_weights[1] = molar_weight   # Index 1 = FDI 47 (lower right 2nd molar)
        tooth_weights[26] = molar_weight  # Index 26 = FDI 27 (upper left 2nd molar)
        tooth_weights[27] = molar_weight  # Index 27 = FDI 37 (lower left 2nd molar)

        # Reshape weights to (1, 28, 1, 1) for broadcasting
        tooth_weights = tooth_weights.view(1, 28, 1, 1)

        # Expand mask to match noise dimensions: (B, 28, 1, 1) -> (B, 28, 3, 1024)
        mask_expanded = mask.expand_as(noise)  # (B, 28, 3, 1024)

        # Expand weights to match noise dimensions: (1, 28, 1, 1) -> (B, 28, 3, 1024)
        weights_expanded = tooth_weights.expand_as(noise)  # (B, 28, 3, 1024)

        # DEBUG: Check mask and weights
        if debug:
            print(f"[MSE DEBUG] mask shape: {mask.shape}, num_nonzero: {mask.sum().item()}")
            print(f"[MSE DEBUG] mask_expanded shape: {mask_expanded.shape}, num_nonzero: {mask_expanded.sum().item()}")
            print(f"[MSE DEBUG] tooth_weights for 2nd molars: {molar_weight}")

        # Apply mask and compute weighted MSE only on latent teeth
        masked_diff = (noise - eps_recon) * mask_expanded

        # DEBUG: Check inputs
        if debug:
            print(f"[MSE DEBUG] noise - min: {noise.min():.4f}, max: {noise.max():.4f}, has_nan: {torch.isnan(noise).any()}")
            print(f"[MSE DEBUG] eps_recon - min: {eps_recon.min():.4f}, max: {eps_recon.max():.4f}, has_nan: {torch.isnan(eps_recon).any()}")
            print(f"[MSE DEBUG] masked_diff - min: {masked_diff.min():.4f}, max: {masked_diff.max():.4f}, has_nan: {torch.isnan(masked_diff).any()}")

        # Apply weights to squared differences
        weighted_squared_diff = (masked_diff ** 2) * weights_expanded

        # DDP FIX: All zero-loss returns must stay connected to eps_recon (model output)
        # so that DDP gradient reduction sees all model parameters as participating.
        # Using (eps_recon * 0).sum() gives value=0 but preserves the graph.
        graph_connected_zero = (eps_recon * 0).sum()

        # Check for NaN/Inf in intermediate computation
        if torch.isnan(weighted_squared_diff).any() or torch.isinf(weighted_squared_diff).any():
            if debug:
                print(f"[MSE DEBUG] *** RETURNING 0.0: NaN/Inf in weighted_squared_diff ***")
            return graph_connected_zero

        # Normalize by weighted number of masked elements for proper weighted averaging
        # This ensures the loss scale remains similar to unweighted version
        weighted_num_elements = (mask_expanded * weights_expanded).sum()

        if debug:
            print(f"[MSE DEBUG] weighted_num_elements: {weighted_num_elements.item()}")

        if weighted_num_elements == 0:
            # No latent teeth to reconstruct - return zero loss
            if debug:
                print(f"[MSE DEBUG] *** RETURNING 0.0: weighted_num_elements == 0 ***")
            return graph_connected_zero

        # Weighted loss: sum of weighted squared differences / sum of weights
        loss = weighted_squared_diff.sum() / weighted_num_elements

        if debug:
            print(f"[MSE DEBUG] computed weighted loss: {loss.item():.6f}")

        # Final check for NaN/Inf in computed loss
        if torch.isnan(loss) or torch.isinf(loss):
            if debug:
                print(f"[MSE DEBUG] *** RETURNING 0.0: NaN/Inf in final loss ***")
            return graph_connected_zero

        return loss

    def p_losses(self, denoise_fn, t, noise, model_kwargs, debug=False):

        data_start = model_kwargs['x0']

        B = data_start.shape[0]
        assert t.shape == torch.Size([B])

        data_t = self.q_sample(x_start=data_start, t=t, noise=noise)

        if self.loss_type == 'mse':

            # DDP FIX: Capture attn_mask to ensure all model parameters receive gradients
            eps_recon, attn_mask  = denoise_fn(data_t,
                                       t,
                                       model_kwargs,
                                       debug_nan=debug)

            if debug:
                print(f"[P_LOSSES DEBUG] eps_recon shape: {eps_recon.shape}")
                print(f"[P_LOSSES DEBUG] eps_recon - min: {eps_recon.min():.4f}, max: {eps_recon.max():.4f}, has_nan: {torch.isnan(eps_recon).any()}")

            # Simple MSE loss on target teeth only (no orientation loss - it's handled as conditioning)
            losses = self.mse_mean_flat(B, noise, eps_recon, model_kwargs['l_mask'], debug=debug)

            if debug:
                print(f"[P_LOSSES DEBUG] mse_loss: {losses.item():.6f}")

            # DDP FIX: ALWAYS add attn_mask to ensure consistent computational graph across all batches
            # This prevents "unused parameters" error in DDP (multiplying by 0 doesn't change loss value)
            if attn_mask is not None:
                losses = losses + (attn_mask * 0.0).sum()

            # DDP FIX: Add eps_recon to loss even if it's already used, to ensure graph connection
            # This guarantees all model outputs participate in loss computation
            losses = losses + (eps_recon * 0.0).mean()

        elif self.loss_type == 'kl':
            pass
            # losses = self._vb_terms_bpd(
            #     denoise_fn=denoise_fn, data_start=data_start, data_t=data_t, t=t, clip_denoised=False,
            #     return_pred_xstart=False)
        else:
            raise NotImplementedError(self.loss_type)

        return losses



class PVCNN2(PVCNN2Base):

    num_n = 128 # Number of neighbors

    sa_blocks = [
        ((32, 2, 32), (512, 0.1, 32, (32, 64))),
        ((64, 3, 16), (256, 0.2, 32, (64, 128))),
        ((128, 3, 8), (64, 0.4, 32, (128, 256))),
        (None, (16, 0.8, 32, (256, 256, 512))),
    ]
    fp_blocks = [
        ((256, 256), (256, 3, 8)),
        ((256, 256), (256, 3, 8)),
        ((256, 128), (128, 2, 16)),
        ((128, 128, 64), (64, 2, 32)),
    ]

    def __init__(self, num_classes, embed_dim, use_att, dropout, extra_feature_channels=3,
                 width_multiplier=1.0, voxel_resolution_multiplier=1.0):
        super().__init__(num_classes=num_classes, embed_dim=embed_dim, use_att=use_att,
                         dropout=dropout, extra_feature_channels=extra_feature_channels,
                         width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier)


class Model(nn.Module):
    def __init__(self, args, betas, loss_type: str, model_mean_type: str, model_var_type: str,
                 width_mult: float, vox_res_mult: float):
        super(Model, self).__init__()

        # Create diffusion
        self.diffusion = GaussianDiffusion(betas, loss_type, model_mean_type, model_var_type)

        # Create point-voxel-cnn network (root axis conditioning always active via per-point concat)
        self.model = PVCNN2(num_classes=args.nc, embed_dim=args.embed_dim,
                            use_att=args.attention, dropout=args.dropout, extra_feature_channels=args.extra_feature_nc,
                            width_multiplier=width_mult, voxel_resolution_multiplier=vox_res_mult)


    def _denoise(self, xt, t, model_kwargs, debug_nan=False):
        B = xt.shape[0]

        assert xt.dtype == torch.float
        assert t.shape == torch.Size([B]) and t.dtype == torch.int64

        # Extract fdi_indices (embedding indices [0-27] mapped from FDI positions)
        fdi_indices = model_kwargs['fdi_indices']
        out, attn_mask = self.model(xt, t, model_kwargs['x0'], fdi_indices,
                                     model_kwargs['l_mask'], model_kwargs['o_mask'],
                                     model_kwargs.get('original_missing_mask'),
                                     root_axis=model_kwargs.get('root_axis'),
                                     debug_nan=debug_nan)

        return out, attn_mask

    def get_loss_iter_teethmask(self, noise_batch, model_kwargs, debug=False):

        dentition_points = model_kwargs['x0']
        B = dentition_points.shape[0]
        t = torch.randint(0, self.diffusion.num_timesteps, size=(B,), device=noise_batch.device)

        if debug:
            print(f"[GET_LOSS DEBUG] batch size: {B}, timestep range: [0, {self.diffusion.num_timesteps})")
            print(f"[GET_LOSS DEBUG] dentition_points shape: {dentition_points.shape}")
            print(f"[GET_LOSS DEBUG] l_mask sum: {model_kwargs['l_mask'].sum().item()}")

        losses = self.diffusion.p_losses(denoise_fn=self._denoise,
                                         t=t,
                                         noise=noise_batch,
                                         model_kwargs=model_kwargs,
                                         debug=debug)

        return losses

    def gen_samples(self, model_kwargs, noise_fn=torch.randn, keep_running=False, use_ddim=True, ddim_steps=100, ddim_eta=0.0):

        if use_ddim:
            out = self.diffusion.ddim_sample_loop(model_kwargs, self._denoise,
                                                  ddim_steps=ddim_steps, eta=ddim_eta, noise_fn=noise_fn)
        else:
            out = self.diffusion.p_sample_loop(model_kwargs, self._denoise, noise_fn=noise_fn,
                                               keep_running=keep_running)

        return out

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def multi_gpu_wrapper(self, f):
        self.model = f(self.model)

    # No set_epoch needed - orientation is pure conditioning (no loss warmup)

def validate_training_scenario(latent_mask, obs_mask, original_missing_mask, batch_idx, logger):
    """
    Validate that the training scenario is realistic:
    1. We're not trying to reconstruct naturally missing teeth
    2. We have sufficient context teeth for meaningful training
    3. The latent and observation masks are consistent
    """
    B = latent_mask.shape[0]

    for b in range(B):
        # Check 1: Ensure latent teeth were originally present
        if original_missing_mask is not None:
            latent_indices = torch.where(latent_mask[b, :, 0, 0] == 1)[0]
            originally_missing_indices = torch.where(original_missing_mask[b])[0]

            invalid_latent = torch.isin(latent_indices, originally_missing_indices)
            if torch.any(invalid_latent):
                invalid_count = torch.sum(invalid_latent).item()
                if logger:
                    logger.warning(f"Batch {batch_idx}, sample {b}: Trying to reconstruct {invalid_count} naturally missing teeth")
                return False

        # Check 2: Ensure we have sufficient context teeth
        obs_count = torch.sum(obs_mask[b, :, 0, 0]).item()
        latent_count = torch.sum(latent_mask[b, :, 0, 0]).item()

        if obs_count < 2:  # Need at least 2 context teeth
            if logger:
                logger.warning(f"Batch {batch_idx}, sample {b}: Insufficient context teeth ({obs_count})")
            return False

        if latent_count == 0:  # Need at least 1 tooth to reconstruct
            if logger:
                logger.warning(f"Batch {batch_idx}, sample {b}: No teeth to reconstruct")
            return False

        # Check 2b: Ensure minimum total existing teeth for meaningful training
        total_existing = obs_count + latent_count
        if total_existing < 3:
            if logger:
                logger.warning(f"Batch {batch_idx}, sample {b}: Too few total existing teeth ({total_existing})")
            return False

        # Check 3: Ensure masks are consistent (no overlap)
        overlap = torch.sum(latent_mask[b, :, 0, 0] * obs_mask[b, :, 0, 0]).item()
        if overlap > 0:
            if logger:
                logger.warning(f"Batch {batch_idx}, sample {b}: Latent and observation masks overlap")
            return False

    return True

def get_betas(schedule_type, b_start, b_end, time_num):
    if schedule_type == 'linear':
        betas = np.linspace(b_start, b_end, time_num)

    elif schedule_type == 'warm0.1':
        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.1)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)

    elif schedule_type == 'warm0.2':
        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.2)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)

    elif schedule_type == 'warm0.5':
        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.5)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)

    else:
        raise NotImplementedError(schedule_type)

    return betas



def get_dataset(path, tooth_npoints, mode='train'):

    dataset = ToothDataset(
                                path = path,
                                mode=mode, 
                                tooth_npoints = tooth_npoints,
                                aug_transforms=transforms.Compose(
                                    [
                                        RandomMirror(p=0.5),
                                        # Increased rotation ranges for better generalization
                                        # Changed from ±1/48 (~3.75°) to ±1/12 (~15°) for x,y and ±1/6 (~30°) for z
                                        RandomRotate(angle=[-1 / 12, 1 / 12], axis="x", p=0.5),
                                        RandomRotate(angle=[-1 / 12, 1 / 12], axis="y", p=0.5),
                                        RandomRotate(angle=[-1 / 6, 1 / 6], axis="z", p=0.5),
                                        RandomScale(scale=[0.9, 1.1], p=0.75),  # Increased from [0.95,1.05]
                                        ShufflePoint(p=0.9)
                                    ] 
                )) if mode == 'train' else ToothDataset(
                                path = path,
                                mode=mode, 
                                tooth_npoints = tooth_npoints,
                                aug_transforms=None
                        )

    print(f'How many training dataset? : {len(dataset)}')

    return dataset


def get_dataloader(opt, dataset, local_rank, mode = 'train'):

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=opt.world_size,
        rank=local_rank,
        shuffle = (mode=='train')
    )

    dataloader = torch.utils.data.DataLoader(dataset,
                                                batch_size=opt.bs,
                                                sampler=sampler,
                                                shuffle=sampler is None,
                                                num_workers=int(opt.workers),
                                                pin_memory=True,  # CRITICAL: Fast CPU→GPU transfer
                                                persistent_workers=True,  # CRITICAL: Keep workers alive between epochs
                                                prefetch_factor=4,  # CRITICAL: Prefetch 4 batches per worker
                                                drop_last=True)  # CRITICAL: All DDP ranks must process same number of batches
    
    return dataloader, sampler


def generate_val_samples(opt, model, val_dataset, outf_syn, epoch, device, logger):
    logger.info('Generation: eval')

    model.eval()

    sample_batch_size = 1 # how many validation samples to generate
    val_sample_list = val_dataset.sample_patient(sample_batch_size)
    val_dentition_points = torch.stack([sample['dentition_points'] for sample in val_sample_list], 0).to(device)
    val_fdi_indices = torch.stack([sample['fdi_indices'] for sample in val_sample_list], 0).to(device)  # Embedding indices [0-27] from FDI positions
    val_bound = None
    if 'bounds_cyl' in val_sample_list[0]:
        val_bound = torch.stack([sample['bounds_cyl'] for sample in val_sample_list], 0).to(device)
    val_dentition_ids = [sample['patient_id'] for sample in val_sample_list]

    # Handle original missing teeth in validation
    val_original_missing_mask = None
    if 'original_missing_mask' in val_sample_list[0]:
        val_original_missing_mask = torch.stack([sample['original_missing_mask'] for sample in val_sample_list], 0).to(device)

    latent_mask_val = torch.zeros_like(val_dentition_points[:,:,:1,:1]).to(device)

    # Seed RNG for validation sampling (for reproducibility)
    val_seed = opt.manualSeed + epoch * 999
    random.seed(val_seed)
    torch.manual_seed(val_seed)

    for i in range(sample_batch_size):
        # Get existing teeth for validation sampling
        if val_original_missing_mask is not None:
            existing_teeth_indices = torch.where(~val_original_missing_mask[i])[0]
        else:
            existing_teeth_indices = torch.arange(val_dentition_points.shape[1], device=device)

        if len(existing_teeth_indices) == 0:
            continue  # Skip if no teeth exist

        # Select fewer teeth for validation to get cleaner results
        n_missing_val = min(random.randint(1, min(3, opt.max_missing_teeth)), len(existing_teeth_indices))

        # Simple random selection (wisdom teeth already excluded from 28-tooth dataset)
        selected_existing = existing_teeth_indices[torch.randperm(len(existing_teeth_indices))[:n_missing_val]]
        latent_mask_val[i, selected_existing, 0, 0] = 1

    obs_mask_val = torch.ones_like(latent_mask_val) - latent_mask_val

    # Ensure we don't try to observe naturally missing teeth in validation
    if val_original_missing_mask is not None:
        obs_mask_val = obs_mask_val * (~val_original_missing_mask.unsqueeze(-1).unsqueeze(-1))

    # Load pre-computed root axis from dataset (same pattern as training)
    val_root_axis = torch.stack([sample['root_axis'] for sample in val_sample_list], 0).to(device)  # (B, 28, 3)

    val_data_dict = {
        'x0': val_dentition_points,
        'fdi_indices': val_fdi_indices,  # Embedding indices [0-27] from FDI positions
        'l_mask': latent_mask_val,
        'o_mask':obs_mask_val,
        'bound':val_bound,
        'root_axis': val_root_axis,  # Root axis conditioning (same pattern as bound!)
        'original_missing_mask': val_original_missing_mask  # For attention masking
    }

    # generate some samples
    with torch.no_grad():

        sample_out = model.gen_samples(model_kwargs=val_data_dict).detach().cpu()
        val_dentition_points = val_dentition_points.detach().cpu()
        obs_mask_val = obs_mask_val.detach().cpu()
        latent_mask_val = latent_mask_val.detach().cpu()

        # CRITICAL FIX: Denormalize generated samples back to original space
        # Model generates in normalized space [-1, 1], but we need original coordinates
        DENORM_SHIFT = np.array([[-0.72, 25.35, -6.64]])  # From dataset normalization
        DENORM_SCALE = 39.2  # From dataset normalization

        # Compute validation metrics for REAL geometry quality
        val_metrics = []

        for i in range(sample_batch_size):

            missing_idx = np.where(latent_mask_val.numpy().squeeze())[0]

            condition_points = val_dentition_points[i][obs_mask_val[i].view(-1).bool()]
            generated_points = sample_out[i][latent_mask_val[i].view(-1).bool()]
            print(f"[DIAG] raw generated range: {generated_points.min():.1f} to {generated_points.max():.1f}")
            print(f"[DIAG] raw generated centroid: {generated_points.mean((0,2))}")
            ground_truth_points = val_dentition_points[i][latent_mask_val[i].view(-1).bool()]
            patient_id = val_dentition_ids[i]

            # DENORMALIZE ALL: condition, generated, and ground truth
            # All are in normalized space (3, N) from the dataset
            # Need to denormalize to (N, 3) in original coordinate system

            def denormalize_points(points_tensor):
                """Denormalize points from (K, 3, N) normalized space to (K, 3, N) original space.
                export_dentition_npy handles the transpose/reshape to (K*N, 3) internally."""
                points_np = points_tensor.numpy()  # (K, 3, N)
                points_denorm = points_np * DENORM_SCALE + DENORM_SHIFT.T  # (K, 3, N), DENORM_SHIFT.T=(3,1)
                return points_denorm  # (K, 3, N)

            condition_np = denormalize_points(condition_points)  # (K, 3, N)
            gen_np       = denormalize_points(generated_points)  # (M, 3, N)
            gt_np        = denormalize_points(ground_truth_points)  # (M, 3, N)

            # Reshape to (K*N, 3) / (M*N, 3) for metric computation
            gen_pts = gen_np.transpose(0, 2, 1).reshape(-1, 3)
            gt_pts  = gt_np.transpose(0, 2, 1).reshape(-1, 3)

            # Scale/spread metric (computed in original space)
            gen_std = np.std(gen_pts)
            gt_std = np.std(gt_pts)
            std_ratio = gen_std / (gt_std + 1e-6)

            # Centroid alignment (in original space)
            gen_mean = np.mean(gen_pts, axis=0)
            gt_mean = np.mean(gt_pts, axis=0)
            centroid_dist = np.linalg.norm(gen_mean - gt_mean)

            val_metrics.append({
                'gen_std': gen_std,
                'gt_std': gt_std,
                'std_ratio': std_ratio,
                'centroid_dist': centroid_dist
            })

            # Save DENORMALIZED points for visualization
            export_dentition_npy('{}_{}_condition'.format(patient_id,missing_idx), '%s/epoch_%03d' % (outf_syn, epoch), condition_np)
            export_dentition_npy('{}_{}_gen'.format(patient_id,missing_idx), '%s/epoch_%03d' % (outf_syn, epoch), gen_np)
            export_dentition_npy('{}_{}_gt'.format(patient_id,missing_idx), '%s/epoch_%03d' % (outf_syn, epoch), gt_np)

        # Log validation metrics
        if len(val_metrics) > 0:
            avg_std_ratio = np.mean([m['std_ratio'] for m in val_metrics])
            avg_centroid_dist = np.mean([m['centroid_dist'] for m in val_metrics])
            logger.info(f'[VALIDATION] Epoch {epoch} - Std Ratio: {avg_std_ratio:.2f}x (target: 1.0x), Centroid Dist: {avg_centroid_dist:.4f}')
            if avg_std_ratio > 3.0:
                logger.warning(f'[VALIDATION] Generated points are {avg_std_ratio:.1f}x too spread out! Model needs more training.')

    model.train()


def remove_module_prefix(state_dict):

    new_state_dict = OrderedDict()
    pattern = re.compile('module.')

    for k,v in state_dict.items():
        if re.search("module", k):
            new_state_dict[re.sub(pattern, '', k)] = v    
        else:
            new_state_dict[k] = v

    return new_state_dict


def train(local_rank, opt, output_dir):
    logger = setup_logging(output_dir)

    # DDP 관련
    take_action = local_rank == 0
    # we only need one process to do auxillary stuff during training.

    if take_action:
        outf_syn, = setup_output_subdirs(output_dir, 'syn')
        logger.info(opt) # print options

    # CRITICAL FIX: Set random seed on ALL ranks for reproducibility and DDP sync
    # Each rank needs the same base seed so random data selection is synchronized
    import random
    import numpy as np
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed_all(opt.manualSeed)
    if take_action:
        logger.info(f'Random seed set to {opt.manualSeed} on all ranks')

    import datetime
    os.environ['TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC'] = '3600'
    dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=opt.world_size, rank=local_rank,
                            timeout=datetime.timedelta(minutes=60))

    opt.bs = int(opt.bs / opt.ngpus_per_node)
    opt.workers = int(opt.workers / opt.ngpus_per_node)


    ''' Dataset and data loader '''
    train_dataset = get_dataset(path = opt.path, tooth_npoints = opt.tooth_npoints, mode='train')
    val_dataset = get_dataset(path = opt.path, tooth_npoints = opt.tooth_npoints, mode='val')

    dataloader, sampler = get_dataloader(opt, train_dataset,local_rank, mode = 'train')


    ''' Create networks '''
    betas = get_betas(opt.schedule_type, opt.beta_start, opt.beta_end, opt.time_num)

    if take_action:
        logger.info(f'Root axis per-point conditioning active (3-ch concat, zeros for missing teeth)')

    model = Model(opt, betas, opt.loss_type, opt.model_mean_type, opt.model_var_type,
                  opt.width_mult, opt.vox_res_mult)

    # Use Adam optimizer with original settings (more stable for diffusion models)
    # AdamW with ReduceLROnPlateau was too aggressive - caused LR collapse
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.decay, betas=(0.5, 0.999), eps=1e-8)

    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)

    # CRITICAL: Aggressive memory management to prevent OOM
    torch.cuda.empty_cache()
    gc.collect()

    # Reduce memory fraction from 0.85 to 0.70 for safety margin
    if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
        torch.cuda.set_per_process_memory_fraction(0.70)

    # Enable expandable segments to reduce fragmentation (CRITICAL)
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'

    # DDP DEBUG: Enable detailed DDP logging to diagnose unused parameters
    # Set to 'DETAIL' for maximum verbosity, 'INFO' for moderate, or comment out to disable
    # os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

    # Log initial GPU memory state
    if take_action:
        logger.info(f'GPU {local_rank} - Initial memory allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB')
        logger.info(f'GPU {local_rank} - Initial memory reserved: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB')

    model = model.to(device)


    if opt.model != '':
        ckpt = torch.load(opt.model, map_location=device)

        # DDP CRITICAL FIX: Ensure all ranks load model the same way
        # Try-except with different code paths causes rank divergence
        # Always try to remove module prefix first (handles both DDP and non-DDP checkpoints)
        state_dict = ckpt['model_state']

        # Check if checkpoint is from DDP (has 'module.' prefix)
        is_ddp_checkpoint = any('module.' in k for k in state_dict.keys())

        if is_ddp_checkpoint:
            # Remove 'module.' prefix for loading into non-wrapped model
            state_dict = remove_module_prefix(state_dict)
            if take_action:
                logger.info(f'RANK {local_rank}: Loading DDP checkpoint (removed module prefix)')
        else:
            if take_action:
                logger.info(f'RANK {local_rank}: Loading non-DDP checkpoint (no prefix removal needed)')

        # strict=False: silently drops bound_embedding/bound_transformer/bound_final_ln keys
        # from old checkpoints — those parameters were always dead (bound=None always)
        model.load_state_dict(state_dict, strict=False)
        optimizer.load_state_dict(ckpt['optimizer_state'])

        # Override the LR restored from checkpoint with the one explicitly set via --lr.
        # load_state_dict restores param_groups[i]['lr'] from the saved state, which would
        # otherwise silently ignore any --lr argument passed at the command line.
        for param_group in optimizer.param_groups:
            param_group['lr'] = opt.lr

        if take_action:
            logger.info(f'RANK {local_rank}: Checkpoint loaded successfully')
            logger.info(f'RANK {local_rank}: Learning rate set to {opt.lr:.2e} (overrides checkpoint value)')

        start_epoch = ckpt['epoch'] + 1
    else:
        start_epoch = 0


    def _transform_(m):
        return nn.parallel.DistributedDataParallel(
            m, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=False,  # CRITICAL: Changed to False - all params must be used
            broadcast_buffers=True)

    model.multi_gpu_wrapper(_transform_)

    # Fixed LR decay schedule: decay at 1/4, 1/2, 3/4 of total epochs
    # For niter=10000: decays at epochs 2500, 5000, 7500
    lr_decay_epochs = [(opt.niter//4), (opt.niter//4)*2, (opt.niter//4)*3]
    if take_action:
        logger.info(f'Learning rate will decay by factor {opt.lr_decay_factor} at epochs: {lr_decay_epochs}')

    for epoch in range(start_epoch, opt.niter):

        sampler.set_epoch(epoch)

        model.train()  # Ensure model is in training mode (critical for DDP)

        logger.info(f'RANK {local_rank}: Training start at epoch {epoch}')

        # Track epoch loss for logging
        epoch_losses = []

        # DDP SYNCHRONIZATION FIX:
        # All ranks MUST execute the same operations (forward/backward/step) in every iteration.
        # We sanitize bad data (NaN/Inf) instead of skipping, so all ranks stay in sync.
        # NEVER use conditional optimizer.step() or continue statements - they cause NCCL timeouts.

        for i, data in enumerate(dataloader):

            # DDP FIX: Removed per-iteration reseeding which was causing RNG divergence
            # Each rank processes different data, so forcing same RNG state causes
            # different consumption patterns → accumulated divergence → NCCL timeout
            # Correct DDP pattern: seed once globally (line 925-932), then let ranks evolve independently

            dentition_points = data['dentition_points'].to(device) # (b, K, 3, 1024)
            fdi_indices = data['fdi_indices'].to(device)  # (b, 28) embedding indices [0-27] from FDI positions
            bound = data.get('bounds_cyl')  # Optional, may not be in dataset
            if bound is not None:
                bound = bound.to(device)  # (b, 28, 5)
            original_missing_mask = data['original_missing_mask'].to(device) if 'original_missing_mask' in data else None
            B, nT, nD, nP = dentition_points.shape

            latent_mask = torch.zeros_like(dentition_points[:,:,:1,:1]).to(device)

            for b in range(B):
                # Get existing teeth for this batch item
                if original_missing_mask is not None:
                    existing_teeth_indices = torch.where(~original_missing_mask[b])[0]
                else:
                    # Fallback: assume all teeth exist (legacy behavior)
                    existing_teeth_indices = torch.arange(nT, device=device)

                # Simplified random selection (wisdom teeth already excluded from 28-tooth dataset)
                if len(existing_teeth_indices) > 0:
                    # Randomly select number of teeth to simulate as missing
                    n_missing = min(torch.randint(1, opt.max_missing_teeth + 1, (1,)).item(), len(existing_teeth_indices))

                    # Simple random selection from existing teeth (no wisdom tooth filtering needed)
                    selected_indices = existing_teeth_indices[torch.randperm(len(existing_teeth_indices))[:n_missing]]
                    latent_mask[b, selected_indices, 0, 0] = 1
                else:
                    # No teeth exist - set first position as dummy latent (will be masked out anyway)
                    latent_mask[b, 0, 0, 0] = 1

            obs_mask = torch.ones_like(latent_mask) - latent_mask

            # Ensure we don't try to observe naturally missing teeth
            if original_missing_mask is not None:
                # Set obs_mask to 0 for naturally missing teeth
                obs_mask = obs_mask * (~original_missing_mask.unsqueeze(-1).unsqueeze(-1))

            # REMOVED: Validation creates rank divergence since each rank has different data
            # Different masks per rank → different loss → different gradients → potential NCCL issues
            # Standard DDP pattern: Each rank processes its data independently without cross-validation
            # The masks are already correct from lines 1095-1109, no need to validate/override

            # IMPORTANT
            # obs_mask (observed mask): mask that indicates which teeth are existing context teeth
            # latent_mask: mask that indicates which teeth are simulated for omission and trying to reconstruct back
            noise_batch = torch.randn_like(dentition_points).to(device)

            # Load pre-computed root axis from dataset (same pattern as CrownGen's bound!)
            root_axis = data['root_axis'].to(device)  # (B, 28, 3) - pre-computed offline, loaded here

            # Validate input data before forward pass — sanitize instead of skipping
            # DDP CRITICAL: All ranks sanitize data the same way to stay in sync
            if torch.isnan(dentition_points).any() or torch.isinf(dentition_points).any():
                if take_action:
                    logger.warning(f'Batch {i} - NaN/Inf in dentition_points, sanitizing')
                dentition_points = torch.nan_to_num(dentition_points, nan=0.0, posinf=1.0, neginf=-1.0)

            if torch.isnan(noise_batch).any() or torch.isinf(noise_batch).any():
                if take_action:
                    logger.warning(f'Batch {i} - NaN/Inf in noise_batch, regenerating')
                noise_batch = torch.randn_like(noise_batch)  # Regenerate proper noise

            data_dict = {
                'x0': dentition_points,
                'fdi_indices': fdi_indices,  # Embedding indices [0-27] from FDI positions
                'l_mask': latent_mask,
                'o_mask':obs_mask,
                'bound':bound,                              # (B, 28, 5) - bounding box conditioning (original)
                'root_axis': root_axis,                     # (B, 28, 3) - root axis conditioning (same pattern!)
                'original_missing_mask': original_missing_mask
            }

            loss = model.get_loss_iter_teethmask(noise_batch, model_kwargs=data_dict, debug=False)

            # DDP CRITICAL: Sanitize NaN/Inf loss but DO NOT skip optimizer step
            # All ranks must execute backward() and optimizer.step() together
            if torch.isnan(loss) or torch.isinf(loss):
                if take_action:
                    logger.warning(f'Batch {i} - NaN/Inf loss detected, sanitizing to zero')
                loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

            # Optimize network parameters
            optimizer.zero_grad()

            # Backward pass — MUST always run for DDP AllReduce sync across all ranks
            # DDP CRITICAL: Removed try-except to ensure all ranks execute backward together
            # If OOM occurs, it will crash all ranks simultaneously (maintaining sync)
            loss.backward()

            # Clear cache after backward to free intermediate gradient tensors
            torch.cuda.empty_cache()

            # DDP DEBUG: Check for parameters without gradients (helps diagnose DDP issues)
            if i == 0 and epoch == start_epoch and take_action:
                params_without_grad = []
                for idx, param in enumerate(model.parameters()):
                    if param.requires_grad and (param.grad is None or torch.all(param.grad == 0)):
                        params_without_grad.append(idx)
                if len(params_without_grad) > 0:
                    logger.warning(f'First batch: {len(params_without_grad)} parameters without gradients: {params_without_grad[:10]}...')

            # Gradient clipping
            if opt.grad_clip is not None and opt.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)

            # DDP CRITICAL: Check for NaN gradients/parameters and sanitize them
            # DO NOT skip optimizer.step() - all ranks must step together
            has_nan = False
            for param in model.parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
                        has_nan = True

            if has_nan and take_action:
                logger.warning(f'Batch {i} - NaN/Inf gradients detected and sanitized')

            # DDP CRITICAL: ALWAYS call optimizer.step() on all ranks
            # This ensures all ranks stay synchronized
            optimizer.step()

        # Output Important Data during training process
            if i % opt.print_freq == 0 and take_action:
                loss_value = loss.item()

                # Mark sanitized batches in logs
                sanitized_marker = " [SANITIZED]" if has_nan else ""

                # Simple loss logging (no orientation loss components)
                logger.info('[{:>3d}/{:>3d}][{:>3d}/{:>3d}]    loss: {:>10.4f}{}'
                            .format(epoch, opt.niter, i, len(dataloader), loss_value, sanitized_marker))

            # Track all losses for epoch statistics (including sanitized ones)
            epoch_losses.append(loss.item())

            # Clean up after successful iteration to prevent memory accumulation
            del loss, data_dict

            # Periodic cache clearing to prevent fragmentation
            if i % 50 == 0:
                torch.cuda.empty_cache()

        # Clear GPU cache at the end of each epoch to prevent accumulation
        torch.cuda.empty_cache()
        gc.collect()

        # DDP CRITICAL: Synchronize all ranks at end of epoch
        # This ensures no rank gets ahead and helps detect desync issues early
        dist.barrier()

        if take_action:
            logger.info(f'Epoch {epoch} completed - GPU cache cleared, all ranks synchronized')

        # Log average epoch loss
        if len(epoch_losses) > 0 and take_action:
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f'Epoch {epoch} avg loss: {avg_epoch_loss:.6f}, Current LR: {current_lr:.2e}')

        # Fixed LR decay schedule (from original code)
        if (epoch+1) in lr_decay_epochs:
            for param_group in optimizer.param_groups:
                if take_action:
                    logger.info(f'Learning rate decay triggered at epoch {epoch+1}')
                    logger.info(f'  Old LR: {param_group["lr"]:.2e}')
                param_group['lr'] = param_group['lr'] * opt.lr_decay_factor
                if take_action:
                    logger.info(f'  New LR: {param_group["lr"]:.2e}')

        if (epoch + 1) % opt.vizIter == 0:
            if take_action:
                generate_val_samples(opt, model, val_dataset, outf_syn, epoch, device, logger)
                torch.cuda.empty_cache()
                gc.collect()
                logger.info(f'GPU cache cleared after validation sampling at epoch {epoch}')
            dist.barrier()

        if (epoch + 1) % opt.saveIter == 0:
            if take_action:
                save_dict = {'epoch': epoch,
                             'model_state': model.state_dict(),
                             'optimizer_state': optimizer.state_dict()
                             }
                torch.save(save_dict, '%s/epoch_%d.pth' % (output_dir, epoch))
            dist.barrier()


    dist.destroy_process_group()


def main():
    opt = parse_args()
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '9992'

    exp_id = os.path.splitext(os.path.basename(__file__))[0]
    dir_id = os.path.dirname(__file__)

    output_dir = get_output_dir(dir_id, exp_id)

    with open(os.path.join(output_dir,'options.txt'),'w') as f:
        json.dump(opt.__dict__, f, indent=2)

    copy_source(__file__, output_dir)

    # initiate distributed training
    opt.ngpus_per_node = torch.cuda.device_count()
    opt.world_size = opt.ngpus_per_node

    print(f'DDP World_size:{opt.world_size}')

    mp.spawn(train, nprocs=opt.ngpus_per_node, args=(opt, output_dir))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./data', help="set the path to the dataset here")

    # Data loader parameters
    parser.add_argument('--bs', type=int, required=True, help='input batch size, 28teeth/1024points 기준 1batch=1gpu')
    parser.add_argument('--workers', type=int, required=True,  help='workers dataloader, 1batch=1worker as starting point')
    parser.add_argument('--niter', type=int, required=True, help='number of epochs to train for')

    # Input point cloud
    parser.add_argument('--nc', type=int, default=3, help="dimension of one point (usually 3 for x, y,z)")
    parser.add_argument('--extra_feature_nc', type=int, default=13, help="8 for positional embedding + 1 for obs_mask + 1 for tooth_exists_mask + 3 for root_axis")

    parser.add_argument('--max_missing_teeth', type=int, default=3, help="maximum missing teeth to simulate (reduced for half-arch data)") 
    parser.add_argument('--tooth_npoints', type=int, default=1024, help="num points per toorh") 

    ''' Model '''
    # Diffusion process parameters (variance schedule, number of steps)
    parser.add_argument('--beta_start', type=float, default=0.0001)
    parser.add_argument('--beta_end', type=float, default=0.02)
    parser.add_argument('--schedule_type', type=str, default='linear')
    parser.add_argument('--time_num', type=int, default=1000, help='number of timesteps T in diffusion process')
    # parser.add_argument('--debug', action='store_true')

    # Model parameters
    parser.add_argument('--attention', type=eval, default=True)
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate (increased from 0.1 for better regularization)')
    parser.add_argument('--embed_dim', type=int, default=96, help='embedding dimension (increased from 96 for more capacity)')
    parser.add_argument('--loss_type', type=str, default='mse')
    parser.add_argument('--model_mean_type', type=str, default='eps')
    parser.add_argument('--model_var_type', type=str, default='fixedsmall')
    parser.add_argument('--vox_res_mult', type=float, default=1.0)
    parser.add_argument('--width_mult', type=float, default=0.75, help='width multiplier (increased from 0.75 for full capacity)')

    parser.add_argument('--lr', type=float, default=4e-4, help='learning rate for E, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam (0.5 is standard for diffusion models)')
    parser.add_argument('--decay', type=float, default=1e-4, help='weight decay for regularization')
    parser.add_argument('--grad_clip', type=float, default=2.0, help='gradient clipping (increased from 1.0 to 2.0 for stability)')
    parser.add_argument('--lr_gamma', type=float, default=1, help='lr decay for EBM')
    parser.add_argument('--lr_decay_factor', type=float, default=0.5, help='LR decay factor at scheduled epochs (default: 0.5 = halve LR)')

    # Model path (for continuing the training of existing models)
    parser.add_argument('--model', default='', help="path to model (to continue training)")

    parser.add_argument('--saveIter', type=int, default=50, help='unit: epoch')
    parser.add_argument('--vizIter', type=int, default=100, help='unit: epoch')
    parser.add_argument('--print_freq', type=int, default=10, help='unit: iter')

    # Manual seed for deterministic sampling, etc.
    # TODO apply manual seed for reproducibility using "set_seed" in file_utils.py
    parser.add_argument('--manualSeed', default=1234, type=int, help='random seed')

    # Parse arguments
    opt = parser.parse_args()

    return opt


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
