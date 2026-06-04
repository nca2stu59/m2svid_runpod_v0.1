"""
Copyright 2026 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

from einops import rearrange, repeat

from sgm.util import append_dims, instantiate_from_config
from sgm.modules.encoders.modules import GeneralConditioner
from sgm.modules.diffusionmodules.loss import StandardDiffusionLoss
from sgm.modules.diffusionmodules.denoiser import Denoiser
from sgm.modules.autoencoding.lpips.loss.lpips import LPIPS
from sgm.modules.diffusionmodules.discretizer import Discretization
from sgm.modules.encoders.modules import GeneralConditioner
from sgm.util import append_dims, instantiate_from_config
from sgm.modules.autoencoding.temporal_ae import VideoDecoder


class OneStepSampling:
    def __init__(self, sigma=700.0):
        self.sigma = sigma

    def __call__(self, n_samples, rand=None):
        return self.sigma * torch.ones((n_samples,))


class OneStepLoss(nn.Module):
    def __init__(
        self,
        sigma_sampler_config: dict,
        loss_weighting_config: dict,
        loss_type: str = None,
        image_loss_type: str = None,

        loss_types: Optional[List[str]] = None,
        loss_weights: Optional[List[float]] = None,
        image_loss_types: Optional[List[str]] = None,
        image_loss_weights: Optional[List[float]] = None,

        batch2model_keys: Optional[Union[str, List[str]]] = None,
        num_frames: int = 1,
        focal_loss_gamma = None,
        focal_loss_correction = None,
    ):
        super().__init__()
        assert loss_type in [None, "l2", "l1", "lpips"]
        assert image_loss_type in [None, "l2", "l1", "lpips"]
        self.sigma_sampler = instantiate_from_config(sigma_sampler_config)
        self.loss_weighting = instantiate_from_config(loss_weighting_config)

        if loss_types is None:
            if loss_type is not None:
                loss_types = [loss_type]
                loss_weights = [1.0]
            else:
                loss_types = ["l2"]
                loss_weights = [1.0]

        if image_loss_types is None:
            if image_loss_type is not None:
                image_loss_types = [image_loss_type]
                image_loss_weights = [1.0]
            else:
                image_loss_types = None
                image_loss_weights = None

        self.loss_types = loss_types
        self.loss_weights = loss_weights if loss_weights is not None else [1.0] * len(loss_types)

        self.image_loss_types = image_loss_types
        self.image_loss_weights = image_loss_weights if image_loss_weights is not None else [1.0] * len(image_loss_types) if image_loss_types is not None else None

        if len(self.loss_types)!= len(self.loss_weights):
            raise ValueError("Number of losses and weights must be equal.")

        if self.image_loss_types is not None and len(self.image_loss_types)!= len(self.image_loss_weights):
            raise ValueError("Number of image losses and weights must be equal.")


        if "lpips" in self.loss_types or (self.image_loss_types is not None and "lpips" in self.image_loss_types):
            self.lpips = LPIPS().eval()

        if "ssim" in self.loss_types or (self.image_loss_types is not None and "ssim" in self.image_loss_types):
            from piqa import SSIM
            self.ssim = SSIM().cuda().eval()


        if not batch2model_keys:
            batch2model_keys = []

        if isinstance(batch2model_keys, str):
            batch2model_keys = [batch2model_keys]

        self.batch2model_keys = set(batch2model_keys)

        self.num_frames = num_frames
        self.focal_loss_gamma = focal_loss_gamma
        self.focal_loss_correction = focal_loss_correction

    def forward(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        conditioner: GeneralConditioner,
        input: torch.Tensor,
        batch: Dict,
        apply_first_stage=False,
        first_stage_model=None,
        scale_factor=None,
        disable_first_stage_autocast=False,
        en_and_decode_n_samples_a_time=None,
    ):
        if "num_video_frames" in batch:
            num_video_frames = batch['num_video_frames']
        else:
            num_video_frames = self.num_frames

        if self.image_loss_types is not None:
            assert apply_first_stage

        if apply_first_stage:
            x = first_stage_model.encode(input)
            x = scale_factor * x
        else:
            x = input

        cond = conditioner(batch)

        additional_model_inputs = {
            key: batch[key] for key in self.batch2model_keys.intersection(batch)
        }
        b = x.shape[0] // num_video_frames
        sigmas = self.sigma_sampler(b).to(x)
        sigmas = repeat(sigmas, "b -> (b t)", t=num_video_frames)

        noise = torch.zeros_like(x)
        z = denoiser(
            network, noise, sigmas, cond, **additional_model_inputs,
        )
        w = self.loss_weighting(sigmas)
        loss_dict = {}

        loss_latent = 0.0
        for loss_type, weight in zip(self.loss_types, self.loss_weights):
            cur_loss = self.get_loss(z, x, w, loss_type)
            loss_dict[f'loss_latent_{loss_type}'] = cur_loss.mean()
            loss_dict[f'loss_latent_{loss_type}_weighted'] = weight * cur_loss.mean()
            loss_latent = loss_latent + weight * cur_loss

        if apply_first_stage:
            z = 1.0 / scale_factor * z
            if isinstance(first_stage_model.decoder, VideoDecoder):
                model_output = first_stage_model.decode(z, timesteps=num_video_frames)
            else:
                model_output = first_stage_model.decode(z)

            loss_image = 0.0
            for loss_type, weight in zip(self.image_loss_types, self.image_loss_weights):
                cur_loss = self.get_loss(model_output, input, w, loss_type)
                loss_dict[f'loss_image_{loss_type}'] = cur_loss.mean()
                loss_dict[f'loss_image_{loss_type}_weighted'] = weight * cur_loss.mean()
                loss_image = loss_image + weight * cur_loss

            final_loss = loss_latent + loss_image
            final_loss = final_loss.reshape(-1, num_video_frames)
            final_loss = torch.mean(final_loss, 1)
            if self.focal_loss_gamma:
                focal_loss_weight = (1 + final_loss) ** self.focal_loss_gamma
                if self.focal_loss_correction:
                    focal_loss_weight = focal_loss_weight / (((1 + self.focal_loss_correction) ** self.focal_loss_gamma))
            return loss_latent + loss_image, loss_dict
        else:
            final_loss = loss_latent
            final_loss = final_loss.reshape(-1, num_video_frames)
            final_loss = torch.mean(final_loss, 1)
            return loss_latent, loss_dict

    def get_loss(self, model_output, target, w, loss_type):
        if loss_type is None:
            return 0
        elif loss_type == "l2":
            loss = (model_output - target) ** 2
            return w * torch.mean(loss.reshape(target.shape[0], -1), 1)
        elif loss_type == "l1":
            loss = (model_output - target).abs()
            return w * torch.mean(loss.reshape(target.shape[0], -1), 1)
        elif loss_type == "lpips":
            return self.lpips(model_output, target).reshape(-1)
        elif loss_type == "ssim":
            def rescale_to_0_1(x):
                return torch.clamp((x + 1.0) / 2.0, min=0, max=1)

            model_output_rescaled = rescale_to_0_1(model_output)
            target_rescaled = rescale_to_0_1(target)

            loss = 1 - self.ssim(model_output_rescaled, target_rescaled).reshape(-1)
            return loss

        else:
            raise NotImplementedError(f"Unknown loss type {loss_type}")
