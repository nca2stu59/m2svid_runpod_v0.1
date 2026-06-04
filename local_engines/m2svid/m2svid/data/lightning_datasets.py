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

import pytorch_lightning as pl
import webdataset as wds

from torch.utils.data import ConcatDataset
from torch.utils.data.distributed import DistributedSampler

from m2svid.data.datasets import Ego4dDataset, Stereo4dDataset, EvalDataset


class StereoLightningDataset(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers, train_kwargs={}, eval_kwargs={}, common_kwargs={}, seed=0):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        self.train_kwargs = train_kwargs
        self.eval_kwargs = eval_kwargs
        self.common_kwargs = common_kwargs

    def setup(self):
        raise NotImplementedError

    def train_dataloader(self):
        sampler = DistributedSampler(self.train_dataset, seed=self.seed)
        return wds.WebLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, sampler=sampler)

    def val_dataloader(self):
        loader = wds.WebLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        return loader

    def test_dataloader(self):
        return wds.WebLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


class Ego4dLightningDataset(StereoLightningDataset):
    def setup(self):
        self.train_dataset = Ego4dDataset(**self.train_kwargs, **self.common_kwargs)
        self.val_dataset = Ego4dDataset(**self.eval_kwargs, **self.common_kwargs)


class Stereo4dLightningDataset(StereoLightningDataset):
    def setup(self):
        self.train_dataset = Stereo4dDataset(**self.train_kwargs, **self.common_kwargs)
        self.val_dataset = Stereo4dDataset(**self.eval_kwargs, **self.common_kwargs)


class Ego4dStereo4dLightningDataset(StereoLightningDataset):
    def setup(self):
        self.train_dataset = ConcatDataset([
            Stereo4dDataset(**self.train_kwargs['stereo4d'], **self.common_kwargs),
            Ego4dDataset(**self.train_kwargs['ego4d'], **self.common_kwargs)
        ])
        self.val_dataset = Stereo4dDataset(**self.eval_kwargs, **self.common_kwargs)


class EvalLightningDataset(StereoLightningDataset):
    def setup(self):
        self.train_dataset = EvalDataset(**self.train_kwargs, **self.common_kwargs)
        self.val_dataset = EvalDataset(**self.eval_kwargs, **self.common_kwargs)
