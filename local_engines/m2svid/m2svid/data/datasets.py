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

import os
import torch
from torchvision import transforms
import numpy as np
import json

import ffmpeg
import random

from m2svid.data.utils import preprocess_frames, format_output, get_video_frames, apply_closing, apply_dilation, mask_random_frames, mask_tubes
from m2svid.utils.video_utils import get_video_fps


class GeneralVideoDataset(torch.utils.data.Dataset):
    def __init__(self,
                    data_root,
                    split,
                    metadata_path=None,
                    fps=6,
                    frame_number=1,
                    spatial_transform_type=None,
                    resolution=[512, 512],
                    left_condition_view=True,

                    reprojected_root=None,
                    mask_root=None,
                    return_reprojected=False,

                    ensure_correct=False,
                    fake_dataset_size=None,
                    use_random_index=False,
                    dataset_name='GeneralVideoDataset',
                    baseline=0,
                    focal_length=0,
                    scale=None,

                    resize_scale=(0.1, 1),

                    apply_reprojected_mask_augmentation=False,
                    max_masked_percentage=0.1,

                    reprojected_closing_holes_kernel=None,

                    raise_error_fewer_frames=False,
                    mask_antialias=True
                       ):
        super().__init__()

        assert split in ['test', 'train', 'val']
        self.data_root = data_root
        self.split = split
        self.frame_number = frame_number
        self.resolution = resolution
        self.spatial_transform_type = spatial_transform_type
        self.left_condition_view = left_condition_view
        self.reprojected_root = reprojected_root
        self.mask_root = mask_root
        self.return_reprojected = return_reprojected
        self.ensure_correct = ensure_correct
        self.fake_dataset_size = fake_dataset_size
        self.use_random_index = use_random_index
        self.dataset_name = dataset_name
        self.baseline = baseline
        self.focal_length = focal_length
        self.scale = scale
        self.fps = fps
        self.resize_scale = resize_scale

        self.reprojected_closing_holes_kernel = reprojected_closing_holes_kernel

        self.apply_reprojected_mask_augmentation = apply_reprojected_mask_augmentation
        self.max_masked_percentage = max_masked_percentage
        self.raise_error_fewer_frames = raise_error_fewer_frames
        self.mask_antialias = mask_antialias

        if not isinstance(self.fps, int) and self.fps is not None:
            assert len(self.fps) == 2
            assert isinstance(self.fps[0], int)
            assert isinstance(self.fps[1], int)

        if return_reprojected:
            assert self.left_condition_view

        if metadata_path is not None:
            with open(metadata_path) as fin:
                self.video_names = json.load(fin)
        else:
            video_names = os.listdir(data_root)
            self.video_names = [name for name in video_names if '.mp4' in name]
            self.video_names = list(sorted(self.video_names))

        if self.return_reprojected:
            assert reprojected_root is not None
            assert mask_root is not None

    def __len__(self):
        return len(self.video_names) if self.fake_dataset_size is None else self.fake_dataset_size

    def _read_left_right_videos(self, video_id, frame_number, fps, random_start):
        raise NotImplementedError
        # return left_videos, right_videos, start

    def _get_reprojected_path(self, video_id):
        raise NotImplementedError

    def _get_reprojected_mask_path(self, video_id):
        raise NotImplementedError

    def __getitem__(self, index):
        if self.use_random_index:
            index = np.random.randint(len(self.video_names))

        index = index % len(self.video_names)

        video_id = self.video_names[index]

        if not isinstance(self.fps, int) and self.fps is not None:
            fps = np.random.randint(self.fps[0], self.fps[1] + 1)
        else:
            fps = self.fps

        if not isinstance(self.frame_number, int) and self.frame_number is not None:
            frame_number_id = np.random.randint(len(self.frame_number))
            frame_number = self.frame_number[frame_number_id]
            resolution = self.resolution[frame_number_id]
        else:
            frame_number = self.frame_number
            resolution = self.resolution

        random_start = (self.split == 'train')

        new_index = np.random.randint(len(self)) # if this doesn't work

        try:
            left_videos, right_videos, start, fps  = self._read_left_right_videos(video_id, frame_number, fps, random_start)
        except Exception as excep:
            if self.ensure_correct:
                print(f"Error: {excep}. Cannot load index: {index} ({video_id}), trying to return {new_index}", flush=True)
                return self[new_index]
            else:
                raise excep

        list_of_videos = [left_videos, right_videos]

        if self.return_reprojected:
            try:
                reprojected_path = self._get_reprojected_path(video_id)
                reprojected = get_video_frames(reprojected_path, fps=fps, num_frames=frame_number, start=start, raise_error_fewer_frames=self.raise_error_fewer_frames)

                mask_path = self._get_reprojected_mask_path(video_id)
                masks = get_video_frames(mask_path, fps=fps, num_frames=frame_number, start=start, video_is_grayscale=True, raise_error_fewer_frames=self.raise_error_fewer_frames)

                test_closing_holes_kernel = 11
                test_masks = apply_closing(masks, test_closing_holes_kernel)
                test_masks = test_masks.repeat(1, 3, 1, 1)

                if self.reprojected_closing_holes_kernel is not None:
                    masks = apply_closing(masks, self.reprojected_closing_holes_kernel)
                    reprojected[masks.repeat(1, 3, 1, 1) > 0.5] = 0
                    # make dilating since after resizing in future, you'll get more black pixels
                    masks = apply_dilation(masks, 3)

                if self.apply_reprojected_mask_augmentation:
                    if random.random() < 0.5:
                        masks[:, 0] = mask_random_frames(masks[:, 0], max_masked_percentage=self.max_masked_percentage)
                    else:
                        masks[:, 0] = mask_tubes(masks[:, 0], max_masked_percentage=self.max_masked_percentage)
                    reprojected[masks.repeat(1, 3, 1, 1) > 0.5] = 0

                masks = masks.repeat(1, 3, 1, 1)

                list_of_videos.append(reprojected)
                list_of_videos.append(masks)
                list_of_videos.append(test_masks)
            except Exception as excep:
                if self.ensure_correct:
                    print(f"Error: {excep}. Cannot load index: {index} ({video_id}), trying to return {new_index}", flush=True)
                    return self[new_index]
                else:
                    raise excep

        # transform, [t,c,h,w] -> [c,t,h,w], [0, 1] --> [-1, 1]
        scale_factor, list_of_videos = preprocess_frames(list_of_videos, self.spatial_transform_type, resolution, self.resize_scale)

        if self.return_reprojected:
            left_frames, right_frames, reprojected_frames, masks, test_masks = list_of_videos
        else:
            left_frames, right_frames = list_of_videos
            reprojected_frames, masks = None, None

        output = format_output(video_id, left_frames, right_frames, reprojected_frames, masks, test_masks,
                                resolution=resolution, left_condition_view=self.left_condition_view, return_reprojected=self.return_reprojected,
                                mask_antialias=self.mask_antialias)

        output['fps_id'] = fps
        output['baseline'] = self.baseline
        output['focal_length'] = scale_factor * self.focal_length
        if self.scale is None:
            output['scale'] = output['baseline'] * output['focal_length']
        else:
            output['scale'] = self.scale
        return output


class SBS_VideoDataset(GeneralVideoDataset):
    def _preprocess_to_left_right(self, video_id, frames):
        raise NotImplementedError

    def _get_reprojected_path(self, video_id):
        return os.path.join(self.reprojected_root, video_id)

    def _get_reprojected_mask_path(self, video_id):
        return os.path.join(self.mask_root, video_id)

    def _read_left_right_videos(self, video_id, frame_number, fps, random_start):
        video_path = os.path.join(self.data_root, video_id + '')
        probe = ffmpeg.probe(video_path)
        duration = float(probe['format']['duration'])
        if duration == 0:
            duration = 5
        width = int(probe['streams'][0]['width'])
        height = int(probe['streams'][0]['height'])

        num_sec = frame_number / fps
        if random_start:
            start = np.random.random() * (duration - num_sec)
        else:
            start = 0
        frames = get_video_frames(video_path, fps=fps, num_frames=frame_number, width=width, height=height, duration=duration, start=start,
                                 raise_error_fewer_frames=self.raise_error_fewer_frames)

        left_videos, right_videos = self._preprocess_to_left_right(video_id, frames)

        return left_videos, right_videos, start, fps


class Ego4dDataset(SBS_VideoDataset):
    def __init__(self,
                    rectified=False,
                    **kwargs):
        super().__init__(**kwargs, dataset_name='Ego4dDataset')
        self.rectified = rectified

    def _preprocess_to_left_right(self, video_id, frames):
        if self.rectified:
            _, _, h, w = frames.shape
            size = int(w // 2)
            left_videos = frames[:, :, :, :size]
            right_videos = frames[:, :, :, size:]
        else:
            _, _, h, w = frames.shape
            if h > w:
                transform = transforms.Resize([w, h], antialias=True)
                frames = transform(frames)

            _, _, h, w = frames.shape
            size = int(w // 2)
            left_videos = frames[:, :, :, :size]
            right_videos = frames[:, :, :, size:]
        return left_videos, right_videos


class Stereo4dDataset(GeneralVideoDataset):
    def _get_reprojected_path(self, video_id):
        return os.path.join(self.reprojected_root, video_id + '.mp4')

    def _get_reprojected_mask_path(self, video_id):
        return os.path.join(self.mask_root, video_id + '.mp4')

    def _read_left_right_videos(self, video_id, frame_number, fps, random_start):
        left_video_path = os.path.join(self.data_root, 'left_rectified', f'{video_id}-left_rectified.mp4')
        rihgt_video_path = os.path.join(self.data_root, 'right_rectified', f'{video_id}-right_rectified.mp4')
        probe = ffmpeg.probe(left_video_path)
        duration = float(probe['format']['duration'])
        if duration == 0:
            duration = 5
        width = int(probe['streams'][0]['width'])
        height = int(probe['streams'][0]['height'])

        num_sec = frame_number / fps
        if random_start:
            start = np.random.random() * (duration - num_sec)
        else:
            start = 0
        left_videos = get_video_frames(left_video_path, fps=fps, num_frames=frame_number, width=width, height=height, duration=duration, start=start,
                                       raise_error_fewer_frames=self.raise_error_fewer_frames)
        right_videos = get_video_frames(rihgt_video_path, fps=fps, num_frames=frame_number, width=width, height=height, duration=duration, start=start,
                                       raise_error_fewer_frames=self.raise_error_fewer_frames)

        return left_videos, right_videos, start, fps


class EvalDataset(GeneralVideoDataset):
    def __init__(self,
                    dataset_subfolder=None,
                    reprojected_subfolder=None,
                    **kwargs):
        assert dataset_subfolder is not None
        assert reprojected_subfolder is not None

        data_root = os.path.join(kwargs.pop("data_root"), dataset_subfolder)
        reprojected_root = os.path.join(kwargs.pop("reprojected_root"), reprojected_subfolder, dataset_subfolder)
        mask_root = os.path.join(kwargs.pop("mask_root"), reprojected_subfolder, dataset_subfolder)

        super().__init__(**kwargs, data_root=data_root, reprojected_root=reprojected_root, mask_root=mask_root,
                            dataset_name=dataset_subfolder)


    def _get_reprojected_path(self, video_id):
        return os.path.join(self.reprojected_root, os.path.splitext(video_id)[0] + '_reprojected.mp4')

    def _get_reprojected_mask_path(self, video_id):
        return os.path.join(self.reprojected_root, os.path.splitext(video_id)[0] + '_reprojected_mask.mp4')

    def _read_left_right_videos(self, video_id, frame_number, fps, random_start):
        video_path = os.path.join(self.data_root, video_id)

        probe = ffmpeg.probe(video_path)
        duration = float(probe['format']['duration'])
        if duration == 0:
            duration = 5
        width = int(probe['streams'][0]['width'])
        height = int(probe['streams'][0]['height'])

        assert random_start == False
        start = 0

        frames = get_video_frames(video_path, fps=fps, num_frames=frame_number, width=width, height=height, duration=duration, start=start,
                                  raise_error_fewer_frames=self.raise_error_fewer_frames)
        if fps is None:
            fps = get_video_fps(video_path, probe)

        return frames, frames, start, fps

