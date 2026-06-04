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
import re
import io
import torch
import torchvision
import tqdm
from PIL import Image
from torchvision import transforms
import numpy as np
import json
import torch as th
import ffmpeg
import cv2
import random
from torch.utils.data.distributed import DistributedSampler

from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode


def make_spatial_transformations(resolution, spatial_transform_type, resize_scale=(0.1, 1)):
    """
    resolution: target resolution, a list of int, [h, w]
    """

    if spatial_transform_type == "resize_random_crop":
        resize = MyResize(max(resolution), antialias=True)
        other_transformations = transforms.RandomCrop(resolution)
    elif spatial_transform_type == "resize_center_crop":
        resize = MyResize(max(resolution), antialias=True)
        other_transformations = transforms.CenterCrop(resolution)
    elif spatial_transform_type == "random_crop":
        resize = NoResize()
        other_transformations = transforms.RandomCrop(resolution)
    elif spatial_transform_type == "center_crop":
        resize = NoResize()
        other_transformations = transforms.CenterCrop(resolution)
    elif spatial_transform_type == "resize":
        resize = MyResize(max(resolution), antialias=True)
        other_transformations = CenterCropDivisable(64)
    elif spatial_transform_type == "resize_if_needed":
        resize = MyResizeIfNeeded(max(resolution), antialias=True)
        other_transformations = CenterCropDivisable(64)
    elif spatial_transform_type == "maxresize_if_needed":
        resize = MaxResizeIfNeeded(max(resolution), antialias=True)
        other_transformations = CenterCropDivisable(64)
    elif spatial_transform_type == "random_resize_crop":
        resize = transforms.Compose([
            RandomSquareCrop(),
            MyRandomResizedCrop(max(resolution), scale=resize_scale)]
        )
        other_transformations = transforms.CenterCrop(resolution)
    elif spatial_transform_type == "no_transform":
        resize = NoResize()
        other_transformations = transforms.Lambda(lambda x: x)
    else:
        raise NotImplementedError
    return resize, other_transformations


def preprocess_frames(list_of_videos, spatial_transform_type, resolution, resize_scale):
    if spatial_transform_type is not None:
        resize, other_transforms = make_spatial_transformations(resolution, spatial_transform_type, resize_scale=resize_scale)
        all_frames = torch.cat(list_of_videos, dim=0)
        scale_factor, all_frames = resize(all_frames)
        all_frames = other_transforms(all_frames)

        new_list_of_videos = []
        offset = 0
        for video in list_of_videos:
            new_list_of_videos.append(all_frames[offset:offset + len(video)])
            offset += len(video)
        list_of_videos = new_list_of_videos
    else:
        scale_factor = 1

    new_list_of_videos = []
    for video in list_of_videos:
        video = video.permute(1, 0, 2, 3).float() # [t,c,h,w] -> [c,t,h,w]
        video = video * 2 - 1
        new_list_of_videos.append(video)
    list_of_videos = new_list_of_videos

    return scale_factor, list_of_videos


def format_output(video_id, left_frames, right_frames, reprojected_frames=None, masks=None, test_masks=None, resolution=None,
                 left_condition_view=False, return_reprojected=False, mask_antialias=True):
    output = {
        'video_id': video_id,
        'video': right_frames if left_condition_view else left_frames,
        'video_2nd_view': left_frames if left_condition_view else right_frames,
        'caption': ""
    }

    if return_reprojected:
        c,t,h,w = masks.shape
        downsampled_resolution = [int(h / 8), int(w / 8)]

        original_masks = masks
        masks = masks.permute(1, 0, 2, 3).float() # [c,t,h,w] -> [t,c,h,w]
        masks = transforms.Resize(downsampled_resolution, antialias=mask_antialias)(masks)
        masks = masks[:, [0]]
        masks = masks.permute(1, 0, 2, 3).float() # [t,c,h,w] -> [c,t,h,w]

        output.update({
            'reprojected_video': reprojected_frames,
            'reprojected_mask': masks,
            'original_reprojected_mask': original_masks,
        })

    if test_masks is not None:
        output.update({'inpainting_mask_for_testing': test_masks})

    output.update({"fps_id": 7, "motion_bucket_id": 127, 'elevation': 0})

    return output


def get_video_frames(video_path, fps=None, num_frames=None, width=None, height=None, duration=None,
                     sample_beginning=False, normalize=True, start=0, video_is_grayscale=False,
                     raise_error_fewer_frames=False):
    try:
        if (width is None) or (height is None) or (duration is None):
            probe = ffmpeg.probe(video_path)

        if duration is None:
            duration = float(probe['format']['duration'])
            if duration == 0:
                duration = num_frames

        if width is None:
            width = int(probe['streams'][0]['width'])

        if height is None:
            height = int(probe['streams'][0]['height'])

    except Exception as excep:
        print("Warning: ffmpeg error. video path: {} error. Error: {}".format(video_path, excep), flush=True)

    if video_is_grayscale:
        pix_fmt = 'gray'
        channels = 1
    else:
        pix_fmt = 'rgb24'
        channels = 3

    try:
        if fps is not None and num_frames is not None:
            num_sec = num_frames / fps
            if sample_beginning:
                start = np.random.random() * (duration - num_sec)
        else:
            num_sec = None
            assert sample_beginning == False

        if num_sec is None:
            cmd = ffmpeg.input(video_path, ss=start)
        else:
            cmd = ffmpeg.input(video_path, ss=start, t=num_sec + 0.1)

        if fps is not None:
            cmd = cmd.filter('fps', fps=fps)

        out, _ = (
            cmd.output('pipe:', format='rawvideo', pix_fmt=pix_fmt)
                .run(capture_stdout=True, quiet=True)
        )

        video = np.frombuffer(out, np.uint8).reshape([-1, height, width, channels])
        video = th.tensor(video)
        video = video.permute(0, 3, 1, 2)
        if num_frames is not None:
            if video.shape[0] < num_frames:
                if raise_error_fewer_frames:
                    raise ValueError(f"less frames than necessary:  {video.shape[0]} out of {num_frames} (fps={fps})")
                zeros = th.zeros((num_frames - video.shape[0], channels, height, width), dtype=th.uint8)
                video = th.cat((video, zeros), axis=0)
            elif video.shape[0] > num_frames:
                video = video[:num_frames]
    except Exception as excep:
        print("Warning: ffmpeg error. video path: {} error. Error: {}".format(video_path, excep), flush=True)
        raise excep
    if normalize:
        video = video.float() / 255.

    return video


class MyRandomResizedCrop(transforms.RandomResizedCrop):

    def __init__(
        self,
        size,
        scale=(0.08, 1.0),
    ):
        super().__init__(size, scale, ratio=(1, 1), interpolation=InterpolationMode.BILINEAR, antialias=True)

    def forward(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        assert h == w, f"new_height, new_width: {h}, {w}"
        scale = self.size[0] / h

        return scale, F.resized_crop(img, i, j, h, w, self.size, self.interpolation, antialias=self.antialias)


class MyResize(transforms.Resize):
    def forward(self, img):
        _, height, width = F.get_dimensions(img)
        input_size = min(height, width)
        output_size = self.size
        scale = output_size / input_size

        return scale, super().forward(img)


class MyResizeIfNeeded(transforms.Resize):
    def forward(self, img):
        _, height, width = F.get_dimensions(img)
        input_size = min(height, width)
        if input_size > self.size:
            output_size = self.size
            scale = output_size / input_size

            return scale, super().forward(img)
        else:
            return 1, img

class MaxResizeIfNeeded(transforms.Resize):
    def forward(self, img):
        _, height, width = F.get_dimensions(img)
        max_side = max(height, width)
        if max_side > self.size:
            scale = self.size / max_side
            new_height = int(round(height * scale))
            new_width = int(round(width * scale))
            resized_img = F.resize(img, (new_height, new_width), antialias=self.antialias)
            return scale, resized_img
        else:
            return 1, img

class NoResize():
    def __call__(self, img):
        return 1, img


class RandomSquareCrop:
    def __call__(self, img):
        _, height, width = F.get_dimensions(img)
        crop_size = min(height, width)
        if height == crop_size and width == crop_size:
            return img
        i = torch.randint(0, height - crop_size + 1, size=(1,)).item()
        j = torch.randint(0, width - crop_size + 1, size=(1,)).item()
        return F.crop(img, i, j, crop_size, crop_size)


class Identity():
    def __call__(self, img):
        return img


class CenterCropDivisable(torch.nn.Module):
    def __init__(self, divisable):
        super().__init__()
        self.divisable = divisable

    def forward(self, img):
        _, height, width = F.get_dimensions(img)
        height = int(np.floor(height / self.divisable) * self.divisable)
        width = int(np.floor(width / self.divisable) * self.divisable)
        return F.center_crop(img, (height, width))


def select_frames(list_of_pathlists, frame_stride, frame_number, random_offset=False,
                pre_read_frames=None):

    pathlist = list_of_pathlists[0]
    # like list of left frames, or list of right frames

    if random_offset:
        offset = np.random.randint(0, max(1, len(pathlist) - frame_stride * (frame_number - 1)))
    else:
        offset = 0

    list_of_pathlists = [pathlist[offset::frame_stride][:frame_number] for pathlist in list_of_pathlists]
    pathlist = list_of_pathlists[0]

    if len(list_of_pathlists[0]) != frame_number:
        list_of_pathlists = [pathlist + [pathlist[-1]] * (len(pathlist) - frame_number) for pathlist in list_of_pathlists]

    def read_image(path, pre_read_frames):
        try:
            if (pre_read_frames is not None) and (path in pre_read_frames):
                # image = pre_read_frames[path]
                image = Image.open(io.BytesIO(pre_read_frames[path]))
            else:
                image = Image.open(path)
            image = image.convert('RGB')
            image = transforms.ToTensor()(image)
        except:
            print(f"Cannot read image: {path}", flush=True)
            image = None
        return image

    list_of_framelist = []
    for pathlist in list_of_pathlists:
        list_of_framelist.append([read_image(path, pre_read_frames) for path in pathlist])

    list_of_framelist_final = [list() for _ in range(len(list_of_framelist))]
    for frames in zip(*list_of_framelist):
        if all((frame is not None) for frame in frames):
            for curlist, cur_frame in zip(list_of_framelist_final, frames):
                curlist.append(cur_frame)

    return tuple([torch.stack(framelist) for framelist in list_of_framelist_final])





def apply_closing(tensor, kernel):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel))
    for frame in range(tensor.shape[0]):
        img = tensor[frame, 0].numpy()
        closed_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        tensor[frame, 0] = torch.from_numpy(closed_img)
    tensor = (tensor > 0.5).to(tensor.dtype)
    return tensor


def apply_dilation(tensor, kernel_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    for frame in range(tensor.shape[0]):
        img = tensor[frame, 0].numpy()
        dilated_img = cv2.dilate(img, kernel)
        tensor[frame, 0] = torch.from_numpy(dilated_img)
    tensor = (tensor > 0.5).to(tensor.dtype)
    return tensor


def generate_random_shape(H, W):
    """Generates a random occlusion shape."""
    mask_shape = np.zeros((H, W), dtype=np.uint8)

    shape_type = random.choice(["circle", "rectangle", "ellipse", "polygon", "squiggle"])

    if shape_type == "circle":
        x_start, y_start = random.randint(W // 10, W - W // 10), random.randint(H // 10, H - H // 10)
        radius = random.randint(min(W, H) // 20, min(W, H) // 5)
        cv2.circle(mask_shape, (x_start, y_start), radius, 1, -1)

    elif shape_type == "rectangle":
        x_start, y_start = random.randint(0, W - W // 5), random.randint(0, H - H // 5)
        width, height = random.randint(W // 20, W // 5), random.randint(H // 20, H // 5)
        cv2.rectangle(mask_shape, (x_start, y_start), (x_start + width, y_start + height), 1, -1)

    elif shape_type == "ellipse":
        x_start, y_start = random.randint(W // 10, W - W // 10), random.randint(H // 10, H - H // 10)
        axes = (random.randint(W // 20, W // 5), random.randint(H // 20, H // 5))
        angle = random.randint(0, 180)
        cv2.ellipse(mask_shape, (x_start, y_start), axes, angle, 0, 360, 1, -1)

    elif shape_type == "polygon":
        num_points = random.randint(3, 6)
        points = np.array([
            [random.randint(W // 10, W - W // 10), random.randint(H // 10, H - H // 10)]
            for _ in range(num_points)
        ])
        cv2.fillPoly(mask_shape, [points], 1)

    elif shape_type == "squiggle":
        num_points = random.randint(5, 10)
        points = np.array([
            [random.randint(W // 10, W - W // 10), random.randint(H // 10, H - H // 10)]
            for _ in range(num_points)
        ])
        for i in range(len(points) - 1):
            cv2.line(mask_shape, tuple(points[i]), tuple(points[i + 1]), 1, thickness=random.randint(H // 100, H // 50))


    return mask_shape


def mask_random_frames(mask, max_masked_percentage=0.1):
    T, H, W = mask.shape
    total_pixels = H * W * T
    max_masked_pixels = int(max_masked_percentage * total_pixels)

    frame_indices = list(range(T))
    random.shuffle(frame_indices)

    masked_pixels = mask.sum().item()

    for frame_idx in frame_indices:
        if masked_pixels + H * W > max_masked_pixels:
            break
        mask[frame_idx, :, :] = 1
        masked_pixels += H * W

    mask = (mask > 0).to(mask.dtype)
    return mask


def mask_tubes(mask, max_masked_percentage=0.1):
    T, H, W = mask.shape
    total_pixels = H * W * T
    max_masked_pixels = int(max_masked_percentage * total_pixels)

    masked_pixels = mask.sum().item()
    num_tubes = random.randint(2, 6)

    for _ in range(num_tubes):
        shape_mask = generate_random_shape(H, W)

        if masked_pixels + np.sum(shape_mask) * T > max_masked_pixels:
            break
        for t in range(T):
            mask[t] += torch.from_numpy(shape_mask)
        masked_pixels += np.sum(shape_mask) * T
    mask = (mask > 0).to(mask.dtype)
    return mask
