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

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import imageio
import torch


def to_pil(image):
    image = Image.fromarray(image)
    return image


def make_anaglyph_image(left, right):
    width, height = left.size
    output_image = left.copy()
    leftMap = output_image.load()
    rightMap = right.load()
    m = [ [ 1, 0, 0, 0, 0, 0, 0, 0, 0 ], [ 0, 0, 0, 0, 1, 0, 0, 0, 1 ] ]

    for y in range(0, height):
        for x in range(0, width):
            r1, g1, b1 = leftMap[x, y]
            r2, g2, b2 = rightMap[x, y]
            leftMap[x, y] = (
                int(r1*m[0][0] + g1*m[0][1] + b1*m[0][2] + r2*m[1][0] + g2*m[1][1] + b2*m[1][2]),
                int(r1*m[0][3] + g1*m[0][4] + b1*m[0][5] + r2*m[1][3] + g2*m[1][4] + b2*m[1][5]),
                int(r1*m[0][6] + g1*m[0][7] + b1*m[0][8] + r2*m[1][6] + g2*m[1][7] + b2*m[1][8])
            )
    return output_image


def make_anaglyph_video(left_video, right_video, unnormalized_videos=False):
    if unnormalized_videos:
        device = left_video.device
        left_video = left_video.cpu().numpy().transpose(1, 2, 3, 0)
        right_video = right_video.cpu().numpy().transpose(1, 2, 3, 0)
        left_video = (((left_video + 1) / 2).clip(0, 1) * 255).astype(np.uint8)
        right_video = (((right_video + 1) / 2).clip(0, 1) * 255).astype(np.uint8)

    output_frames = []
    for left_image, right_image in zip(left_video, right_video):
        left_image = to_pil(left_image)
        right_image = to_pil(right_image)
        output_image = make_anaglyph_image(left_image, right_image)
        output_frames.append(output_image)
    output_frames = np.stack(output_frames, axis=0)

    if unnormalized_videos:
        output_frames = ((output_frames / 255.0) - 0.5) / 0.5
        output_frames = torch.from_numpy(output_frames.transpose(3, 0, 1, 2)).to(device)

    return output_frames

