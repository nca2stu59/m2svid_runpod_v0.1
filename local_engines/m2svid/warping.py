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

import argparse
from m2svid.utils.video_utils import open_ffmpeg_process, read_frames_in_batches_ffmpeg, get_video_fps
from m2svid.warping.warping import scatter_image

import numpy as np
import tqdm
from pathlib import Path
import ffmpeg
import os
import cv2


def process_video_with_depth(
    video_path,
    depth_path,
    output_path_reprojected,
    output_path_mask,
    disparity_scale=None,
    disparity_perc=None,
    batch_size=10,
):
    relative_depth_data = np.load(depth_path)
    relative_depth = relative_depth_data['depth']
    probe = ffmpeg.probe(video_path)
    video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    width = int(video_stream['width'])
    height = int(video_stream['height'])

    if disparity_perc is not None:
        disparity_scale = int(width * disparity_perc)

    fps = get_video_fps(video_path, probe)

    ffmpeg_process_reprojected = None
    ffmpeg_process_mask = None

    for i, left_frames in enumerate(
        tqdm.tqdm(
            read_frames_in_batches_ffmpeg(video_path, batch_size, width, height),
            total=int(relative_depth.shape[0] // batch_size),
        )
    ):
        depth_batch = relative_depth[i * batch_size : (i + 1) * batch_size]
        depth_batch = np.array([
            cv2.resize(depth_frame, (width, height), interpolation=cv2.INTER_CUBIC)
            for depth_frame in depth_batch
        ])

        disparities = depth_batch * disparity_scale

        reprojected_right_videos = []
        reprojected_right_masks = []

        for left_frame, disparity in zip(left_frames, disparities):
            reprojected_image, inpainting_mask, reprojected_depth = scatter_image(
                left_frame, disparity, direction=-1, scale_factor=1, reproject_depth=True
            )
            reprojected_right_videos.append(reprojected_image)
            reprojected_right_masks.append(inpainting_mask)

        reprojected_right_videos = np.stack(reprojected_right_videos, axis=0)
        reprojected_right_masks = np.stack(reprojected_right_masks, axis=0)

        if ffmpeg_process_reprojected is None:
            _, height, width, _ = reprojected_right_videos.shape
            ffmpeg_process_reprojected = open_ffmpeg_process(
                output_path_reprojected, width, height, fps
            )
            ffmpeg_process_mask = open_ffmpeg_process(
                output_path_mask, width, height, fps, grayscale=True, no_compression=True
            )

        for reprojected_frame, mask_frame in zip(
            reprojected_right_videos, reprojected_right_masks
        ):
            ffmpeg_process_reprojected.stdin.write(reprojected_frame.tobytes())
            ffmpeg_process_mask.stdin.write(mask_frame.tobytes())

    ffmpeg_process_reprojected.stdin.close()
    ffmpeg_process_reprojected.wait()
    ffmpeg_process_mask.stdin.close()
    ffmpeg_process_mask.wait()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video frames with depth data to generate reprojected videos and masks.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--depth_path", type=str, required=True, help="Path to the depth numpy file.")
    parser.add_argument("--output_path_reprojected", type=str, required=True, help="Path to save the reprojected output video.")
    parser.add_argument("--output_path_mask", type=str, required=True, help="Path to save the mask output video.")
    parser.add_argument("--disparity_scale", type=float, default=None, help="List of disparity scales to apply.")
    parser.add_argument("--disparity_perc", type=float, default=None, help="List of disparity scales to apply.")

    args = parser.parse_args()

    assert (args.disparity_scale is None) or (args.disparity_perc is None)
    assert (args.disparity_scale is not None) or (args.disparity_perc is not None)

    os.makedirs(os.path.dirname(args.output_path_reprojected), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_path_mask), exist_ok=True)

    process_video_with_depth(
        args.video_path,
        args.depth_path,
        args.output_path_reprojected,
        args.output_path_mask,
        disparity_scale=args.disparity_scale,
        disparity_perc=args.disparity_perc,
    )
