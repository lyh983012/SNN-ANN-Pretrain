"""
generate DVS dataset mode-1(by default)

contrast threshold (CT) is constant in each video clip spatially
while CT is dynamically adjusted in different video clips to guarantee the event ratio to be 0.15
"""

import os
import glob
import numpy as np
import cv2
from tqdm import tqdm
from configs import configs


def gen_event_from_video(folder, configs):
    image_list = glob.glob(os.path.join(configs['base_dir'], folder, '*.jpg'))
    samples = np.round(np.linspace(
        0, len(image_list) - 1, configs['num_frames'] + 1))
    image_list = [image_list[int(sample)] for sample in samples]

    img0 = cv2.imread(image_list[0])
    H, W, C = img0.shape

    images = np.zeros((len(image_list), H, W))
    for i in range(len(image_list)):
        img = cv2.imread(image_list[i], cv2.IMREAD_GRAYSCALE).astype(np.float) / 255.0
        images[i] = img

    diff = np.zeros((configs['num_frames'], H, W))
    for i in range(configs['num_frames']):
        diff[i, :, :] = images[i + 1] - images[i]

    CT1, CT2 = np.percentile(diff, [50 * configs['event_ratio'], 100 - 50 * configs['event_ratio']])

    event = np.zeros((2, configs['num_frames'], H, W), dtype=np.uint8)
    event[0, :, :, :][diff > CT2] = 1  # positive events
    event[1, :, :, :][diff < CT1] = 1   # negative events

    save_path = os.path.join(configs['output_path'], folder + '.npz')
    np.savez_compressed(save_path, event)


dir_list = []
for root, dirs, files in os.walk(configs['base_dir']):
    dir_list = dirs
    break

for folder in tqdm(dir_list):
    gen_event_from_video(folder=folder, configs=configs)


print('Generate DVS events from video successfully!')

