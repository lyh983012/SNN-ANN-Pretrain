# visualize the generated events
import cv2
import numpy as np

save_path = r'./events_dvs/v_ApplyEyeMakeup_g01_c01.npz'
H, W, C = (256, 342, 3)

npz_file_event = np.load(save_path)
event = npz_file_event['arr_0']
# print(event.shape)   # (2, 30, 256, 342)

# visualization
e_rgb = np.zeros((H, W, 3), dtype=np.uint8) * 255
e_rgb[event[0, 0, :, :] == 1] = (0, 0, 255)
e_rgb[event[1, 0, :, :] == 1] = (255, 0, 0)

cv2.imshow('visual', e_rgb)
cv2.waitKey()
