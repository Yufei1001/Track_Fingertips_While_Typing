import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn.functional as F
import cv2
from pathlib import Path
from fire import Fire
import os

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicks.append([x, y])
        print(f"Clicked at (x, y): ({x}, {y})")

def main(user_id, camera_setting, typing_content, hand, img_no, image_size=(512,896), timestride=1):
    frame_root = f'./data/{user_id}/{camera_setting}/{typing_content}/{hand}'
    print(frame_root)
    hand_image_files = [os.path.join(frame_root, f) for f in sorted(os.listdir(frame_root)) if f.endswith('00001.png')]
    hand_images = []

    for hand_file_path in hand_image_files:
        img = cv2.imread(hand_file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hand_images.append(img)
    rgbs = np.array(hand_images) 
    rgb_seq = rgbs
    rgb_seq = torch.from_numpy(rgb_seq).permute(0,3,1,2).to(torch.float32) # S,3,H,W
    rgb_seq = F.interpolate(rgb_seq, image_size, mode='bilinear').unsqueeze(0) # 1,S,3,H,W
    print(rgb_seq.shape)
    # 从rgb_seq中选择一个图像显示，这里选择第一个图像
    image_to_display = rgb_seq[0,img_no].permute(1, 2, 0).cpu().numpy()

    
    print("image", image_to_display.shape)
    image_data = np.clip(image_to_display, 0, 255).astype(np.uint8)

    cv2.namedWindow('Image')
    cv2.imshow('Image', image_data)
    cv2.setMouseCallback('Image', mouse_callback)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    clicks = []
    Fire(main) 
    print([clicks])

# python .\initial_fingertips.py --user_id zhuolinnumbs051223 --camera_setting Camera_Right50_Vertical20 --typing_content label-zhuolinnumbs051223-051223_072851-4-local_handposes --hand cropped_right_hand --img_no 0
# img_no frame-1


