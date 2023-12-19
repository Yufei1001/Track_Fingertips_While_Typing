import matplotlib.pyplot as plt
import json
import cv2
import os


user_id = 'yufei081523'
camera_setting = 'Camera_Right60_Vertical20'
typing_content = 'label-yufei081523-081523_143727-37-local_handposes'
hand = 'cropped_right_hand'
begin = 1
end = 3485

# frame = cv2.imread(f'./data/{user_id}/{camera_setting}/{typing_content}/{hand}/frame_00001.png')

frame=cv2.imread('./result/zhuolinnumbs051223/Camera_Right50_Vertical20/label-zhuolinnumbs051223-051223_073615-13-local_handposes/cropped_right_hand/fourth_fingertip_guanjie/cluster/fourth_fingertip_guanjie/frame_00001.png')

height, width, layers = frame.shape
size = (width, height)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter(f'./video/{user_id}/{camera_setting}/{typing_content}/{hand}/{begin}_{end}.mp4', fourcc, 30.0, size)
out = cv2.VideoWriter(f'./result/zhuolinnumbs051223/Camera_Right50_Vertical20/label-zhuolinnumbs051223-051223_073615-13-local_handposes/cropped_right_hand/fourth.mp4', fourcc, 30.0, size)

for i in range(begin, end+1):
    # print(i)
    # img = cv2.imread(f'./data/{user_id}/{camera_setting}/{typing_content}/{hand}/frame_{i:05d}.png')
    img=cv2.imread(f'./result/zhuolinnumbs051223/Camera_Right50_Vertical20/label-zhuolinnumbs051223-051223_073615-13-local_handposes/cropped_right_hand/fourth_fingertip_guanjie/cluster/fourth_fingertip_guanjie/frame_{i:05d}.png')
    out.write(img)
out.release()
