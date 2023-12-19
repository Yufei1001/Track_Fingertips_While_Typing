import time
import numpy as np
import saverloader
from nets.pips2 import Pips
import utils.improc
from utils.basic import print_, print_stats
import torch
import torch.nn.functional as F
from fire import Fire
import sys
import cv2
from pathlib import Path
import os
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import math

def calculate_centroid_mean(points):
    total_points = len(points)
    sum_x = sum(point[0] for point in points)
    sum_y = sum(point[1] for point in points)

    centroid_x = sum_x / total_points
    centroid_y = sum_y / total_points

    return [round(centroid_x), round(centroid_y)]

def polar_to_cartesian(radius, angle_degrees):
    angle_radians = math.radians(angle_degrees)
    x = radius * math.cos(angle_radians)
    y = radius * math.sin(angle_radians)
    return round(x), round(y)

def generate_points(center, radius, num_points):
    points = []
    for i in range(num_points):
        angle = 60 * i
        x, y = polar_to_cartesian(radius, angle)
        point = [center[0] + x, center[1] + y]
        points.append(point)
    return points

def run_model(model, rgbs, start_points, S_max=128, iters=16):
    rgbs = rgbs.cuda().float() # B, S, C, H, W
    B, S, C, H, W = rgbs.shape
    # print("in run model: ", rgbs.shape)
    assert(B==1)

    xy0 = torch.tensor(start_points).to(rgbs.device)

    _, S, C, H, W = rgbs.shape

    # zero-vel init
    trajs_e = xy0.unsqueeze(1).repeat(1,S,1,1)
 
    iter_start_time = time.time()

    preds, preds_anim, _, _ = model(trajs_e, rgbs, iters=iters, feat_init=None, beautify=True)

    trajs_e = preds[-1]

    iter_time = time.time()-iter_start_time
    # print('inference time: %.2f seconds (%.1f fps)' % (iter_time, S/iter_time))

    return trajs_e

def count_clusters(trace):
    cluster_num_list=[]
    target_index_list=[]
    index = trace.shape[1]-1
    for i in range(trace.shape[1]):
      frame_coords = trace[0][i]
      dbscan = DBSCAN(eps=10, min_samples=1)
      clusters = dbscan.fit_predict(frame_coords.cpu())
      
      cluster_num = len(set(clusters))
      if cluster_num>=2:
          target_index_list.append(i)
      
      cluster_num_list.append(cluster_num)
    
    print("in this segment, cluster num list is: ", cluster_num_list)

    for j in range(len(target_index_list)):
        target_index = target_index_list[j]
        if (cluster_num_list[target_index] >= 2) and (cluster_num_list[min(target_index+1,trace.shape[1]-1)] == 1) and (cluster_num_list[min(target_index+2,trace.shape[1]-1)] == 1) and (cluster_num_list[min(target_index+3,trace.shape[1]-1)]) == 1:
            print("ignore a abnormal")
            continue
        return target_index-3
    return trace.shape[1]-1


def main(
        start_points,
        test_what,
        save_name,
        S=48, # seqlen
        user_id='yufei081523',
        camera_setting='Camera_Right60_Vertical20',
        typing_content='label-yufei081523-081523_143727-37-local_handposes',
        hand = 'cropped_right_hand',
        log_dir='./logs_demo',
        timestride=1, # temporal stride of the model
        iters=16, # inference steps of the model
        image_size=(512,896), # input resolution
        max_iters=4, # number of clips to run
        shuffle=False, # dataset shuffling
        log_freq=1, # how often to make image summaries
        init_dir='./reference_model',
        device_ids=[0],
):
    
    frame_root = f'../ccs/vr_typing/{user_id}/{typing_content}'
    print(frame_root)
    print(test_what)
    print(save_name)

    hand_image_files = [os.path.join(frame_root, f) for f in sorted(os.listdir(frame_root)) if f.startswith('frame') and f.endswith('.png')]
    hand_images = []
    for hand_file_path in hand_image_files:
        img = cv2.imread(hand_file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hand_images.append(img)
    rgbs = np.array(hand_images)  
    rgbs = rgbs[:,:,:,::-1].copy() # BGR->RGB
    rgbs = rgbs[::timestride]
    S_here,H,W,C = rgbs.shape
    print('rgbs', rgbs.shape)

    fingertips = torch.zeros(1, rgbs.shape[0],len(start_points[0]), 2)

    model_name = 'model'
    import datetime
    model_date = datetime.datetime.now().strftime('%H:%M:%S')
    model_name = model_name + '_' + model_date
    print('model_name', model_name)  
    model = Pips(stride=8).cuda()
    parameters = list(model.parameters())
    if init_dir:
        _ = saverloader.load(init_dir, model)
    model.eval()

    si = 0
    relabel=0
    while si < rgbs.shape[0]-1:
    # while si < 1599:        
      
        print(si+1)
        if (si == 0) and (relabel==0):
            frame0_points = start_points
        
        

        iter_start_time = time.time()


        
        if si==0:
            rgb_seq = rgbs[si:si+S]
        else:
            rgb_seq = rgbs[si:min(si+S, rgbs.shape[0])]


        rgb_seq = torch.from_numpy(rgb_seq).permute(0,3,1,2).to(torch.float32) # S,3,H,W
        rgb_seq = F.interpolate(rgb_seq, image_size, mode='bilinear').unsqueeze(0) # 1,S,3,H,W

        with torch.no_grad():
            trajs_e = run_model(model, rgb_seq, frame0_points, S_max=S, iters=iters) # (1,S_here,N,2)
            
            print(f"begin frame:{si+1},end frame:{si+trajs_e.shape[1]}")
            for i in range(trajs_e.shape[1]):
                fingertips[0][si+i] = trajs_e[0][i]
            
            offset = count_clusters(trajs_e)
            print("in this segment, offset is: ", offset)
            if offset<5:
                print("relabel______________________")
                
                center=calculate_centroid_mean(frame0_points[0])
                new_points=[]
                new_points.append(generate_points(center, 5, 6))

                # center=calculate_centroid_mean(frame0_points[0][5:]) # [[],[],[]...]
                # new = frame0_points[0][:5]+generate_points(center, 5, 6)
                # new_points=[]
                # new_points.append(new)
                
                if new_points==frame0_points:
                    offset = 1
                    print("ineffective relabel, shift right 1")
                
                else:
                    relabel=1
                    offset=0
                    frame0_points=new_points
            
            if(offset):
                temp = fingertips[0][si+offset].round()
                frame0_points = [temp.int().cpu().tolist()]
                
                center=calculate_centroid_mean(frame0_points[0])
                new_points=[]
                new_points.append(generate_points(center, 5, 6))
                frame0_points=new_points

                # center=calculate_centroid_mean(frame0_points[0][5:]) # [[],[],[]...]
                # new = frame0_points[0][:5]+generate_points(center, 5, 6)
                # new_points=[]      
                # new_points.append(new)

            si+=offset




            



   
        
        iter_time = time.time()-iter_start_time
        
        # print('%s; step %06d/%d; itime %.2f' % (
        #     model_name, global_step, max_iters, iter_time))

        


        

    
    # stacked_trajs_e = torch.cat(all_trajs_e, dim=1)
    print("final trajs's shape: ", fingertips.shape)
    save_path = f'./result/{user_id}/{camera_setting}/{typing_content}/{hand}/{test_what}'
    os.makedirs(save_path, exist_ok=True)
    torch.save(fingertips, f'{save_path}/{save_name}.pt')  


if __name__ == '__main__':
    Fire(main)



