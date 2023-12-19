import os
os.environ['OMP_NUM_THREADS'] = '1'
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from fire import Fire
import cv2
import torch.nn.functional as F
import torch

from sklearn.cluster import KMeans

# Set the environment variable within the script
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Environment variables are strings, not booleans

def count_clusters(trace):
    cluster_num_list=[]
    flag = 0
    index = len(trace)-1
    for i in range(trace.shape[1]):
        frame_coords = trace[0][i]
        dbscan = DBSCAN(eps=12, min_samples=1)
        clusters = dbscan.fit_predict(frame_coords)
        cluster_num = len(set(clusters))
        if (cluster_num>=2) and (flag==0):
            index = i-3
            flag = 1
        cluster_num_list.append(cluster_num)
    print("in this segment, cluster num list is: ", cluster_num_list)
    return index

def main(user_id, camera_setting, typing_content, hand, offset, test_what, save_name, begin_no, image_size=(512,896)):

    
    coords = torch.load(f'./result/{user_id}/{camera_setting}/{typing_content}/{hand}/{test_what}/{save_name}.pt', map_location=torch.device('cpu'))  # 形状为[1, 64, 4, 2]
    # coords = torch.load(f'./data/{user_id}/{camera_setting}/{typing_content}/{save_name}.pt', map_location=torch.device('cpu'))  # 形状为[1, 64, 4, 2]
    coords = coords[:,begin_no:,:,:]
    print(coords.shape)

    frame_root = f'./data/{user_id}/{camera_setting}/{typing_content}/{hand}'
    hand_image_files = [os.path.join(frame_root, f) for f in sorted(os.listdir(frame_root)) if f.startswith('frame') and f.endswith('.png')]
    hand_images = []
    for hand_file_path in hand_image_files:
        img = cv2.imread(hand_file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hand_images.append(img)
    rgbs = np.array(hand_images) 
    rgb_seq = rgbs
    seq = torch.from_numpy(rgb_seq).permute(0,3,1,2).to(torch.float32) # S,3,H,W
    
    for frame_idx in range(seq.shape[0]):
        selected_seq = seq[frame_idx:frame_idx+1, :, :, :]  # 选择第 i 个索引的数据，并在第一维上增加一个维度
        rgb_seq = F.interpolate(selected_seq, image_size, mode='bilinear').unsqueeze(0) # 1,S=1,3,H,W
        


        
        rgb_seq = rgb_seq.to('cpu').numpy().astype('uint8')
        
        

        print(frame_idx+offset+1)  #frame number
        frame = rgb_seq[0, 0]
        frame_coords = coords[0, frame_idx]




        

        # for i in range(0,5):
        #     plt.scatter(frame_coords[i, 0], frame_coords[i, 1], c='green', s=10) 
        
        
        
        dbscan = DBSCAN(eps=8, min_samples=1)
        # print(frame_coords.shape)
        clusters = dbscan.fit_predict(frame_coords)


        color = ["red","orange","yellow","green","blue","purple"]
        plt.imshow(frame.transpose(1, 2, 0))      
        
        for point, cluster in zip(frame_coords, clusters):
            if cluster == -1:
                plt.scatter(point[0], point[1], color='black',s=10)
            else:
                plt.scatter(point[0], point[1], color=color[cluster], s=10)


        # color = ["red","orange","yellow","green","blue","purple"]
        # plt.imshow(frame.transpose(1, 2, 0))  
        # kmeans = KMeans(n_clusters=4)
        # kmeans.fit(frame_coords)
        # labels = kmeans.predict(frame_coords)
        # for point, label in zip(frame_coords, labels):
        #     plt.scatter(point[0], point[1], color=color[label], s=10)
        
        
        cluster_path = f'./result/{user_id}/{camera_setting}/{typing_content}/{hand}/{test_what}/cluster/{save_name}'
        os.makedirs(cluster_path, exist_ok=True)
        plt.savefig(f'{cluster_path}/frame_{frame_idx+offset+1:05d}.png')
        plt.clf()

# python .\validate_trajectory.py --user_id zhuolinnumbs051223 --camera_setting Camera_Right50_Vertical20 --typing_content label-zhuolinnumbs051223-051223_072851-4-local_handposes --hand cropped_right_hand --offset --save_name
# offset frame-1 从895帧开始 offset=894

if __name__ == '__main__':
    clicks = []
    Fire(main) 

