import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Environment variables are strings, not booleans
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy import signal

x_begin=445
x_end=645
y_begin=130
y_end=275
user_id='zhuolinnumbs051223'
typing_content='label-zhuolinnumbs051223-051223_072920-5-local_handposes'
camera_setting='Camera_Right50_Vertical20'
hand='cropped_left_hand'
test_what='second_finger_left_hand'
save_name='0_all_48'
raw_data_path = f'E:/ccs/vrtyping/raw_data/{typing_content}.csv'
from0=True

def gen_SN_points(x_begin,x_end,y_begin,y_end):
    # x是水平 y是竖直
    # return的是按行
    points_x = np.linspace(x_begin, x_end, 5).astype(int)
    points_y = np.linspace(y_begin, y_end, 4).astype(int)
    grid_x, grid_y = np.meshgrid(points_x, points_y)
    result = np.column_stack((grid_x.flatten(), grid_y.flatten()))
    return result

# initial_SN_points=[]
# initial_SN_points.append(gen_SN_points(x_begin,x_end,y_begin,y_end).tolist())
# print(initial_SN_points)




SN_points = torch.load(f'./result/{user_id}/{camera_setting}/{typing_content}/{hand}/{test_what}/{save_name}.pt', map_location=torch.device('cpu'))  # 形状为[1, 64, 4, 2]
SN_points=SN_points[0][:,:,1]  #[64,4]
SN_points=SN_points[:,1]
# SN_points=torch.mean(SN_points, dim=1) #[64]



def get_peaks_seperate_hand(handposes, thres, point_num=1, which_hand='left', vertical_axis=1, filters=(3, 0.15)): 
    # axis 1 是竖直方向
    # point_num=1: handposes [64]
    
    # setup butterworth threshold
    b1, a1 = signal.butter(filters[0], filters[1])
    
    if which_hand == 'left': 
        # filter raw position data and then take double derivatives
        if(point_num==1):
            all_trend = np.asarray([np.diff(np.diff(signal.filtfilt(b1, a1, handposes)))])
        else:
            all_trend = np.asarray([np.diff(np.diff(signal.filtfilt(b1, a1, handposes[1, vertical_axis, :]))),  # [,,,]
                                    np.diff(np.diff(signal.filtfilt(b1, a1, handposes[2, vertical_axis, :]))),
                                    np.diff(np.diff(signal.filtfilt(b1, a1, handposes[3, vertical_axis, :]))),
                                    np.diff(np.diff(signal.filtfilt(b1, a1, handposes[4, vertical_axis, :]))),
                                    ])
    else:
        all_trend = np.asarray([np.diff(np.diff(signal.filtfilt(b1, a1, handposes[6, vertical_axis, :]))),
                                np.diff(np.diff(signal.filtfilt(b1, a1, handposes[7, vertical_axis, :]))),
                                np.diff(np.diff(signal.filtfilt(b1, a1, handposes[8, vertical_axis, :]))),
                                np.diff(np.diff(signal.filtfilt(b1, a1, handposes[9, vertical_axis, :])))
                                ])

    # get the maximum negative of all 8 fingers' acceleration       
    
    trends_max = np.max(all_trend, axis=0)
    # trends_max = np.hstack((np.asarray([0] * (len(handposes[0, 1, :]) - len(trends_max))), trends_max))
    if point_num==1:
        trends_max = np.hstack((np.asarray([0] * (len(handposes) - len(trends_max))), trends_max))

    # peak detection
    predicted_timestamps = signal.find_peaks(trends_max, prominence=thres)[0] # detect timestamps based on double derivatives 
    prominences = signal.peak_prominences(trends_max, predicted_timestamps)[0]
    peak_values = trends_max[predicted_timestamps]

    return predicted_timestamps, prominences, peak_values, trends_max



# ----------------------------------
timestamp_range_begin=0
timestamp_range_end=3037
x_labels = range(timestamp_range_begin, timestamp_range_end)
threshold=2
# ------------------------------------

predicted_timestamps, prominences, peak_values, trends_max=get_peaks_seperate_hand(SN_points,threshold)

plt.figure(figsize=(10, 6))
plt.plot(x_labels, trends_max[timestamp_range_begin:timestamp_range_end], label='Trend Data')

for i in range(len(predicted_timestamps)):
    if predicted_timestamps[i] < timestamp_range_begin:
        continue
    if predicted_timestamps[i] >= timestamp_range_end:
        break
    plt.scatter(predicted_timestamps[i], peak_values[i], color='red')  # 在峰值位置绘制红色点


data = pd.read_csv(raw_data_path)
keystroke_data = data[' keystroke']
is_keystroke_data = keystroke_data.dropna()
keystroke_frame_dict = {}
for index, value in is_keystroke_data.items():
    if from0:
        keystroke_frame_dict[index] = value
    else:
        keystroke_frame_dict[index + 1] = value
keystroke_frame_dict.popitem()
right_target=['y','h','u','j','n','b','i','k','m',',','o','l','.','p','backspace']
right_frame_keystroke_dict = {key: value for key, value in keystroke_frame_dict.items() if value.lower() in right_target}
left_frame_keystroke_dict = {key: value for key, value in keystroke_frame_dict.items() if value.lower() not in right_target}


for key in left_frame_keystroke_dict.keys():
    if key < timestamp_range_begin:
        continue
    if key>=timestamp_range_end:
        break
    plt.scatter(key, trends_max[key], color='blue')
    





# plt.title('Trend Data with Detected Peaks')
# plt.xlabel('Timestamp')
# plt.ylabel('Value')
# plt.legend()
# plt.show()

# ---------------------------------------------------------------------
print(list(predicted_timestamps))
print(list(left_frame_keystroke_dict.keys()))

predictions=list(predicted_timestamps)
ground_truth=list(left_frame_keystroke_dict)

# 重新计算TP, FP, FN，使用新的规则
# 初始化计数器和剩余的预测值和真实值列表
remaining_predictions = set(predictions)
remaining_gt = set(ground_truth)
tp = 0

# 对于每个真实值，寻找最近且比它小的预测值
for gt in ground_truth:
    closest_pred = None
    min_distance = float('inf')

    for pred in predictions:
        if pred <= gt:
            distance = gt - pred
            if distance <= 6 and distance < min_distance:
                min_distance = distance
                closest_pred = pred

    if closest_pred is not None:
        # 成功匹配，更新TP，从剩余列表中删除匹配项
        tp += 1
        remaining_predictions.discard(closest_pred)
        remaining_gt.discard(gt)
        print(closest_pred,gt)

# 剩余的预测值是FP，剩余的真实值是FN
fp = len(remaining_predictions)
fn = len(remaining_gt)

sorted_remaining_gt = sorted(remaining_gt)
sorted_remaining_predictions = sorted(remaining_predictions)
print(sorted_remaining_predictions)
print(sorted_remaining_gt)
print(tp, fp, fn)
print("Precision,correct predictions: ", tp/(tp+fp))
print("Recall, predicted keypress: ", tp/(tp+fn))















































# diff_SN_points=SN_points[1:] - SN_points[:-1]
# diff_diff_SN_points=diff_SN_points[1:] - diff_SN_points[:-1]
# print(diff_diff_SN_points.shape) # (S-2,N) 若N为1，则是一维向量

# num_rows, num_cols = diff_diff_SN_points.size()
# fig, axes = plt.subplots(nrows=num_cols, ncols=1, figsize=(10, 20), sharex=True)

# for i in range(num_cols):
#     axes[i].plot(range(num_rows), diff_diff_SN_points[:, i])
#     axes[i].set_ylabel(f'Column {i + 1}')

# axes[-1].set_xlabel('Index')
# plt.tight_layout()
# plt.show()

# num_timestamps, num_points = diff_diff_SN_points.size()
# fig, axes = plt.subplots(nrows=num_points, ncols=1, figsize=(10, 15), sharex=True)
# for i in range(num_points):
#     axes[i].plot(range(num_timestamps), diff_diff_SN_points[:, i])
#     axes[i].set_ylabel(f'Point {i + 1}')
#     peaks, _ = find_peaks(diff_diff_SN_points[:, i],height=8)
#     axes[i].plot(peaks, diff_diff_SN_points[peaks, i], 'rx', label='Peaks')
#     peak_timestamps = peaks + 1  
#     print(f'Point {i + 1} Peaks Timestamps:', peak_timestamps)

# axes[-1].set_xlabel('Timestamp')
# plt.tight_layout()
# plt.show()
