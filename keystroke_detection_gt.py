import numpy as np
import pandas as pd

user_id='zhuolinnumbs051223'
typing_content='label-zhuolinnumbs051223-051223_072920-5-local_handposes'
raw_data_path = f'E:/ccs/vrtyping/raw_data/{typing_content}.csv'
from0=True


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
print(left_frame_keystroke_dict)




