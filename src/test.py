import os

name1 = "..\data\layer_1_train\clean_data\Step_2_Right\HandWash_002_A_01_G01_sv_0_rotate_180 (3rd copy).npy"

name2 = "..\data\layer_1_train\clean_data\Step_2_Right\HandWash_002_A_01_G01_sv_0_rotate_180 (another copy).npy"

name3 = "..\data\layer_1_train\clean_data\Step_2_Right\HandWash_002_A_01_G01_sv_0_rotate_180 (copy).npy"


process_data_path = os.path.join("..", "data", "test", "clean")
for _, label_list, _ in os.walk(process_data_path):
    for label in label_list:
        if label != 'Step_2_Left':
            continue
        # read each video file in each label
        for dirname, _, filenames in os.walk(os.path.join(process_data_path, label)):
            for filename in filenames:
                org_path = os.path.join(dirname, filename)
                print("org_path: ", org_path)
                if " (copy)" in org_path:
                    new_path = org_path.replace(" (copy)","_2")
                    print("new_path: ", new_path)
                    os.rename(org_path, new_path)
                if " (another copy)" in org_path:
                    new_path = org_path.replace(" (another copy)","_1")
                    print("new_path: ", new_path)
                    os.rename(org_path, new_path)
                if " (3rd copy)" in org_path:
                    new_path = org_path.replace(" (3rd copy)","_0")
                    print("new_path: ", new_path)
                    os.rename(org_path, new_path)
                