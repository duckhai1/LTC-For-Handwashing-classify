import os, random, sys, time
from moviepy.editor import *
import cv2

from property import *

GENERATED_VIDEO_PATH = os.path.join("..", "data", "storage", "generated")
VIDEO_GEN_NUM = 100

def generate_raw_video(is_skip_step=False):
    cnt =  55
    for num in range(VIDEO_GEN_NUM):
        cnt +=1
        print(f"Generating video num {num}")
        # determine skip step list
        if is_skip_step:
            skip_step_num = random.randint(1,6)
            skip_list = ["Step_1", "Step_2", "Step_3", "Step_4", "Step_5", "Step_6"]
            skip_step_list = random.sample(skip_list, skip_step_num)
            save_path = os.path.join(GENERATED_VIDEO_PATH, "not_full")
        else:
            skip_step_list=[]
            save_path = os.path.join(GENERATED_VIDEO_PATH, "full")

        print("skip_step_list: ", skip_step_list)

        for _, label_list, _ in os.walk(LAYER_1_TRAIN_RAWDATA_PATH):
            clip_list = []
            
            skip = False
            skip_test_cnt = 1
            for step in label_list:
                # skip step
                if any(skip_step in step for skip_step in skip_step_list):
                    continue

                # keep left or right only
                if "Left" in step or "Right" in step:
                    skip_test_cnt += 1
                    if skip_test_cnt == 2:
                        skip_test_cnt = 0
                        skip = random.choice([True, False])
                    if skip:
                        skip = False
                        continue
                    else:
                        skip = True

                candidate = random.choice(os.listdir(os.path.join(LAYER_1_TRAIN_RAWDATA_PATH, step)))
                if not candidate.endswith(VIDEO_EXTENSION):
                    continue
                candidate_path = os.path.join(LAYER_1_TRAIN_RAWDATA_PATH, step, candidate)
                print(f"\tGet video: {candidate} from step: {step}")
                
                clip = VideoFileClip(candidate_path)
                clip = clip.resize((720, 480))

                clip_len = int(clip.duration)
                split_len = random.randint(1,3)
                start = random.randint(0, clip_len-split_len)
                end = start + split_len
                clip = clip.subclip(start, end)        
                clip_list.append(clip)

            try:
                final = concatenate_videoclips(clip_list, method="compose")
                final.write_videofile(os.path.join(save_path, f"generated_{cnt}.mp4"))
            except ValueError:  # pass error raise by package fault
                pass


SCREENSHOOT_VID_PATH = os.path.join("..", "data", "storage", "handwash")
SCREENSHOOT_NUM = 2
SCREENSHOOT_PER_VID = 1
GENERATED_SCREENSHOOT_PATH = os.path.join("..", "data", "storage", "generated", "screenshoot")

def generate_video_screenshoot():
    cnt = 0
    for i in range(SCREENSHOOT_NUM // SCREENSHOOT_PER_VID):
        candidate = random.choice(os.listdir(SCREENSHOOT_VID_PATH))
        if not candidate.endswith(VIDEO_EXTENSION):
            continue

        cap = cv2.VideoCapture(os.path.join(SCREENSHOOT_VID_PATH, candidate))

        for j in range(SCREENSHOOT_PER_VID):
            cnt +=1
            cap.set(cv2.CAP_PROP_POS_FRAMES, random.randint(0,int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
            res, frame = cap.read() 
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imwrite(os.path.join(GENERATED_SCREENSHOOT_PATH, f"screenshoot_{cnt}.jpg"), frame) 


if __name__ == '__main__':
    generate_raw_video(False)
    # generate_video_screenshoot()
