import cv2

label_map = {
    "Step_1": 0,
    "Step_2_Left": 1,
    "Step_2_Right": 1,
    "Step_3": 2,
    "Step_4_Left": 3,
    "Step_4_Right": 3,
    "Step_5_Left": 4,
    "Step_5_Right": 4,
    "Step_6_Left": 5,
    "Step_6_Right": 5,
    "Step_7_Left": 6,
    "Step_7_Right": 6
}


video_map = {"no": 0, "yes": 1}

rotation_map = {
    "rotate_0": None,
    "rotate_90": cv2.ROTATE_90_COUNTERCLOCKWISE,
    "rotate_180": cv2.ROTATE_180,
    "rotate_270": cv2.ROTATE_90_CLOCKWISE
}
