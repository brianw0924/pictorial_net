import os
import glob
import cv2
from PIL import Image


'''
root
    |
    |-processed
    |   |
    |   |-image
    |   |   |-0.png
    |   |   |-1.png
    |   |   ...
    |   |-gaze
    |       |-gaze.txt
    |
    |-raw
        |-righteye
        |   |
        |   |-dir1
        |   |   |-0.mp4
        |   |   |-1.mp4
        |   |   |-2.mp4
        |   |   ...
        |   |-dir2
        |   ...
        |
        |-lefteye
            |
            |-dir1
            |   |-0.mp4
            |   |-1.mp4
            |   |-2.mp4
            |   ...
            |-dri2
            ...
'''


root = "/home/brianw0924/Desktop/Neurobit"

if not os.path.exists(os.path.join(root,"processed")):
    os.mkdir(os.path.join(root,"processed"))
if not os.path.exists(os.path.join(root,"processed","image")):
    os.mkdir(os.path.join(root,"processed","image"))
if not os.path.exists(os.path.join(root,"processed","gaze")):
    os.mkdir(os.path.join(root,"processed","gaze"))


'''
Have to determine center by yourself (left eye & right eye are different)
'''
righteye_center_x, righteye_center_y = 320,230
lefteye_center_x, lefteye_center_y = 320,250
shape = (240,320) # crop size

with open(os.path.join(root,"processed", "gaze", "gaze.txt"), 'w') as f:
    f.write("yaw,pitch\n")
    Left_dirs = glob.glob(os.path.join(root,"raw","lefteye","*"))
    Right_dirs = glob.glob(os.path.join(root,"raw","righteye","*"))
    image_idx = 0

    # left eyes
    for d in Left_dirs:
        video_files = sorted(glob.glob(os.path.join(d,"*")))
        assert(len(video_files) == 9*13)
        for i, v in enumerate(video_files):
            yaw, pitch = -30 + (i%13) * 5, 20 - (i//13) * 5
            print(yaw, pitch)
            video = cv2.VideoCapture(v)
            success = True
            while(success):
                success, frame = video.read()
                if not success:
                    break
                frame = frame[lefteye_center_y-shape[0]//2:lefteye_center_y+shape[0]//2, 640+lefteye_center_x-shape[1]//2:640+lefteye_center_x+shape[1]//2, :]
                im = Image.fromarray(frame)
                im.save(os.path.join(root,"processed", "image", f'{str(image_idx).zfill(7)}.png'))
                image_idx+=1
                f.write(f'{yaw},{pitch}\n')


    # right eyes
    for d in Right_dirs:
        video_files = sorted(glob.glob(os.path.join(d,"*")))
        assert(len(video_files) == 9*13)
        for i, v in enumerate(video_files):
            yaw, pitch = -30 + (i%13) * 5, 20 - (i//13) * 5
            print(yaw, pitch)
            video = cv2.VideoCapture(v)
            success = True
            while(success):
                success, frame = video.read()
                if not success:
                    break
                frame = frame[righteye_center_y-shape[0]//2:righteye_center_y+shape[0]//2, righteye_center_x-shape[1]//2:righteye_center_x+shape[1]//2, :]
                im = Image.fromarray(frame)
                im.save(os.path.join(root,"processed", "image", f'{str(image_idx).zfill(7)}.png'))
                image_idx+=1
                f.write(f'{yaw},{pitch}\n')
