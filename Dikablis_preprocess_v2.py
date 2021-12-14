import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
import argparse

'''
Only process on Dikablis dataset


output directory tree will looks like this:

args.root/TEyeD
            |-image
            |   |-0000000.png
            |   |-0000001.png
            |   ...
            |
            |-gaze
            |   |-gaze.txt
            |
            |-landmark
            |   |-pupil_landmark.txt
            |   |-iris_landmark.txt
            |   |-lid_landmark.txt
            |
            |-pupilsegmentation
            |   |-0000000.png
            |   |-0000001.png
            |   ...
            |
            |-irissegmentation
            |   |-0000000.png
            |   |-0000001.png
            |   ...
            |
            |-lidsegmentation
            |   |-0000000.png
            |   |-0000001.png
            |   ...
            ...

'''

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, required=True, help='directory path before TEyeD')
parser.add_argument('--fpv', type=int, default=3000, help='how many frame per video you want to store.')
args = parser.parse_args()

# ROOT
# root = "/home/brianw0924/hdd/"
root = args.root

# PATH
video_path = os.path.join(root, "20MioEyeDS/TEyeDSSingleFiles/Dikablis/VIDEOS")
label_path = os.path.join(root, "20MioEyeDS/TEyeDSSingleFiles/Dikablis/ANNOTATIONS")
video_list = os.listdir(video_path)
tqdm.write(f'Number of videos: {len(video_list)}')
save_path = os.path.join(root,"TEyeD")

# OUTPUT DIR
if not os.path.exists(save_path):
    os.mkdir(save_path)
    os.mkdir(save_path,"image")
    os.mkdir(save_path,"gaze")
    os.mkdir(save_path,"landmark")
    os.mkdir(save_path,"pupilsegmentation")
    os.mkdir(save_path,"irissegmentation")
    os.mkdir(save_path,"lidsegmentation")

# BROKEN FILE
broken = []
broken.append("DikablisSS_10_1.mp4")
with open("./others/pupil_seg_broken.txt", 'r') as p:
    with open("./others/iris_seg_broken.txt", 'r') as i:
        with open("./others/lid_seg_broken.txt", 'r') as l:
            for line in p.readlines():
                broken.append(line.strip())
            for line in i.readlines():
                broken.append(line.strip())
            for line in l.readlines():
                broken.append(line.strip())

with open(os.path.join(save_path,"gaze","gaze.txt"), 'w') as gaze:
    with open(os.path.join(save_path,"landmark","pupil_landmark.txt"), 'w') as p_landmark:
        with open(os.path.join(save_path,"landmark","iris_landmark.txt"), 'w') as i_landmark:
            with open(os.path.join(save_path,"landmark","lid_landmark.txt"), 'w') as l_landmark:
                gaze.write("x,y,z\n")
                p_landmark.write("x,y,x,y, ...\n")
                i_landmark.write("x,y,x,y, ...\n")
                l_landmark.write("x,y,x,y, ...\n")
                source_image_idx = 0
                pupil_seg_idx = 0
                iris_seg_idx = 0
                lid_seg_idx = 0
                for video_name in tqdm(video_list):
                    if video_name in broken:
                        continue
                    tqdm.write(video_name)
                    '''
                    image shape: (288, 384, 3) in Dikablis
                    '''

                    # Source video
                    video = cv2.VideoCapture(os.path.join(video_path,video_name))
                    success = True
                    source_image_count = 0
                    while(success):
                        success, frame = video.read()
                        if not success:
                            break
                        im = Image.fromarray(frame)
                        im.save(os.path.join(save_path,"image",f'{str(source_image_idx).zfill(7)}.png'))
                        source_image_count+=1
                        source_image_idx+=1
                        if source_image_count == args.fpv: # reach desired frames per video
                            break

                    # Gaze vector: FRAME;x;y;z;\n
                    text_full_path = os.path.join(label_path,f'{video_name}gaze_vec.txt')
                    count = 0
                    with open(text_full_path) as f:
                        next(f)
                        for i, line in enumerate(f.readlines()):
                            l = ','.join(line.split(';')[1:-1]) # x,y,z
                            gaze.write(f'{l}\n') # x,y,z\n
                            count+=1
                            if i+1 == source_image_count:
                                break
                    if count != source_image_count:
                        tqdm.write(f'{video_name}gaze_vec.txt is broken.')

                    # pupil landmark: FRAME;AVG INACCURACY;x;y;x;y;...;\n
                    text_full_path = os.path.join(label_path,f'{video_name}pupil_lm_2D.txt')
                    count = 0
                    with open(text_full_path) as f:
                        next(f)
                        for i, line in enumerate(f.readlines()):
                            l = ','.join(line.split(';')[2:-1]) # x,y,x,y,x,y, ...
                            p_landmark.write(f'{l}\n')  # x,y,x,y,x,y, ...\n
                            count+=1
                            if i+1 == source_image_count:
                                break
                    if count != source_image_count:
                        tqdm.write(f'{video_name}pupil_lm_2D.txt is broken.')

                    # iris landmark: FRAME;AVG INACCURACY;x;y;x;y;...;\n
                    text_full_path = os.path.join(label_path,f'{video_name}iris_lm_2D.txt')
                    count = 0
                    with open(text_full_path) as f:
                        next(f)
                        for i, line in enumerate(f.readlines()):
                            l = ','.join(line.split(';')[2:-1]) # x,y,x,y,x,y, ...
                            i_landmark.write(f'{l}\n') # x,y,x,y,x,y, ...\n
                            count+=1
                            if i+1 == source_image_count:
                                break
                    if count != source_image_count:
                        tqdm.write(f'{video_name}iris_lm_2D.txt is broken.')

                    # lid landmark: FRAME;AVG INACCURACY;x;y;x;y;...;\n
                    text_full_path = os.path.join(label_path,f'{video_name}lid_lm_2D.txt')
                    count = 0
                    with open(text_full_path) as f:
                        next(f)
                        for i, line in enumerate(f.readlines()):
                            l = ','.join(line.split(';')[2:-1]) # x,y,x,y,x,y, ...
                            l_landmark.write(f'{l}\n') # x,y,x,y,x,y, ...\n
                            count+=1
                            if i+1 == source_image_count:
                                break
                    if count != source_image_count:
                        tqdm.write(f'{video_name}lid_lm_2D.txt is broken.')

                    # pupil 2D seg
                    video_full_path = os.path.join(label_path,f'{video_name}pupil_seg_2D.mp4')
                    video = cv2.VideoCapture(video_full_path)
                    success = True
                    count = 0
                    while(success):
                        success, frame = video.read()
                        if not success:
                            break
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        frame[frame<=128] = 0
                        frame[frame>128] = 255
                        im = Image.fromarray(frame)
                        im.save(os.path.join(save_path,"pupilsegmentation",f'{str(pupil_seg_idx).zfill(7)}.png'))
                        count+=1
                        pupil_seg_idx+=1
                        if count == source_image_count:
                            break
                    if count != source_image_count:
                        tqdm.write(f'{video_name}pupil_seg_2D.mp4 is broken.')
                        
                        
                    # iris 2D seg
                    video_full_path = os.path.join(label_path,f'{video_name}iris_seg_2D.mp4')
                    video = cv2.VideoCapture(video_full_path)
                    success = True
                    count = 0
                    while(success):
                        success, frame = video.read()
                        if not success:
                            break
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        frame[frame<=128] = 0
                        frame[frame>128] = 255
                        im = Image.fromarray(frame)
                        im.save(os.path.join(save_path,"irissegmentation",f'{str(iris_seg_idx).zfill(7)}.png'))
                        count+=1
                        iris_seg_idx+=1
                        if count == source_image_count:
                            break
                    if count != source_image_count:
                        tqdm.write(f'{video_name}iris_seg_2D.mp4 is broken.')


                    # lid 2D seg   
                    video_full_path = os.path.join(label_path,f'{video_name}lid_seg_2D.mp4')
                    video = cv2.VideoCapture(video_full_path)
                    success = True
                    count = 0
                    while(success):
                        success, frame = video.read()
                        if not success:
                            break
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        frame[frame<=128] = 0
                        frame[frame>128] = 255
                        im = Image.fromarray(frame)
                        im.save(os.path.join(save_path,"lidsegmentation",f'{str(lid_seg_idx).zfill(7)}.png'))
                        count+=1
                        lid_seg_idx+=1
                        if count == source_image_count:
                            break
                    if count != source_image_count:
                        tqdm.write(f'{video_name}lid_seg_2D.mp4 is broken.')
