import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse

'''
Only process on Dikablis dataset


output:

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
            |   |-landmark.txt


'''

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, required=True)
parser.add_argument('--fpv', type=int, default=1000)
args = parser.parse_args()

# root = "/home/brianw0924/hdd/"
root = args.root

video_path = os.path.join(root, "20MioEyeDS/TEyeDSSingleFiles/Dikablis/VIDEOS")
video_list = os.listdir(video_path)
tqdm.write(f'Number of videos: {len(video_list)}')
label_path = os.path.join(root, "20MioEyeDS/TEyeDSSingleFiles/Dikablis/ANNOTATIONS")
save_path = os.path.join(root,"TEyeD")
if not os.path.exists(save_path):
    os.mkdir(save_path)
    os.mkdir(save_path,"image")
    os.mkdir(save_path,"gaze")
    os.mkdir(save_path,"landmark")


frame_per_video = args.fpv # only get first 1000 frame per video
image_count = 0
with open(os.path.join(save_path,"gaze","gaze.txt"), 'w') as gaze:
    with open(os.path.join(save_path,"landmark","landmark.txt"), 'w') as landmark:
        for video_name in tqdm(video_list):
            '''
            image shape: (288, 384, 3)
            '''
            tqdm.write(f'Video: {video_name}')
            video_full_path = os.path.join(video_path,video_name)
            video = cv2.VideoCapture(video_full_path)
            success = True
            frame_id = 0
            while(success):
                success, frame = video.read()
                if not success:
                    break
                im = Image.fromarray(frame)
                im.save(os.path.join(save_path,"image",f'{str(image_count).zfill(7)}.png'))
                image_count+=1
                frame_id+=1
                if frame_id == frame_per_video:
                    break

            # Gaze vector
            text_full_path = os.path.join(label_path,f'{video_name}gaze_vec.txt')
            with open(text_full_path) as f:
                next(f)
                for i, line in enumerate(f.readlines()):
                    gaze.write(line)
                    if i+1 == frame_id:
                        break

            # Gaze vector
            text_full_path = os.path.join(label_path,f'{video_name}pupil_lm_2D.txt')
            with open(text_full_path) as f:
                next(f)
                for i, line in enumerate(f.readlines()):
                    landmark.write(line)
                    if i+1 == frame_id:
                        break

            # # pupil 2D seg
            # pupil_2D = []
            # video_full_path = os.path.join(label_path,f'{video_name}pupil_seg_2D.mp4')
            # video = cv2.VideoCapture(video_full_path)
            # success = True
            # count = 0
            # while(success):
            #     success, frame = video.read()
            #     if not success:
            #         break
            #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #     frame[frame<=128] = 0
            #     frame[frame>128] = 255
            #     pupil_2D.append(frame)
            #     count+=1
            #     if count == frame_id:
            #         break
            # if count != frame_id:
            #     tqdm.write(f'{video_name}pupil_seg_2D.mp4 is broken.')
            #     continue
                
                
            # # iris 2D seg
            # iris_2D = []
            # video_full_path = os.path.join(label_path,f'{video_name}iris_seg_2D.mp4')
            # video = cv2.VideoCapture(video_full_path)
            # success = True
            # count = 0
            # while(success):
            #     success, frame = video.read()
            #     if not success:
            #         break
            #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #     frame[frame<=128] = 0
            #     frame[frame>128] = 255
            #     iris_2D.append(frame)
            #     count+=1
            #     if count == frame_id:
            #         break
            # if count != frame_id:
            #     tqdm.write(f'{video_name}iris_seg_2D.mp4 is broken.')
            #     continue


            # # lid 2D seg   
            # lid_2D = []
            # video_full_path = os.path.join(label_path,f'{video_name}lid_seg_2D.mp4')
            # video = cv2.VideoCapture(video_full_path)
            # success = True
            # count = 0
            # while(success):
            #     success, frame = video.read()
            #     if not success:
            #         break
            #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #     frame[frame<=128] = 0
            #     frame[frame>128] = 255
            #     lid_2D.append(frame)
            #     count+=1
            #     if count == frame_id:
            #         break
            # if count != frame_id:
            #     tqdm.write(f'{video_name}lid_seg_2D.mp4 is broken.')
            #     continue
