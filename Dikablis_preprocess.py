import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

'''
Only process on Dikablis dataset
'''

video_path = '/home/brianw0924/hdd/20MioEyeDS/TEyeDSSingleFiles/Dikablis/VIDEOS'
video_list = os.listdir(video_path)
tqdm.write(f'Number of videos: {len(video_list)}')
label_path = '/home/brianw0924/hdd/20MioEyeDS/TEyeDSSingleFiles/Dikablis/ANNOTATIONS'
save_path = '/home/brianw0924/hdd/processed_data'
if not os.path.exists(save_path):
    os.mkdir(save_path)

frame_each_video = 1000 # only get first 1000 frame per video
image_count = 0
for video_name in tqdm(video_list):
    '''
    image shape: (288, 384, 3)
    '''
    tqdm.write(f'Video: {video_name}')
    images = []
    video_full_path = os.path.join(video_path,video_name)
    video = cv2.VideoCapture(video_full_path)
    success = True
    frame_id = 0
    while(success):
        success, frame = video.read()
        if not success:
            break
        images.append(frame)
        frame_id+=1
        if frame_id == frame_each_video:
            break
    image_count+=len(images)


    # # pupil valid
    # pupil_validities = []
    # text_full_path = os.path.join(label_path,f'{video_name}validity_pupil.txt')
    # with open(text_full_path) as f:
    #     next(f)
    #     pupil_validities = []
    #     for line in f.readlines():
    #         txt = line.split(';')
    #         validity = txt[1]
    #         pupil_validities.append(validity)
    #         if int(txt[0]) == frame_id:
    #             break
    #     pupil_validities = np.asarray(pupil_validities) # [frame_id, x, y, z]

    # # iris valid
    # iris_validities = []
    # text_full_path = os.path.join(label_path,f'{video_name}validity_iris.txt')
    # with open(text_full_path) as f:
    #     next(f)
    #     for line in f.readlines():
    #         txt = line.split(';')
    #         validity = txt[1]
    #         iris_validities.append(validity)
    #         if int(txt[0]) == frame_id:
    #             break
    #     iris_validities = np.asarray(iris_validities) # [frame_id, x, y, z]

    # # lid valid
    # lid_validities = []
    # text_full_path = os.path.join(label_path,f'{video_name}validity_lid.txt')
    # with open(text_full_path) as f:
    #     next(f)
    #     for line in f.readlines():
    #         txt = line.split(';')
    #         validity = txt[1]
    #         lid_validities.append(validity)
    #         if int(txt[0]) == frame_id:
    #             break
    #     lid_validities = np.asarray(lid_validities) # [frame_id, x, y, z]



    # Gaze vector
    gazes = []
    text_full_path = os.path.join(label_path,f'{video_name}gaze_vec.txt')
    with open(text_full_path) as f:
        next(f)
        for line in f.readlines():
            txt = line.split(';')[:-1]
            gazes.append(txt)
            if int(txt[0]) == frame_id:
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

    tqdm.write(f'Image count: {image_count}')

    np.savez(
        os.path.join(save_path,f'{video_name.split(".")[0]}.npz'),
        image = images,
        gaze = gazes
    )
