import os
import glob
import cv2
import math
from tqdm import tqdm
from PIL import Image
import argparse
import random


'''

Please put the directory containing videos under righteye/ or lefteye/

args.root
    |
    |-processed
    |   |
    |   |-image
    |   |   |-0000000.png
    |   |   |-0000001.png
    |   |   ...
    |   |-gaze
    |       |-gaze.txt
    |
    |-raw
        |-righteye
        |   |
        |   |-2021xxxx_H14_NSS11111
        |   |   |-0.mp4
        |   |   |-1.mp4
        |   |   |-2.mp4
        |   |   ...
        |   |-2021xxxx_H14_NSS11111
        |   ...
        |
        |-lefteye
            |
            |-2021xxxx_H14_NSS00000
            |   |-0.mp4
            |   |-1.mp4
            |   |-2.mp4
            |   ...
            |-2021xxxx_H14_NSS00000
            ...


Have to determine center  (left eye & right eye are different) & crop size by yourself

video frame size (H, W): (400, 1280) 
'''

def arg_parser():
        
    parser = argparse.ArgumentParser()

    ''' Paths '''
    parser.add_argument('--root', type=str, default="../Neurobit")
    parser.add_argument('--data_dir', type=str, default="20220121_raw")
    parser.add_argument('--out_dir', type=str, default="dataset_nocrop")

    ''' Parameters'''
    parser.add_argument('--h', type=float, default=6.8)   
    parser.add_argument('--w', type=float, default=7.7)
    parser.add_argument('--c', type=tuple, default=(0,0)) # (x,y)
    parser.add_argument('--d', type=float, default=78)    # offset
    parser.add_argument('--Lefteye_ROI', type=tuple, default=(0, 400, 640, 1280)) # (left, right, top, bot)
    parser.add_argument('--Righteye_ROI', type=tuple, default=(0, 400, 0, 640))   # (left, right, top, bot)
    


    ''' Resume the last index'''
    parser.add_argument('--resume', action="store_true")
    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.root, args.out_dir)):
        os.mkdir(os.path.join(args.root, args.out_dir))
    if not os.path.exists(os.path.join(args.root, args.out_dir, "image")):
        os.mkdir(os.path.join(args.root, args.out_dir, "image"))
    if not os.path.exists(os.path.join(args.root, args.out_dir, "gaze")):
        os.mkdir(os.path.join(args.root, args.out_dir, "gaze"))

    return args

def get_yaw_pitch(i, h, w, c, d):
    top_left_y = 4 * h + c[1]
    top_left_x = -6 * w + c[0]
    pitch = math.atan( (top_left_y - (i//13) * h) / d) * 180 / math.pi
    yaw   = math.atan( (top_left_x + (i%13)  * w) / d) * 180 / math.pi
    return yaw, pitch


def DataPreprocessing_nocrop():

    ''' 直接影片轉圖片，不做 cropping '''

    args = arg_parser()

    ''' parametrers '''
    L_top, L_bot, L_left, L_right = args.Lefteye_ROI[0], args.Lefteye_ROI[1], args.Lefteye_ROI[2], args.Lefteye_ROI[3]
    R_top, R_bot, R_left, R_right = args.Righteye_ROI[0], args.Righteye_ROI[1], args.Righteye_ROI[2], args.Righteye_ROI[3]

    Left_dirs = glob.glob(os.path.join(args.root, args.data_dir, "lefteye", "*"))
    Right_dirs = glob.glob(os.path.join(args.root, args.data_dir, "righteye", "*"))

    if args.resume:
        image_idx = len(os.listdir(os.path.join(args.root, args.out_dir, "image")))
    else:
        image_idx = 0

    print(f"Index start from: {str(image_idx).zfill(7)}")

    s = 'w' if image_idx == 0 else 'a'
    with open(os.path.join(args.root, args.out_dir, "gaze", "gaze.txt"), s) as f:
        if image_idx == 0:
            f.write("yaw,pitch\n")

        # left eyes
        for d in Left_dirs: # iterate thru left_eye video directories
            video_files = sorted(glob.glob(os.path.join(d,"*")))
            assert(len(video_files) == 9*13)
            for i, v in enumerate(tqdm(video_files)): # each dir represents

                yaw,pitch = get_yaw_pitch(i=float(i),h=args.h,w=args.w,c=args.c,d=args.d) # length-based label
                # yaw, pitch = -30 + (i%13) * 5, 20 - (i//13) * 5 # angle-based label

                video = cv2.VideoCapture(v)
                success = True
                while(success):
                    success, frame = video.read() # only take the first frame (or the dataset will be too big)
                    if(not success): break
                    im = frame[L_top:L_bot, L_left:L_right, :]
                    im = Image.fromarray(im)
                    im.save(os.path.join(args.root, args.out_dir, "image", f'{str(image_idx).zfill(7)}.png'))
                    image_idx+=1
                    f.write(f'{yaw},{pitch}\n')

        # right eyes
        for d in Right_dirs: # iterate thru right_eye video directories

            video_files = sorted(glob.glob(os.path.join(d,"*")))
            assert(len(video_files) == 9*13)

            for i, v in enumerate(tqdm(video_files)):

                yaw,pitch = get_yaw_pitch(i=float(i),h=args.h,w=args.w,c=args.c,d=args.d) # length-based label
                # yaw, pitch = -30 + (i%13) * 5, 20 - (i//13) * 5 # angle-based label

                video = cv2.VideoCapture(v)
                success = True
                while(success):
                    success, frame = video.read() # only take the first frame (or the dataset will be too big)
                    if(not success): break
                    im = frame[R_top:R_bot, R_left:R_right, :]
                    im = Image.fromarray(im)
                    im.save(os.path.join(args.root, args.out_dir, "image", f'{str(image_idx).zfill(7)}.png'))
                    image_idx+=1
                    f.write(f'{yaw},{pitch}\n')

def DataPreprocessing_v1():

    ''' 每個影片只取部分 frame，同一 frame 做上下左右平移 '''

    args = arg_parser()

    ''' parametrers '''
    step = 10
    L_top, L_bot, L_left, L_right = args.Lefteye_ROI[0], args.Lefteye_ROI[1], args.Lefteye_ROI[2], args.Lefteye_ROI[3]
    R_top, R_bot, R_left, R_right = args.Righteye_ROI[0], args.Righteye_ROI[1], args.Righteye_ROI[2], args.Righteye_ROI[3]


    Left_dirs = glob.glob(os.path.join(args.root, args.data_dir, "lefteye", "*"))
    Right_dirs = glob.glob(os.path.join(args.root, args.data_dir, "righteye", "*"))

    if args.resume:
        image_idx = len(os.listdir(os.path.join(args.root, args.out_dir, "image")))
    else:
        image_idx = 0

    print(f"Start from: {str(image_idx).zfill(7)}")

    s = 'w' if image_idx == 0 else 'a'
    with open(os.path.join(args.root,args.out_dir, "gaze", "gaze.txt"), s) as f:
        if image_idx == 0:
            f.write("yaw,pitch\n")

        # left eyes
        for d in Left_dirs: # iterate thru left_eye video directories
            video_files = sorted(glob.glob(os.path.join(d,"*")))
            assert(len(video_files) == 9*13)
            for i, v in enumerate(tqdm(video_files)): # each dir represents

                yaw,pitch = get_yaw_pitch(i=float(i),h=args.h,w=args.w,c=args.c,d=args.d) # length-based label
                # yaw, pitch = -30 + (i%13) * 5, 20 - (i//13) * 5 # angle-based label

                video = cv2.VideoCapture(v)
                success = True
                count = 0
                while(success and count < 251):
                    count+=1
                    success, frame = video.read() # only take the first frame (or the dataset will be too big)
                    if(not success): break
                    if count % 50 == 0:
                        for vertical in range(-50,51,step):
                            for horizontal in range(-70,71,step):
                                im = frame[L_top+vertical:L_bot+vertical, L_left+horizontal:L_right+horizontal, :]
                                im = Image.fromarray(im)
                                im.save(os.path.join(args.root,args.out_dir, "image", f'{str(image_idx).zfill(7)}.png'))
                                image_idx+=1
                                f.write(f'{yaw},{pitch}\n')

        # right eyes
        for d in Right_dirs: # iterate thru right_eye video directories

            video_files = sorted(glob.glob(os.path.join(d,"*")))
            assert(len(video_files) == 9*13)

            for i, v in enumerate(tqdm(video_files)):

                yaw,pitch = get_yaw_pitch(i=float(i),h=args.h,w=args.w,c=args.c,d=args.d) # length-based label
                # yaw, pitch = -30 + (i%13) * 5, 20 - (i//13) * 5 # angle-based label
                video = cv2.VideoCapture(v)

                success = True
                count = 0
                while(success and count < 251):
                    count+=1
                    success, frame = video.read() # only take the first frame (or the dataset will be too big)
                    if(not success): break
                    if count % 50 == 0:
                        for vertical in range(-50,51,step):
                            for horizontal in range(-70,71,step):
                                im = frame[R_top+vertical:R_bot+vertical, R_left+horizontal:R_right+horizontal, :]
                                im = Image.fromarray(im)
                                im.save(os.path.join(args.root,args.out_dir, "image", f'{str(image_idx).zfill(7)}.png'))
                                image_idx+=1
                                f.write(f'{yaw},{pitch}\n')

def DataPreprocessing_v2():

    ''' 每個 frame 隨機上下左右平移'''

    args = arg_parser()

    ''' parametrers '''
    L_top, L_bot, L_left, L_right = args.Lefteye_ROI[0], args.Lefteye_ROI[1], args.Lefteye_ROI[2], args.Lefteye_ROI[3]
    R_top, R_bot, R_left, R_right = args.Righteye_ROI[0], args.Righteye_ROI[1], args.Righteye_ROI[2], args.Righteye_ROI[3]

    Left_dirs = glob.glob(os.path.join(args.root, args.data_dir, "lefteye", "*"))
    Right_dirs = glob.glob(os.path.join(args.root, args.data_dir, "righteye", "*"))

    if args.resume:
        image_idx = len(os.listdir(os.path.join(args.root, args.out_dir, "image")))
    else:
        image_idx = 0

    print(f"Start from: {str(image_idx).zfill(7)}")

    s = 'w' if image_idx == 0 else 'a'
    with open(os.path.join(args.root,args.out_dir, "gaze", "gaze.txt"), s) as f:
        if image_idx == 0:
            f.write("yaw,pitch\n")

        # left eyes
        for d in Left_dirs: # iterate thru left_eye video directories
            video_files = sorted(glob.glob(os.path.join(d,"*")))
            assert(len(video_files) == 9*13)
            for i, v in enumerate(tqdm(video_files)): # each dir represents

                yaw,pitch = get_yaw_pitch(i=float(i),h=args.h,w=args.w,c=args.c,d=args.d) # length-based label
                # yaw, pitch = -30 + (i%13) * 5, 20 - (i//13) * 5 # angle-based label

                video = cv2.VideoCapture(v)
                success = True
                while(success):
                    success, frame = video.read() # only take the first frame (or the dataset will be too big)
                    if(not success): break

                    for i in range(2):
                        vertical = random.randint(-50,50)
                        horizontal = random.randint(-70,70)
                        im = frame[L_top+vertical:L_bot+vertical, L_left+horizontal:L_right+horizontal, :]
                        im = Image.fromarray(im)
                        im.save(os.path.join(args.root,args.out_dir, "image", f'{str(image_idx).zfill(7)}.png'))
                        image_idx+=1
                        f.write(f'{yaw},{pitch}\n')

        # right eyes
        for d in Right_dirs: # iterate thru right_eye video directories

            video_files = sorted(glob.glob(os.path.join(d,"*")))
            assert(len(video_files) == 9*13)

            for i, v in enumerate(tqdm(video_files)):

                yaw,pitch = get_yaw_pitch(i=float(i),h=args.h,w=args.w,c=args.c,d=args.d) # length-based label
                # yaw, pitch = -30 + (i%13) * 5, 20 - (i//13) * 5 # angle-based label

                video = cv2.VideoCapture(v)

                success = True
                while(success):
                    success, frame = video.read() # only take the first frame (or the dataset will be too big)
                    if(not success): break
                    for i in range(2):
                        vertical = random.randint(-50,50)
                        horizontal = random.randint(-70,70)
                        im = frame[R_top+vertical:R_bot+vertical, R_left+horizontal:R_right+horizontal, :]
                        im = Image.fromarray(im)
                        im.save(os.path.join(args.root,args.out_dir, "image", f'{str(image_idx).zfill(7)}.png'))
                        image_idx+=1
                        f.write(f'{yaw},{pitch}\n')

if __name__ == "__main__":
    random.seed(2022)
    DataPreprocessing_nocrop()
    # DataPreprocessing_v1()
    # DataPreprocessing_v2()
