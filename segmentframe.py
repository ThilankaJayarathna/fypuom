import cv2
import numpy as np
from matplotlib import pyplot as plt
import subprocess
from PIL import Image
from pathlib import Path
import glob
image_list = []




cap = cv2.VideoCapture('vedio//seka2.mp4')

imagesFolder = ('frames//segmentframe//phn')
frameRate = cap.get(5) #frame rate


while(cap.isOpened()):
    frameId = cap.get(1) #current frame number
    ret, frame = cap.read()

    if (ret != True):

        break
    if (True):
        # height, width, layers = frame.shape
        # new_h = height / 2
        # new_w = width / 2
        # resize = cv2.resize(frame, (int(new_w), int(new_h)))
        resize = cv2.resize(frame, (640, 480))
        filename = imagesFolder + "/image_" +  str(int(frameId)) + ".jpg"
        print("hjhjhj")
        cv2.imwrite(filename, resize)

cap.release()
pathlist = Path().glob('frames/segmentframe/walk/*.jpg')
for path in pathlist:
    # because path is object not string
    path_in_str = str(path)
    #print(path_in_str)
    subprocess.call(
        ["python", "/home/thilanka/PycharmProjects/fypuom/tf-openpose/run.py", "--model", "mobilenet_thin",
         "--resize", "640*480", "--image", path_in_str])

print('done')

