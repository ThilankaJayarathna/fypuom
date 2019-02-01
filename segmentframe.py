import cv2
import numpy as np
from matplotlib import pyplot as plt
import subprocess
from PIL import Image
from pathlib import Path
import glob
import argparse
import logging
import sys
import time
import os.path


from tf_pose import common
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
image_list = []




cap = cv2.VideoCapture('vedio//testwalk.avi')
#imagesFolder = ('frames//segmentframe//phn')
frameRate = cap.get(5) #frame rate


while(cap.isOpened()):
    frameId = cap.get(1) #current frame number
    ret, frame = cap.read()

    if (ret != True):

        break
    if (True):
        resize1 = cv2.resize(frame, (640, 480))
        filename = "/image_" +  str(int(frameId)) + ".jpg"
        #cv2.imwrite(filename, resize1)

#cap.release()
        logger = logging.getLogger('TfPoseEstimator')
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        model = ('mobilenet_thin')
        resize = '640*480'
        resize_out_ratio = '4'

        w, h = model_wh(resize)
        if w == 0 or h == 0:
            e = TfPoseEstimator(get_graph_path(model), target_size=(432, 368))
            #print("size comes with if")
        else:
            e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))
            #print("size comes with else")
         # estimate human poses from a single image !
        #image1 = common.read_imgfile(image, None, None)
        image1 = resize1

        if image1 is None:
            logger.error('Image can not be read, path=%s' % image1)
            sys.exit(-1)
        t = time.time()
        humans = e.inference(image1, resize_to_default=(w > 0 and h > 0), upsample_size=4.0)
        elapsed = time.time() - t

        logger.info('inference image: %s in %.4f seconds.' % (image1, elapsed))

        image1 = TfPoseEstimator.draw_humans(image1, humans, imgcopy=False)
        import matplotlib.pyplot as plt

        fig = plt.figure()
        a = fig.add_subplot(2, 2, 1)
        a.set_title('Result')
        plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
        k = 'Resultimages//'+ filename
        plt.imsave(k,image1)
        bgimg = cv2.cvtColor(image1.astype(np.uint8), cv2.COLOR_BGR2RGB)
        bgimg = cv2.resize(bgimg, (e.heatMat.shape[1], e.heatMat.shape[0]), interpolation=cv2.INTER_AREA)

        # show network output

        a = fig.add_subplot(2, 2, 2)
        a.set_title('Heatmap')
        plt.imshow(bgimg, alpha=0.5)
        tmp = np.amax(e.heatMat[:, :, :-1], axis=2)
        plt.imshow(tmp, cmap=plt.cm.gray, alpha=0.5)
        plt.colorbar()
        n = 'heatimages//' + filename
        plt.imsave(n, tmp)

        tmp2 = e.pafMat.transpose((2, 0, 1))
        tmp2_odd = np.amax(np.absolute(tmp2[::2, :, :]), axis=0)
        tmp2_even = np.amax(np.absolute(tmp2[1::2, :, :]), axis=0)

        a = fig.add_subplot(2, 2, 3)
        a.set_title('Vectormap-x')
        # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
        plt.imshow(tmp2_odd, cmap=plt.cm.gray, alpha=0.5)
        plt.colorbar()

        a = fig.add_subplot(2, 2, 4)
        a.set_title('Vectormap-y')
        # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
        plt.imshow(tmp2_even, cmap=plt.cm.gray, alpha=0.5)
        plt.colorbar()
        plt.show()

print('done')

