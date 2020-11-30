#!./venv/bin/python3

from multiprocessing import Pool
import os
import seam_carving

import time

import numpy as np
import cv2
import argparse
from numba import jit
from scipy import ndimage as ndi

path = 'C:/Users/lajja/Documents/Fall2020/ML/vqa/test/test/'
#train_img_len = len(os.listdir(path))

# for efficiency reasons in seam_carving.py
IN_DIR = None
OUT_DIR = None
WIDTH = None
USE_SEAM_CARVING = True
DOWNSIZE_WIDTH = 448

def processImages(width, imgDir, outputDir, seam_carve=True):
    '''takes in all images in imgDir, and resizes them with seam carving
       to the dimensions width x width, via forward energy.
    '''
    assert outputDir is not None and imgDir is not None

    count = 0
    start = time.time()
    prev = time.time()

    allImg = sorted(os.listdir(imgDir))

    for filename in allImg:
        if filename in os.listdir(outputDir):
            continue
        # read the image
        full_path = imgDir + filename
        im = cv2.imread(full_path)

        # downsize the image
        im = seam_carving.resize(im, width=DOWNSIZE_WIDTH)
        
        # get image resolution
        h, w = im.shape[:2]

        # make the larges square possible for a particular image
        # TODO ignore widths for now
        # dy is number of vertical seams to extract,
        # dx is number of horizontal seams to extract
        dy = None
        dy = None

        # TODO this makes no sense
        if h > w:
            dy = 0
            dx = w - h
        else:
            dy = h - w
            dx = 0
        assert dy != None and dx != None

        # for now we assume the mask is None
        #print('   processing image', full_path, '...')

        if USE_SEAM_CARVING :
            output = seam_carving.seam_carve(im, dx, dy, mask=None,vis=False)
        else:
            output = cv2.resize(im, (WIDTH, WIDTH), interpolation=cv2.INTER_AREA)

        cv2.imwrite(outputDir + filename, output)

        # print out progress
        count += 1
        now = time.time()
        if count % 50 == 0:
            print('   processed image', full_path, 'in', int(now-prev), 's')
           
            print('-- time elapse:', int(prev-start), 's, processed:', count,
                'average time:', int((prev-start)/count), 's', int(count/len(allImg)*100), 'percent complete')
        prev = now


if __name__ == '__main__':

    # parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-in", help="Path to Directory of raw images", required=True)
    ap.add_argument("-out", help="Path to Directory of output images", required=True)
    ap.add_argument("-w", help="Width of square image output", required=True)
    ap.add_argument("-s", help="Use seam carving to resize images if set to true, normal resizing otherwise", action="store_true")
    args = vars(ap.parse_args())

    USE_SEAM_CARVING = ap.parse_args().s
    IN_DIR, OUT_DIR, WIDTH,  = args['in'], args['out'], int(args['w'])

    if USE_SEAM_CARVING:
        processImages(WIDTH, IN_DIR, OUT_DIR)
    else:
        processImages(WIDTH, IN_DIR, OUT_DIR, seam_carve=False)
        




