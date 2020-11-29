#!venv/bin/python3

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

DOWNSIZE_WIDTH = 500

def scanImageSizes(imgDir):
    '''scans all images in the directory to find the mean size, median
       size, min size, max size
    '''
    return 42

def processImages(width, imgDir, outputDir):
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
        print('   processing image', full_path, '...')
        output = seam_carving.seam_carve(im, dx, dy, mask=None,vis=False)
        cv2.imwrite(outputDir + filename, output)

        # print out progress
        count += 1
        now = time.time()
        print('   processed image', full_path, 'in', int(now-prev), 's')
        prev = now

        print('-- time elapse:', int(prev-start), 's, processed:', count,
              'average time:', int((prev-start)/count), 's')


if __name__ == '__main__':

    # parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-in", help="Path to Directory of raw images", required=True)
    ap.add_argument("-out", help="Path to Directory of output images", required=True)
    ap.add_argument("-w", help="Width of square image output", required=True)
    args = vars(ap.parse_args())

    INPUT_DIR, OUTPUT_DIR, WIDTH = args['in'], args['out'], int(args['w'])
    processImages(WIDTH, INPUT_DIR, OUTPUT_DIR)



