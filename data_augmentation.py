#!venv/bin/python3

from multiprocessing import Pool
import os
import seam_carving

import time

import numpy as np
import cv2
import argparse
from numba import jit
from scipy import ndimage as ndi
import json


# for efficiency reasons in seam_carving.py
IN_DIR = None
OUT_DIR = None
WIDTH = None
USE_SEAM_CARVING = True
DOWNSIZE_WIDTH = 448
TRAIN_ANNOTATION = 'train.json'

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

def rotate(inDir, outDir, inAnnotation, outAnnotation):
    ''' rotates each image by 90 degrees, and updates the annotations as well'''

    count = 0
    start = time.time()
    prev = time.time()

    allImg = sorted(os.listdir(inDir))

    # parse dictionary
    rawFile = open(inAnnotation + '/' + TRAIN_ANNOTATION)
    data = json.load(rawFile)

    # parse images
    skip = 0

    newAnnotation = []

    for i in range(len(allImg)):

        # index into the correct part
        filename = allImg[i]
        annotation = data[i-skip]

        # generate full path to the image
        fullPath = inDir + '/' + filename

        # skip missing annotations
        if annotation['image'] != filename:
            skip += 1
            continue

        # read the image
        im = cv2.imread(fullPath)
        print(fullPath)

        # get rotation matrix
        rows, cols = im.shape[0], im.shape[1]
        print(rows, cols)
        M = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 1)
        im90 = cv2.warpAffine(im, M, (cols, rows))
        im180 = cv2.warpAffine(im90, M, (cols, rows))
        im270 = cv2.warpAffine(im180, M, (cols, rows))

        # generate new file names
        f = filename + '-000d'
        f90 = filename + '-090d'
        f180 = filename + '-180d'
        f270 = filename + '-270d'

        # construct new out filenames
        fp = outDir + '/' + f
        fp90 = outDir + '/' + f90
        fp180 = outDir + '/' + f180
        fp270 = outDir + '/' + f270

        # write to new directory
        cv2.imwrite(fp, im)
        cv2.imwrite(fp90, im90)
        cv2.imwrite(fp180, im180)
        cv2.imwrite(fp270, im270)

        print(filename, annotation['image'])
        #full_path = inDir + filename


if __name__ == '__main__':

    # parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-in", help="Path to Directory of raw images", required=True)
    ap.add_argument("-out", help="Path to Directory of output images", required=True)
    ap.add_argument("-w", help="Width of square image output", required=False)
    ap.add_argument("-s", help="Use seam carving to resize images if set to true, normal resizing otherwise", action="store_true")
    ap.add_argument("-ia", help="directory of current annotations", required=True)
    ap.add_argument("-oa", help="directory of outdated annotations", required=True)



    args = vars(ap.parse_args())

    #USE_SEAM_CARVING = ap.parse_args().s
    IN_DIR, OUT_DIR = args['in'], args['out']
    IN_ANN = args['ia']
    OUT_ANN = args['oa']
    rotate(IN_DIR, OUT_DIR, IN_ANN, OUT_ANN)



#    if USE_SEAM_CARVING:
#        processImages(WIDTH, IN_DIR, OUT_DIR)
#    else:
#        processImages(WIDTH, IN_DIR, OUT_DIR, seam_carve=False)
#




