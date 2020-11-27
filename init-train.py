#!/bin/python3

import json

def readData(f='train.json',printQuestion=False):
    rawFile = open('Annotations/'+f)
    data = json.load(rawFile)

    for elem in data:
        print('IMAGE:', elem["image"], '\nQUESTION:', elem["question"],
              'ANSWERABLE:', elem["answerable"], ' ->  ', elem["answers"][0])
        input()

if __name__ == "__main__":
    readData(printQuestion=True)

