from __future__ import annotations
from winreg import KEY_NOTIFY
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow.keras as keras
import numpy as np
import argparse
import cv2
from PIL import Image
from PIL import ImageDraw
import math
import os
import xmltodict
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def selectiveSearch(img):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()  #Uses selective search to identify Regions of Interest
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()
    #ss.switchToSelectiveSearchQuality()
    rects = ss.process()
    #rects contain x, y, width and height
    return rects


def drawImageWithRects(image, rects):
    fig, ax = plt.subplots()
    ax.imshow(image)
    count = 0
    for rect in rects:
        if count > 10:
            break
        rect = patches.Rectangle((rect[0], rect[1]), rect[2], rect[3], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        count += 1
    plt.show()


def getRoI(img, rects):
    (H, W) = img.shape[:2]
    proposals = []
    for (x, y, w, h) in rects:
        if w / float(W) > 0.5 or h / float(H) > 0.5:
            continue
        # extract the region from the input image, convert it from BGR to
        # RGB channel ordering, and then resize it to 224x224 (the input
        # dimensions required by our pre-trained CNN)
        roi = img[y:y + h, x:x + w]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(roi, (224, 224))
        # further preprocess by the ROI
        roi = img_to_array(roi)
        roi = preprocess_input(roi)
        # update our proposals and bounding boxes lists
        proposals.append(roi)


def arrayRemove(array,indices):
    result = []
    for i in range(len(array)):
        if i not in indices:
            result.append(array[i])
    return np.array(result)



def getModel():
    model = ResNet50(weights="imagenet",include_top = False)
    for layer in model.layers[:143]:
        layer.trainable = False
    #for i, layer in enumerate(model.layers):
    #    print(i, layer.name, "-", layer.trainable)
    x = model.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(1024, activation='relu')(x)
    predictions = keras.layers.Dense(2, activation='softmax')(x)
    return model


def importAnnotations(annotationPath, imageName):
    boundingBoxList = []
    if len(os.listdir(annotationPath)) == 0:
        print("(importAnnotations) ERROR EMPTY ANNOTATION DIRECTORY")
    #read bb from xml to BoundingBoxes[]
    file =  open(annotationPath + imageName + ".xml")
    if file == None:
        print("ERROR FAILED TO OPEN ANNOTATION FILE")
        return None
    file_data = file.read()
    dict_data = xmltodict.parse(file_data)
    boundingBoxesRaw = []
    if "object" in dict_data["annotation"].keys():
        if str(type(dict_data["annotation"]["object"])) == "<class 'collections.OrderedDict'>":
            boundingBoxesRaw.append(dict_data["annotation"]["object"]["bndbox"])
        elif str(type(dict_data["annotation"]["object"])) == "<class 'list'>":
            for i in range(len(dict_data["annotation"]["object"])):
                boundingBoxesRaw.append(dict_data["annotation"]["object"][i]["bndbox"])
    #print(filename + str(boundingBoxesRaw))
    BoundingBoxes = []
    for i in range(len(boundingBoxesRaw)):
        x1 = float(boundingBoxesRaw[i]["xmin"])
        y1 = float(boundingBoxesRaw[i]["ymin"])
        x2 = float(boundingBoxesRaw[i]["xmax"])
        y2 = float(boundingBoxesRaw[i]["ymax"])
        BoundingBoxes.append([x1, y1, x2, y2])
    #boundingBoxes come in xyxy
    if len(BoundingBoxes) == 0:
        return None
    else:
        return BoundingBoxes


def IoU(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    #print(boxA)
    #print(boxB)

    if (boxA[2] < boxB[0] or boxA[0] > boxB[2] or boxA[3] < boxB[1] or boxA[1] > boxB[3]):
        return 0


    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs((xB - xA) * (yB - yA))

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs(boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = abs(boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def xyxy2xywh(x1,y1,x2,y2):
    if x1 <= x2 and y1 <= y2:
        return x1,y1,x2-x1,y2-y1
    elif x1 <= x2 and y1 > y2:
        return x1,y2,x2-x1,y1-y2
    elif x1 > x2 and y1 > y2:
        return x2,y2,x1-x2,y1-y2
    elif x1 > x2 and y1 <= y2:
        return x2,y1,x1-x2,y2-y1
    else:
        print("ERROR: xyxy2xywh exeptional case!")


def xywh2xyxy(x,y,w,h):
    return x,y,x+w,y+h


def NMS(boxes, overlapThresh):
    #print("Executing NMS")
    result = []
    for i in range(len(boxes)):
        indexes_to_delete = []
        if i>=len(boxes):
            break
        result.append(boxes[i])
        for j in range(i+1,len(boxes)):
            boxesXY1 = xywh2xyxy(boxes[i][0],boxes[i][1],boxes[i][2],boxes[i][3])
            boxesXY2 = xywh2xyxy(boxes[j][0],boxes[j][1],boxes[j][2],boxes[j][3])
            if IoU(boxesXY1,boxesXY2) > overlapThresh:
                indexes_to_delete.append(j)
        if indexes_to_delete == []:
            continue
        boxes = arrayRemove(boxes,indexes_to_delete)
    return result


print("Converting Pascal VOC annotations to Yolo v1.1")
startTime = time.time()
imageCount = 0

annots = os.listdir("annotations/")
for filename in annots:
    boxes = importAnnotations("annotations/",filename[:-4])
    f = open("TEMP_new_annot/" + filename[:-3] + "txt","w")
    if boxes != None:
        for box in boxes:
            x = ((box[0] + box[2]) / 2) / 512
            y = ((box[1] + box[3]) / 2) / 512
            w = (box[2] - box[0]) / 512
            h = (box[3] - box[1]) / 512
            f.write("0 " + str(x) + " " + str(y) + " " + str(w) + " " + str(h) + "\n")
    f.close()
    imageCount+=1
    print("{:.2f}".format(imageCount/len(annots)*100) + "% (" + str(imageCount) + "/" + str(len(annots)) + ")", end = '\r')
print("DONE!")














"""
model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9),
loss='binary_crossentropy', metrics=['acc'])
model.fit()
"""