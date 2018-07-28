# check this amazing tutorial for more details:
# https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/

import numpy as np
import cv2

min_confidence = 0.6

# path to Caffe 'deploy' prototxt file
prototxt_path = './dl_face_detection/deploy.prototxt.txt'

# path to Caffe pre-trained model
model_path = './dl_face_detection/res10_300x300_ssd_iter_140000.caffemodel'

# load our serialized model from disk
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# using deep learning model to detect the face


def get_face_from_img(image):
    # add paddings to make the image square
    (h, w) = image.shape[:2]

    pad_right, pad_bottom = (0, 0)

    # pad right if image is too tall
    if h > w:
        pad_right = h - w

    # pad right if image is too wide
    if w > h:
        pad_bottom = w - h

    if pad_right or pad_bottom:
        color = [0, 0, 0]
        image = cv2.copyMakeBorder(image, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT,
                                   value=color)
        w += pad_right
        h += pad_bottom

    # construct an input blob for the image
    # by resizing to a fixed 300x300 pixels and then normalizing it
    resized_image = cv2.resize(image, (300, 300))
    blob = cv2.dnn.blobFromImage(resized_image, 1.0, (300, 300))

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > min_confidence:
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            return (startX, startY, endX - startX, endY - startY)

    return None
