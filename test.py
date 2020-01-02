import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import torch
from models import AlexNet
from torch.autograd import Variable

import warnings
warnings.simplefilter("ignore")


# Detect all faces in an image
# load in a haar cascade cassifier fro detecting frontal faces
face_cascade = cv2.CascadeClassifier(
    './detectors/haarcascade_frontalface_default.xml')


net = AlexNet()
# loading the best saved model parameters
net.load_state_dict(torch.load(
    './saved_models/keypoints_model_AlexNet_50epochs.pth'))

# perepate the net for testing mode
net.eval()


# image = cv2.imread('imgs/10.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


faces = face_cascade.detectMultiScale(image, 1.2, 2)

# make a copy of the original image to plot detections on
image_with_detections = image.copy()

# loop over the detected faces
for (x,y,w,h) in faces:
    # draw a rectangle around each detected face
    cv2.rectangle(image_with_detections, (x,y), (x+w, y+h), (255, 0, 0), 3)
    plt.imshow(image_with_detections)
    plt.show()




# loading in the trained model



image_copy = np.copy(image)


def show_all_keypoints(image, keypoints):
    """
    Visuzlizing the image and the keypoints on it.
    """
    plt.figure(figsize=(5, 5))

    keypoints = keypoints.data.numpy()
    # Becuase of normalization, keypoints won't be placed if they won't reutrn to values before noramlization
    keypoints = keypoints * 48.0 + 48
    # reshape to 2 X 68 keypoint for the fase
    keypoints = np.reshape(keypoints, (68, -1))

    image = image.numpy()
    # Convert to numpy image shape (H x W x C)
    image = np.transpose(image, (1, 2, 0))
    image = np.squeeze(image)
    plt.imshow(image, cmap='gray')
    plt.scatter(keypoints[:, 0], keypoints[:, 1], s=40, marker='.', c='m')

    # plt.show()




# loop over the detected faces from your cascade
for (x,y,w,h) in faces:
    # select the region of interest that is the face in the image
    roi = image_copy[y-50:y+h+50, x-40:x+w+40]
    width_roi = roi.shape[1] # needed for scaling points
    height_roi = roi.shape[0] # needed for scaling points
    # Make a copy from roi to be used as background to display final keypoints.
    roi_copy = np.copy(roi)

    # convert to grayscale
    roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

    # Normalize the grayscale image
    roi = roi/255.0

    # rescale the detected face to be the expected square size
    roi = cv2.resize(roi, (227, 227))

    roi = np.expand_dims(roi, 0)
    roi = np.expand_dims(roi, 0)
    print(roi.shape)

    roi_torch = Variable(torch.from_numpy(roi))

    roi_torch = roi_torch.type(torch.FloatTensor)

    keypoints = net(roi_torch)
    keypoints = keypoints

    show_all_keypoints(roi_torch.squeeze(0), keypoints)


cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # add this
    # image, reject levels level weights.
    object = object_cascade.detectMultiScale(gray, 50, 50)

    # add this
    for (x, y, w, h) in object:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)

    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
