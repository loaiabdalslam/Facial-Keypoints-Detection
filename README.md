# Facial-Keypoints-Detection


[image1]: ./img.jpg "input"
[image2]: ./fig2.png "keypoint detection"
[image3]: ./fig1.png "glasses"
[image4]: ./fig3.png "hat"

[image3]: ./images/mnist.PNG "mnist Output"
[image4]: ./images/faces.PNG "CelebA Output"



input image          | keypoints detection               :| adding sunglasses         | Adding Hat
:-------------------------:|:----------------------------:|:-------------------------:|:-------------------:|
![input][image1]           |![keypoint detection][image2] |![glasses][image3]         |![hat][image4]



In this project, I combined computer vision techniques and deep learning architectures to build a facial keypoint detection system. Facial keypoints include points around the eyes, nose, and mouth on a face and are used in many applications. 

**These applications include:**
- Facial tracking
- Facial pose recognition
- Facial filters
- Emotion recognition. 

## libraries and framework
- PyTorch
- OpenCV
- matplotlib
- pandas
- numpy
- pillow
- scipy

to install a few required pip packages, which are specified in the requirements text file:

`pip install -r requirements.txt`


## Dataset
All of the data needed to train a neural network is in the in the subdirectory `data`. In this folder are training and tests set of image/keypoint data, and their respective csv files. 

