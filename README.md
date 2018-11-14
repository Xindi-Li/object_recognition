# Requirements
python 3.6

pip3 install sklearn

pip3 install tensorflow

pip3 install opencv-python

pip3 install imutils

Dataset: http://www.vision.caltech.edu/Image_Datasets/Caltech256/

# How to run 
move the images into "training_images" directory

choose several test images and move them into "test_images" directory

run retrain.py

Then run test.py to see the results

# Result

## KNN
3  neighbors accuracy: 12.61%
4  neighbors accuracy: 12.94%
5  neighbors accuracy: 12.98%
6  neighbors accuracy: 13.13%
7  neighbors accuracy: 13.27%
8  neighbors accuracy: 13.53%
9  neighbors accuracy: 13.33%
10  neighbors accuracy: 13.55%

## SVM
linear  accuracy: 16.23%
poly  accuracy: 16.05%
rbf  accuracy: 0.74%
sigmoid  accuracy: 0.65%

## transfer learning
83.9%
