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
3  neighbors accuracy: 12.61%\
4  neighbors accuracy: 12.94%\
5  neighbors accuracy: 12.98%\
6  neighbors accuracy: 13.13%\
7  neighbors accuracy: 13.27%\
8  neighbors accuracy: 13.53%\
9  neighbors accuracy: 13.33%\
10  neighbors accuracy: 13.55%

## SVM
linear  accuracy: 16.23%\
poly  accuracy: 16.05%\
rbf  accuracy: 0.74%\
sigmoid  accuracy: 0.65%

## transfer learning
10000 training steps: 84.3%\
8000 training steps: 83.9%\
6000 training steps: 83.4%\
4000 training steps: 82.1%

## BP neural network
100   tanh  accuracy: 11.81%\
500   tanh  accuracy: 13.46%\
1000   tanh  accuracy: 13.66%\
2000   tanh  accuracy: 12.59%\
100   relu  accuracy: 12.92%\
500   relu  accuracy: 14.16%\
1000   relu  accuracy: 14.68%\
2000   relu  accuracy: 15.79%
