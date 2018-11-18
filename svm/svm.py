from sklearn.model_selection import train_test_split
from imutils import paths
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import numpy as np
import cv2
import os


def image_to_feature_vector(image, size=(32, 32)):
    # resize the image to a fixed size, then flatten the image into
    # a list of raw pixel intensities
    return cv2.resize(image, size).flatten()


DATA_SET_PATH = '../training_images'
# grab the list of images that we'll be describing
print("[INFO] describing images...")
imagePaths = list(paths.list_images(DATA_SET_PATH))

# initialize the raw pixel intensities matrix, the features matrix,
# and labels list
rawImages = []
labels = []

# loop over the input images
for (i, imagePath) in enumerate(imagePaths):
    # load the image and extract the class label (assuming that our
    # path as the format: /path/to/dataset/{class}.{image_num}.jpg
    image = cv2.imread(imagePath)
    label = imagePath.split(os.path.sep)[-2].split(".")[1]

    # extract raw pixel intensity "features", followed by a color
    # histogram to characterize the color distribution of the pixels
    # in the image
    pixels = image_to_feature_vector(image)

    # update the raw images, features, and labels matricies,
    # respectively
    rawImages.append(pixels)
    labels.append(label)

    # show an update every 1,000 images
    if i > 0 and i % 1000 == 0:
        print("[INFO] processed {}/{}".format(i, len(imagePaths)))

# show some information on the memory consumed by the raw images
# matrix and features matrix
rawImages = np.array(rawImages)
labels = np.array(labels)
print("[INFO] pixels matrix: {:.2f}MB".format(
    rawImages.nbytes / (1024 * 1000.0)))

# partition the data into training and testing splits, using 75%
# of the data for training and the remaining 25% for testing
(trainRI, testRI, trainRL, testRL) = train_test_split(
    rawImages, labels, test_size=0.15, random_state=42)

# train and evaluate a k-NN classifer on the raw pixel intensities
print("[INFO] evaluating raw pixel accuracy...")
parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
model = SVC(class_weight='balanced')
clf = GridSearchCV(model, parameters, cv=5)
clf.fit(trainRI, trainRL)
acc = clf.score(testRI, testRL)
print("[INFO] raw pixel accuracy: {:.2f}%".format(acc * 100))
print(clf.best_estimator_, '\n')
print(clf.best_score_, '\n')
print(clf.best_params_, '\n')


