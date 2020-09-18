from pyfingerprintdetector.localbinarypatterns import LocalBinaryPatterns
from sklearn.svm import LinearSVC
from imutils import paths
from sklearn.metrics import precision_score, recall_score, accuracy_score
import cv2
import os

training = "images\\training"
testing = "images\\testing"
# initialize the local binary patterns descriptor along with
# the data and label lists
desc = LocalBinaryPatterns(24, 8)
data = []
labels = []
# loop over the training images
for imagePath in paths.list_images(training):
    # load the image
    image = cv2.imread(imagePath)
    # Convert the image into greyscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Describe the image, histogram is computed using the method in localbinarypatterns.py
    hist = desc.describe(gray)
    # extract the label from the image path
    # Append all labels of training images and update the data lists
    labels.append(imagePath.split(os.path.sep)[-2])
    data.append(hist)
    
# train a Linear SVM on the data
model = LinearSVC(C=100.0, random_state=42, max_iter=10000)
model.fit(data, labels)

y_test = []
y_pred = []
count = 0
# loop over the testing images
for imagePath in paths.list_images(testing):
    # load the image, convert it to grayscale, describe it,
    # and classify it
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = desc.describe(gray)
    prediction = model.predict(hist.reshape(1, -1))
    # Extract the label from imagePath and update
    actual_label = imagePath.split(os.path.sep)[-2]
    # Update all the actual labels of test data to y_test
    y_test.append(actual_label)
    # Update all the predicted values to y_pred
    y_pred.append(prediction[0])

    #if actual_label=='Spoof' and prediction[0]=='Live':
     #   count +=1
precision = precision_score(y_test, y_pred, pos_label='Live')
print('Precision = ',precision)

recall = recall_score(y_test, y_pred, pos_label='Live')
print('Recall = ',recall)

print('Accuracy score = ',accuracy_score(y_test, y_pred))
#print(count)