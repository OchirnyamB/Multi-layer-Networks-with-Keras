# Recognizaing handwritten digits with MNIST dataset using Keras library
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
# The train_test_split will be used to create our training and testing splits from the MNIST dataset.
from sklearn.model_selection import train_test_split
# Gives us a nicely formatted report displaying the total accuracy of our model
# along with a breakdown on the classification accuracy for each digit 
from sklearn.metrics import classification_report
from keras.datasets import cifar10
# The Sequential class indicates that our network will feedforward
# and layers will be added to the class sequentially, one on top of another
from keras.models import Sequential
from keras.layers.core import Dense
# For our network to actually learn, we need to apply SGD
# to optimize the parameters of the network
from keras.optimizers import SGD
import argparse

# Construct the argument parse and parse the arguments
# python kerasMNIST.py --output output/kerasMNIST.png, to run the python code from bash or terminal
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())

print("[INFO] loading CIFAR-10 data...")
# CLoading the training and testing data, scale it into the range [0.1] then reshape the design matrix
# presegmented training and testing split data
((trainX, trainY), (testX, testY)) = cifar10.load_data()

trainX = trainX.astype("float")/255.0
testX = testX.astype("float")/255.0
trainX = trainX.reshape((trainX.shape[0], 3072))
testX = testX.reshape((testX.shape[0], 3072))

# label 9 = [0,0,0,0,0,0,0,0,0,1] -> one-hot encoding
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

labelNames = ["airplane", "automobile", "bird", "cat", "deer",
"dog", "frog", "horse", "ship", "truck"]

# Define the 3072-1024-512-10 architecture using Keras
model = Sequential()
model.add(Dense(1024, input_shape=(3072,), activation="relu"))
model.add(Dense(512, activation="relu"))
model.add(Dense(10, activation="softmax"))

# Train the model using SGD
print("[INFO] training network...")
sgd = SGD(0.01)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics="accuracy")
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=100, batch_size=32)

# Evaluate the network
print("[INFO] evaluating network...")
# This would have the shape (X,10) as there are 17,500 total data points in the testing set and then possible class labels
# Batch sizes of powers of two are often desirable because they allow internal linear algebra optimization libraries to be more efficient.
predictions = model.predict(testX, batch_size=32)
# Each entry in a given row in predictions, is a probability due to softmax function
# argmax(1) will give us the index of the class label with the largets probability, aka our final output classification
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))

# Plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0,100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0,100), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0,100), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])