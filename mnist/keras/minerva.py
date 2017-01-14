#!/usr/bin/env python

import operator as op
import numpy as np

from keras.datasets import mnist
from keras.layers import Activation, Dense
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.utils import np_utils

layers = [784, 256, 256, 256, 10]
weightsfile = "mnist.weights"

def PreprocessCategorical(xs, ys, num_classes):
  processed_x = []
  # For both training and testing datasets, flatten, convert to floats, and
  # normalize.
  for x in xs:
    x = x.reshape((x.shape[0], reduce(op.mul, x.shape[1:])))
    x = x.astype("float32")
    mu = np.mean(x, axis=1, keepdims=True)
    std = np.std(x, axis=1, keepdims=True)
    x = (x - mu) / std
    processed_x.append(x)

  processed_y = []
  # For both training and testing labels, convert to categorical (one hot)
  # encoding (for computing the loss function).
  for y in ys:
    y = np_utils.to_categorical(y, num_classes)
    processed_y.append(y)

  return processed_x, processed_y

def LoadData():
  training, testing = mnist.load_data()
  return PreprocessCategorical(*zip(training, testing), num_classes=10)

def CreateModel():
  model = Sequential()
  for insize, outsize in zip(layers[0:-2], layers[1:-1]):
    model.add(Dense(input_dim=insize, output_dim=outsize))
    model.add(Activation("relu"))
  # Last layer uses softmax activation.
  model.add(Dense(input_dim=layers[-2], output_dim=layers[-1], activation="linear"))
  model.add(Activation("softmax"))

  rms = RMSprop()
  model.compile(loss="categorical_crossentropy", optimizer=rms, metrics=["accuracy"])
  return model

def main():
  print "Creating model"
  model = CreateModel()

  print "Loading and preprocessing MNIST data."
  ((xtrain, xtest), (ytrain, ytest)) = LoadData()

  print "Training model"
  model.fit(xtrain, ytrain, nb_epoch=25, batch_size=150)

  print "Saving weights"
  model.save_weights(weightsfile)
  # model.load_weights(weightsfile)

  print "Running test set"
  (loss, accuracy) = model.evaluate(xtest, ytest)
  print "\nAccuracy is %0.4f" % accuracy

if __name__ == "__main__":
  main()
