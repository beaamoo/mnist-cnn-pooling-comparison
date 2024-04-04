# Core CNN implementation
import numpy as np
from cnn.conv import Conv3x3
from cnn.maxpool import MaxPool2
from cnn.softmax import Softmax
from cnn.averagepool import AveragePool2

# The CNN itself.
class CNN:
    
    def __init__(self, pooling='max'):
        self.conv = Conv3x3(8)
        self.softmax = Softmax(13 * 13 * 8, 10)  # 13x13x8 -> 10
        if pooling == 'avg':
            self.pool = AveragePool2()
        else:
            self.pool = MaxPool2()    
    
    def forward(self, image, label):
        '''
        Completes a forward pass of the CNN and calculates the accuracy and
        cross-entropy loss.
        - image is a 2d numpy array
        - label is a digit
        '''
        # We transform the image from [0, 255] to [-0.5, 0.5] to make it easier
        # to work with. This is standard practice.
        out = self.conv.forward((image / 255) - 0.5)
        out = self.pool.forward(out)
        out = self.softmax.forward(out)
    
        # Calculate cross-entropy loss and accuracy. np.log() is the natural log.
        loss = -np.log(out[label])
        acc = 1 if np.argmax(out) == label else 0
    
        return out, loss, acc
    
    def train(self, im, label, lr=.005):
        '''
        Completes a full training step on the given image and label.
        Returns the cross-entropy loss and accuracy.
        - image is a 2d numpy array
        - label is a digit
        - lr is the learning rate
        '''
        # Forward
        # We transform the image from [0, 255] to [-0.5, 0.5] to make it easier
        # to work with. This is standard practice.
        out, loss, acc = self.forward(im, label)
    
        # Calculate initial gradient
        gradient = np.zeros(10)
        gradient[label] = -1 / out[label]
    
        # Backprop
        gradient = self.softmax.backprop(gradient, lr)
        gradient = self.pool.backprop(gradient)
        gradient = self.conv.backprop(gradient, lr)
    
        return loss, acc
    
    def predict(self, image):
        '''
        Predicts the digit in an image.
        - image is a 2d numpy array
        '''
        # Forward pass
        out = self.conv.forward((image / 255) - 0.5)
        out = self.pool.forward(out)
        out = self.softmax.forward(out)

        # Output is a list of probabilities of each digit
        return np.argmax(out)
    
    def test(self, test_data):
        '''
        Tests the CNN on all test data and returns the accuracy.
        - test_data is a list of tuples of the form (im, label).
        '''
        correct = 0
    
        for im, label in test_data:
            # Predict the image
            prediction = self.predict(im)
            if prediction == label:
                correct += 1
    
        return correct / len(test_data)
    
    def stats(self, test_data):
        '''
        Tests the CNN on all test data and returns the accuracy and loss.
        - test_data is a list of tuples of the form (im, label).
        '''
        loss = 0
        correct = 0
    
        for im, label in test_data:
            # Predict the image
            out, l, acc = self.forward(im, label)
            loss += l
            correct += acc
    
        return loss / len(test_data), correct / len(test_data)
    
    def train_batch(self, data, lr, size):
        '''
        Trains the CNN on all data in a batch.
        - data is a list of tuples of the form (im, label).
        - lr is the learning rate
        - size is the batch size
        '''
        loss = 0
        correct = 0
        batches = 0
    
        for i in range(0, len(data), size):
            # Get the batch
            batch = data[i:i+size]
    
            # Train
            for im, label in batch:
                l, acc = self.train(im, label, lr)
                loss += l
                correct += acc
    
            batches += 1
    
        return loss / batches, correct / len(data)
    
    def train_epochs(self, data, lr, epochs, size):
        '''
        Trains the CNN on all data for a number of epochs.
        - data is a list of tuples of the form (im, label).
        - lr is the learning rate
        - epochs is the number of epochs to train for
        - size is the batch size
        '''
        for i in range(epochs):
            loss, acc = self.train_batch(data, lr, size)
            print('--- Epoch %d ---' % (i + 1))
            print('Loss:', loss)
            print('Accuracy:', acc)
            print()
    
        print('Training complete.')

    def save(self, filename):
        np.save(filename, {
            'weights': self.conv.weights,
            'biases': self.conv.biases
            # Add other parameters as necessary
        })
    
    @staticmethod
    def load(filename):
        data = np.load(filename, allow_pickle=True).item()
        cnn = CNN()
        cnn.conv.weights = data['weights']
        cnn.conv.biases = data['biases']
        # Add other parameters as necessary
        return cnn
