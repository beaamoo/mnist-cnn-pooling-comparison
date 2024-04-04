# Script to train the CNN model

import numpy as np
import argparse
from cnn.cnn import CNN
from data.mnist_loader import load_mnist

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train a CNN on MNIST with different pooling methods.')
parser.add_argument('--pooling', type=str, default='max', choices=['max', 'avg'],
                    help='The type of pooling to use: "max" for MaxPooling, "avg" for AveragePooling.')
args = parser.parse_args()

# Print the chosen pooling method
print(f"Using {args.pooling} pooling.")

# Load the MNIST dataset
train_images, train_labels = load_mnist('data', kind='train')
test_images, test_labels = load_mnist('data', kind='t10k')

# Initialize the CNN with the specified pooling method
cnn = CNN(pooling=args.pooling)

# Train the CNN
for epoch in range(1):
    print('--- Epoch %d ---' % (epoch + 1))

    # Shuffle the training data
    permutation = np.random.permutation(len(train_images))
    train_images = train_images[permutation]
    train_labels = train_labels[permutation]

    # Train the CNN on each image
    loss = 0
    num_correct = 0
    for i, (im, label) in enumerate(zip(train_images, train_labels)):
        if i % 100 == 99:
            print(
                '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
                (i + 1, loss / 100, num_correct)
            )
            loss = 0
            num_correct = 0

        # Train the CNN
        im_reshaped = im.reshape(28, 28)  # Reshape to 2D array assuming 'im' is a flat array of 784 pixels (28x28 image)
        l, acc = cnn.train(im_reshaped, label)
        loss += l
        num_correct += acc

# Test the CNN
print('\n--- Testing the CNN ---')
loss = 0
num_correct = 0
for im, label in zip(test_images, test_labels):
    # Reshape the image to 2D (28x28) before passing it to the forward method
    im_reshaped = im.reshape(28, 28)
    _, l, acc = cnn.forward(im_reshaped, label)
    loss += l
    num_correct += acc

num_tests = len(test_images)
print('Test Loss:', loss / num_tests)
print('Test Accuracy:', num_correct / num_tests)

