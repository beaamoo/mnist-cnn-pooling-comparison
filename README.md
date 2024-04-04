# CNN on MNIST Dataset

This project involves training a Convolutional Neural Network (CNN) from scratch on the MNIST dataset to recognize handwritten digits. It explores the impact of different pooling methods (Max Pooling vs. Average Pooling) on the model's performance.

### Max Pooling
- **Strengths**: Max pooling is effective at capturing the presence of features in small regions of the input image, making the detection of features somewhat invariant to scale and orientation changes. It tends to be more efficient for highlighting the most significant features in an image.
- **Use Cases**: Max pooling is often preferred in models where the exact location of features isn't as important as their rough approximation or presence, which is useful in many classification tasks.
### Average Pooling
- **Strengths**: Average pooling reduces the spatial dimensions by calculating the average value of each patch on the feature map. This method is good at preserving background features and can be beneficial in cases where the uniformity of the feature map is important.
- **Use Cases**: Average pooling can be advantageous in scenarios where it's important to maintain the smoothness of features, such as in texture classification or areas where the background information is valuable.


## Getting Started

### Download Dataset

Download the MNIST dataset from [Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) and extract the files into a directory named `data` at the root of this project.

### Install Dependencies

Ensure you have Python installed on your system. Then, install the required dependencies by running:

```bash
pip install -r requirements.txt
```

### Run Tests for Layers
To verify the implementation of CNN layers, execute the layer tests using:

```bash
python -m tests.test_layers
```

### Train the CNN on MNIST Dataset
You can train the CNN using either Max Pooling or Average Pooling by specifying the --pooling argument. Run the following commands in the terminal:

For Max Pooling:

```bash
python train.py --pooling max
```
For Average Pooling:

```bash
python train.py --pooling avg
```

## Results
After running the training scripts, the terminal will display the test loss and accuracy for each model. 

- Max Pooling:
    - Test Loss: 0.1403337233763449
    - Test Accuracy: 0.9569
- Average Pooling
    - Test Loss: 0.31134562592595033
    - Test Accuracy: 0.9096

These metrics provide insight into how each pooling method impacts the CNN's ability to recognize handwritten digits from the MNIST dataset.


