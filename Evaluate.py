import torch 
from model import HOG_INN_Deep
from for_dataset_view import load_mnist_images,load_mnist_labels
from Feature_Extraction import get_hog_features
from data_loader import get_loader_hog
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import torch.nn.functional as F
import numpy as np

def load_model(filename="hog_model.pth",input_size=None,numbers_classes=10):
    model=HOG_INN_Deep(input_size,numbers_classes)
    model.load_state_dict(torch.load(filename))
    model.eval()
    return model

def evaluate_model(test_hog_features,test_labels):
    with torch.no_grad():
        model=load_model(filename="hog_model.pth",input_size=hog_features.shape[1],numbers_classes=10)
        output=model(test_hog_features)
        _ , predicted=torch.max(output,dim=1)
        y_ture=test_labels.numpy()
        y_pred=predicted.numpy()
    return y_ture,y_pred



test_images_path=r"E:\for practice game\simplenet_CNN\data\train-images-idx3-ubyte\train-images.idx3-ubyte"
test_labels_path=r"E:\for practice game\simplenet_CNN\data\train-labels-idx1-ubyte\train-labels.idx1-ubyte"

images=load_mnist_images(test_images_path)
labels=load_mnist_labels(test_labels_path)
print(f"Images shape:{images.shape}")
print(f"Labels shape:{labels.shape}")

hog_features,hog_images=get_hog_features(images)
print(hog_features.shape)




test_hog_features=torch.tensor(hog_features,dtype=torch.float32)
test_labels=torch.tensor(labels,dtype=torch.long)
print(test_hog_features.shape)

y_true,y_pred=evaluate_model(test_hog_features,test_labels)

accuracy = (y_pred == y_true).sum() / len(y_true) * 100
print(f"Test Accuracy: {accuracy:.2f}%")


print("classification report")
print(classification_report(y_true,y_pred))

np.savetxt("predictions.txt", y_pred, fmt="%d")
print("Predictions saved to predictions.txt")
