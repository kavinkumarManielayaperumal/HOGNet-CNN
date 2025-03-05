import torch
from torch.utils.data import Dataset, DataLoader
from Feature_Extraction import get_hog_features
from for_dataset_view import load_mnist_images,load_mnist_labels
import numpy as np

class MNIST(Dataset):
    def __init__(self,images,labels,tranformas=None):
        self.images=torch.tensor(images,dtype= torch.float32)
        self.labels=torch.tensor(np.array(labels),dtype=torch.long)
        self.tranformas=tranformas
    def __len__(self):
        return len(self.images)
    def __gettiem__(self,idx):
        images=self.images[idx]
        labels=self.labels[idx]
        if self.tranformas is not None:
            images=self.tranformas(images)
        return images, labels

def get_loader_hog(images,labels,batch_size=32):
    data= MNIST(images,labels)
    data_loader=DataLoader(data,batch_size=batch_size,shuffle=True)
    return data_loader
        
        
if __name__=="__main__":
	train_images_path = r"E:\for practice game\simplenet_CNN\data\train-images-idx3-ubyte\train-images.idx3-ubyte"
	train_labels_path = r"E:\for practice game\simplenet_CNN\data\train-labels-idx1-ubyte\train-labels.idx1-ubyte"
	
	images=load_mnist_images(train_images_path)
	labels=load_mnist_labels(train_labels_path)
	print(f"Iamges shape:{images.shape}")
	print(f"Labels shape:{labels.shape}")
hog_features,hog_images=get_hog_features(images)
print(hog_features.shape)
print(hog_images.shape)
data_loader=get_loader_hog(hog_features,labels,batch_size=32)

for images,labels in data_loader:
    print(images.shape)
    print(labels.shape)
    break
    