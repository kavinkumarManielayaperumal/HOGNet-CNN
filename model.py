
import torch.nn as nn
from data_loader import get_loader_hog
from Feature_Extraction import get_hog_features
import torch.nn.functional as F
from for_dataset_view import load_mnist_images,load_mnist_labels

class HOG_INN_Deep(nn.Module):
    def __init__(self,input_size,number_classes=10, hidden_size1=1024,hidden_size2=512,hidden_size3=256):
        super(HOG_INN_Deep,self).__init__()# we are inheriting the properties of the parent class
        self.fc1=nn.Linear(input_size,hidden_size1)# this is the first layer 
        self.bn1=nn.BatchNorm1d(hidden_size1)# this is the batch normalization layer for the first layer
        self.dropout1=nn.Dropout(0.3)
        self.fc2=nn.Linear(hidden_size1,hidden_size2)# this is the second layer
        self.bn2=nn.BatchNorm1d(hidden_size2)
        self.dropout2=nn.Dropout(0.3)
        self.fc3=nn.Linear(hidden_size2,hidden_size3)#this is the third layer
        self.fc4=nn.Linear(hidden_size3,number_classes)
    def forward(self,x):
        x=F.relu(self.bn1(self.fc1(x)))
        x=self.dropout1(x)
        x=F.relu(self.bn2(self.fc2(x)))
        x=self.dropout2(x)
        x=F.relu(self.fc3(x))# this is the third layer if you want batch normalization and dropout then you can add here
        x=self.fc4(x)# no activation function in the last layer
        return x

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
	
	#for images,labels in data_loader:
		#print(images.shape)
		#print(labels.shape)
		#break
		
	input_size=hog_features.shape[1]
    # this is the input size we are not use the data loader here beacuse the data loader is used for the training the mode
    # but here we are just checking the model so we are not using the data Loader , so take the input size from the frist image of the hog features 
    #it like frist image is the input size of the model , it like 1D vector then we will take the length of the vector
	number_classes=10
	model=HOG_INN_Deep(input_size,number_classes)
	print(model)
