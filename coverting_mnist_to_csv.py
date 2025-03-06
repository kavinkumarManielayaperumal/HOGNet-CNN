import numpy as np
import pandas as pd
from Feature_Extraction import get_hog_features
from for_dataset_view import load_mnist_images,load_mnist_labels


def convert_to_csv(images,labels):
    hog_features,hog_images=get_hog_features(images)
    pf=pd.DataFrame(hog_features)
    pf['labels']=pd.DataFrame(labels)
    pf.to_csv("mnist_hog_features.csv",index=False)
    
    
if __name__=="__main__":
	train_images_path = r"E:\for practice game\simplenet_CNN\data\train-images-idx3-ubyte\train-images.idx3-ubyte"
	train_labels_path = r"E:\for practice game\simplenet_CNN\data\train-labels-idx1-ubyte\train-labels.idx1-ubyte"
	
	images=load_mnist_images(train_images_path)
	labels=load_mnist_labels(train_labels_path)
	print(f"Iamges shape:{images.shape}")
	print(f"Labels shape:{labels.shape}")
	
	convert_to_csv(images,labels)
	
	df=pd.read_csv("mnist_hog_features.csv")
	print(df.head())
	print(df.shape)
	
	print("Done")