import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import color
from for_dataset_view import load_mnist_images,load_mnist_labels


# from torchvision import datasets, transforms is prebulit in pytorch dataset , so we can't know how it is implemented , so we use the manual way to implement it 
# dataset this is prebuilt dataset in pytorch like mist,CIFAR-10 , so it everything is coming from there , but if you want to implement your own dataset then you have to implement it manually
# tranform is used to apply some transformation on the dataset like resize, crop, normalize,etc # even we can the batch size for the SGD

def get_hog_features(images):
    
    
    hog_features=[]
    hog_images=[]
    for image in images:
        if len(image.shape)==3 and image.shape[2]==3:
            gray_image=color.rgb2gray(image)
        else:
            gray_image=image
        
        hog_feature, hog_image=hog(gray_image,orientations=9,pixels_per_cell=(8,8),cells_per_block=(2,2),visualize=True,block_norm="L2-Hys")# this will return the hog features and hog image , like we will get output hog features its in vector form , then we will covert into array and hog images is matrix 
        hog_features.append(hog_feature) # now its 1D array
        hog_images.append(hog_image)
    return np.array(hog_features) ,np.array(hog_images)


if __name__ =="__main__":
    
	train_images_path = r"E:\for practice game\simplenet_CNN\data\train-images-idx3-ubyte\train-images.idx3-ubyte"
	train_labels_path = r"E:\for practice game\simplenet_CNN\data\train-labels-idx1-ubyte\train-labels.idx1-ubyte"
	
	images=load_mnist_images(train_images_path)
	labels=load_mnist_labels(train_labels_path)
	print(f"Iamges shape:{images.shape}")
	print(f"Labels shape:{labels.shape}")
	
	hog_features,hog_images=get_hog_features(images)# this will return the hof features and hog images 

	
	print(hog_features.shape)
	print(hog_images.shape)
	
	plt.imshow(images[4], cmap="gray")# this also directly shown from the dataset
	plt.title(f"Label: {labels[4]}")  # Show the label on top
	plt.axis("on")
	plt.colorbar()
	plt.show()
	
	plt.plot(hog_features[4])
	plt.show()
 
	plt.imshow(hog_images[4],cmap="gray")
	plt.title("HOG image")
	plt.axis("off")
	plt.show()

	
