import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_loader_hog
from Feature_Extraction import get_hog_features
from for_dataset_view import load_mnist_images,load_mnist_labels
from model import HOG_INN_Deep
import matplotlib.pyplot as plt

device=torch.device("cuda" if torch.cuda.is_available()else"cpu")
print(f"Device we are using:{device}")

def train_model(model,data_loader,number_epochs=5,lr=0.001):
    model.to(device)
    criterion=nn.CrossEntropyLoss()# for the classification problem
    optimizer=optim.Adam(model.parameters(),lr=lr)
    training_loss=[]
    
    for epoch in range(number_epochs):
        total_loss=0
        for batch_idx,(images,labels) in enumerate(data_loader):# enumerate is used to get the indef of the batch for debugging
            images=images.to(device)
            labels=labels.to(device)
            optimizer.zero_grad() # its like negative gradient , its say derivative is equal to zero , so its lead to the zero gradient
            outputs=model(images)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()
        avg_loss=total_loss/len(data_loader)
        training_loss.append(avg_loss)
        
        print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {avg_loss:.4f}")# 4f means 4 decimal point 
    plt.plot(range(1, number_epochs + 1), training_loss, marker='o', linestyle='-')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.show()
        
    save_model(model,file_name="hog_model.pth")
    
    
def save_model(model,file_name="hog_model.pth"):
    torch.save(model.state_dict(),file_name)
    print(f"Model saved as {file_name}")
    
    
if __name__=="__main__":
    
	train_images_path = r"E:\for practice game\simplenet_CNN\data\train-images-idx3-ubyte\train-images.idx3-ubyte"
	train_labels_path = r"E:\for practice game\simplenet_CNN\data\train-labels-idx1-ubyte\train-labels.idx1-ubyte"
	
	images1=load_mnist_images(train_images_path)
	labels1=load_mnist_labels(train_labels_path)
	print(f"Iamges shape:{images1.shape}")
	print(f"Labels shape:{labels1.shape}")
	hog_features,hog_images=get_hog_features(images1)
	print(hog_features.shape)
	print(hog_images.shape)
	data_loader=get_loader_hog(hog_features,labels1,batch_size=32)
	
	input_size=hog_features.shape[1]
	number_classes=10
	model=HOG_INN_Deep(input_size,number_classes)
	train_model(model,data_loader,number_epochs=5,lr=0.001)
	print("Done")
	