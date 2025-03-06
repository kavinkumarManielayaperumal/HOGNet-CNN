import struct
import numpy as np
import matplotlib.pyplot as plt

def load_mnist_images(filename):
    
    # we have more mode like r, w, a, r+, w+, a+, rb, wb, ab, r+b, w+b, a+b, rb+, wb+, ab+
	# r = read, w = write, a = append, r+ = read and write, w+ = write and read, a+ = append and read and rb = read the binary file
 
    with open(filename, 'rb') as f: # with open() as automaticall closes the file when done , like f.close() and rb is read binary
        f.read(16)# 16 means first 16 bytes are not useful beacuse they are metadata which contians magic number and number of images
        raw_data = np.frombuffer(f.read(), dtype=np.uint8)# this frombuffer is used to read the binary file and convert into numpy array and dtype is used to convert into 8 bit unsigned integer
        #raw_data = np.frombuffer(f.read(28 * 28 * 10), dtype=np.uint8).reshape(10, 28, 28) # read 10 images at once
        print(f"Total bytes read: {raw_data.size}")
        images = raw_data.reshape(-1,28,28)# -1 means numpy will automatically calculate the number of images and why are we rehaping beacuse we have in 1D array everyvalue so we need to difine pixels in 2D array or image values like 28*28
    return images

def load_mnist_labels(filename):
    
    with open(filename, 'rb') as f:
         f.read(8)
         labels=np.frombuffer(f.read(), dtype=np.uint8)
         
        #f.read(8) # 8 means first 8 bytes are not useful beacuse they are metadata which contians magic number and number of images
        #lables= np.frombuffer(l.read(), dtype=np.uint8)
        #print(f"Total labels read: {labels.size}")
    return labels

# Change these paths to your actual dataset locations
train_images_path = r"E:\for practice game\simplenet_CNN\data\train-images-idx3-ubyte\train-images.idx3-ubyte"
train_labels_path = r"E:\for practice game\simplenet_CNN\data\train-labels-idx1-ubyte\train-labels.idx1-ubyte"


if __name__=="__main__":
    
    
    images=load_mnist_images(train_images_path)
    labels=load_mnist_labels(train_labels_path)
    print(f"Images shape:{images.shape}")
    print(f"Lables shape:{labels.shape}")
    
    
    #plt.imshow(images[3], cmap="gray")
    #plt.title(f"Label: {labels[3]}")  # Show the label on top
    #plt.axis("off")
    #plt.show()
