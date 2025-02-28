import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import color
from for_dataset_view import load_mnist_images,load_mnist_lables


# from torchvision import datasets, transforms is prebulit in pytorch dataset , so we can't know how it is implemented , so we use the manual way to implement it 
# dataset this is prebuilt dataset in pytorch like mist,CIFAR-10 , so it everything is coming from there , but if you want to implement your own dataset then you have to implement it manually
# tranform is used to apply some transformation on the dataset like resize, crop, normalize,etc # even we can the batch size for the SGD


