import numpy as np
import os
import scipy.io
from skimage.transform import resize
from tqdm import tqdm 

dataset_name = '2015_BOE_Chiu'
def atzero(x):
    a = 0 if x==0 else 1
    return a

atzero = np.vectorize(atzero,otypes=[np.float])

def create_pipeline(paths,wi,hi,wo,ho,indexes,mat):
    x,y=[],[]
    for file_path in tqdm(paths):
        data = scipy.io.loadmat(file_path)
        images,labels = data['images'],data['manualFluid1']
        transposed_images = np.transpose(images,(2,0,1))/255.0
        resized_images = resize(transposed_images,(transposed_images.shape[0],wi,hi))
        labels = np.transpose(labels,(2,0,1))
        labels = atzero(labels)
        labels = resize(labels,(labels.shape[0],wo,ho))
        
        for index in indexes:
            x = x + [np.expand_dims(resized_images[index],0)]
            y = y + [np.expand_dims(labels[index],0)]
    return np.array(x),np.array(y)

def load_dataset(wi,hi,wo,ho):
    dir_path = os.path.join(dataset_name)
    file_path=[]
    for i in range(1,10):
        file_path.append(os.path.join(dir_path, 'Subject_0{}.mat'.format(i)))
    split = len(file_path)  
    indexes = [10, 15, 20, 25, 28, 30, 32, 35, 40, 45, 50]
    data = scipy.io.loadmat(file_path[0])
    train_features,train_labels = create_pipeline(file_path[:split-1],wi,hi,wo,ho,indexes,data)
    val_features,val_labels = create_pipeline(file_path[split-1:],wi,hi,wo,ho,indexes,data)
    return train_features,train_labels,val_features,val_labels

