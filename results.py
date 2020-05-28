from model import network
import prepare_data
import mxnet as mx
import numpy as np
import matplotlib.pyplot as plt

num_workers  = mx.context.num_gpus()
if num_workers:
  context = mx.gpu(0)
else:
  context = mx.cpu(0) 
  
mynet = network(input_channels=1,output_channels=2)
mynet.load_parameters('./net.params', ctx=context)

def show_results(network,features,labels,examples=4):
  figure,axis = plt.subplots(nrows=4,ncols=4,figsize=(15,4*examples))
  dim1 = features.shape[0]
  for row in range(examples):
    img_idx = np.random.randint(dim1)
    image_array = network(mx.nd.array(features[img_idx:img_idx+1],ctx=context).astype('float32')).squeeze(0).asnumpy()
    axis[row][0].imshow(np.transpose(features[img_idx], (1,2,0))[:,:,0])
    axis[row][1].imshow(np.transpose(image_array, (1,2,0))[:,:,0])
    axis[row][2].imshow(image_array.argmax(0))
    axis[row][3].imshow(np.transpose(labels[img_idx], (1,2,0))[:,:,0])
  plt.show()
  


(wi,hi),(wo,ho) = (284,284),(196,196)
data=()
data = prepare_data.load_dataset(wi, hi, wo, ho)

train_features = data[0]
train_labels = data[1]
val_features = data[2]
val_labels = data[3]
train_features = train_features.astype('float32')
val_features = val_features.astype('float32')


print('Enter 0 for running the model on training examples')
print('Enter 1 for running the model on validation data')

a = int(input())
if a==0:
    show_results(mynet,train_features,train_labels)
elif a==1:
    show_results(mynet,val_features,val_labels)
    
