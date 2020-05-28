import mxnet as mx
from mxnet import gluon,nd,autograd,init
from tqdm import trange
from mxnet import metric
import prepare_data
from model import network

num_workers  = mx.context.num_gpus()
if num_workers:
  context = mx.gpu(0)
else:
  context = mx.cpu(0) 
  
def training_step(inputs,labels,optimizer,loss_fn,network,wo,ho,batch_size,flag=0):
    with autograd.record():
        outputs = network(inputs)
        outputs = outputs.transpose((0,2,3,1))
        first_dim = outputs.shape[0]
        outputs = outputs.reshape(first_dim*wo*ho,2)
        labels = labels.reshape(first_dim*wo*ho)
        loss = loss_fn(outputs,labels)
    if flag ==0:
      loss.backward()
      optimizer.step(batch_size)
      return network,loss
    elif flag==1:
      return labels,outputs

def validate(network,data,wo,ho,loss_fn,flag=0):
  _,_,val_X,val_Y = data
  val_X = mx.nd.array(val_X,ctx=context).astype('float32')
  val_Y = mx.nd.array(val_Y,ctx=context).astype('long')
  dim_1  = val_X.shape[0]
  outputs = net(val_X)
  outputs = outputs.reshape(dim_1*wo*ho,2)
  labels = val_Y.reshape(dim_1*wo*ho)
  if flag ==0:
    loss = loss_fn(outputs,labels)
    return loss.mean().asscalar()
  elif flag ==1:
    return labels,outputs

def training_full(network,batch_size,epochs,loss_fn,optimizer,data,wo,ho):
    train_features,train_labels,val_features,val_labels = data
    num_epochs = train_features.shape[0]//batch_size
    train_acc = metric.Accuracy()
    val_acc = metric.Accuracy()
    t = trange(epochs, leave=True)
    for e in t:
      for i in range(num_epochs):
        final_loss = 0
        batch_X = mx.nd.array(train_features[i*batch_size:(i+1)*batch_size],ctx=context).astype('float32')
        batch_Y = mx.nd.array(train_labels[i*batch_size:(i+1)*batch_size],ctx=context).astype('long')
        flag=0
        network,loss = training_step(batch_X,batch_Y,optimizer,loss_fn,network,wo,ho,batch_size,flag)
        final_loss +=loss.mean().asscalar()
        flag=1
        l,o = training_step(batch_X,batch_Y,optimizer,loss_fn,network,wo,ho,batch_size,flag)
        train_acc.update(l,o)
      validation_loss = validate(network,data,wo,ho,loss_fn)
      l1,o1 = validate(network,data,wo,ho,loss_fn,flag=1)
      val_acc.update(l1,o1)
    return network, final_loss,train_acc.get()[1],validation_loss,val_acc.get()[1]

(wi,hi),(wo,ho) = (284,284),(196,196)
data=()
data = prepare_data.load_dataset(wi, hi, wo, ho)

train_features = data[0]
train_labels = data[1]
val_features = data[2]
val_labels = data[3]
train_features = train_features.astype('float32')
val_features = val_features.astype('float32')
data = train_features,train_labels,val_features,val_labels

net = network(input_channels=1,output_channels=2)
net.initialize()
net.summary(nd.ones((5,1,284,284)))

batch_size = 6
epochs = 100
net = network(input_channels=1,output_channels=2)
net.initialize(init=init.Xavier(),ctx=context)
loss_fn = mx.gluon.loss.SoftmaxCrossEntropyLoss()
optimizer =gluon.Trainer(net.collect_params(),optimizer='sgd',optimizer_params={'learning_rate':0.0004})

nnet, loss,acc,validation_loss,validation_accuracy = training_full(net, batch_size, epochs, loss_fn, optimizer, data, wo,ho)


print("Total loss over 100 epochs = ",loss)
train_loss = loss/epochs
print("Mean train loss per epoch = ",train_loss)
print("Training accuracy = ", acc)
print("Validation loss = ", validation_loss)
print("Validation accuracy = ", validation_accuracy)

file_name = "net.params"
nnet.save_parameters(file_name)
