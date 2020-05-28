import mxnet as mx
from mxnet import gluon,init,nd,autograd
mx.random.seed(1)

class network(gluon.nn.HybridBlock):
    def convoluting_part(self,input_channels,output_channels,kernel_size=3):
        shrink_net = gluon.nn.HybridSequential()
        with shrink_net.name_scope():
            shrink_net.add(gluon.nn.Conv2D(in_channels=input_channels,channels=output_channels,kernel_size=kernel_size,activation='relu'))
            shrink_net.add(gluon.nn.BatchNorm(in_channels=output_channels))
            shrink_net.add(gluon.nn.Conv2D(in_channels=output_channels,channels=output_channels,kernel_size=kernel_size,activation='relu'))
            shrink_net.add(gluon.nn.BatchNorm(in_channels=output_channels))
            shrink_net.add(gluon.nn.MaxPool2D(pool_size=(2,2)))  
        return shrink_net
    
    def deconvoluting_part(self,input_channels,hidden_channel,output_channels,kernel_size=3):
        expand_net = gluon.nn.HybridSequential()
        with expand_net.name_scope():
            expand_net.add(gluon.nn.Conv2D(channels=hidden_channel,kernel_size=kernel_size,activation='relu'))
            expand_net.add(gluon.nn.BatchNorm())
            expand_net.add(gluon.nn.Conv2D(channels=hidden_channel,kernel_size=kernel_size,activation='relu'))
            expand_net.add(gluon.nn.BatchNorm())
            expand_net.add(gluon.nn.Conv2DTranspose(channels = output_channels,kernel_size=kernel_size,strides=(2,2),padding=(1,1),output_padding=(1,1)))
        return expand_net        
      
    def plateau_block(self,input_channels,output_channels):
        plateau_net = gluon.nn.HybridSequential()
        with plateau_net.name_scope():
            plateau_net.add(gluon.nn.Conv2D(channels=512,kernel_size=3,activation='relu'))
            plateau_net.add(gluon.nn.BatchNorm())
            plateau_net.add(gluon.nn.Conv2D(channels=512,kernel_size=3,activation='relu'))
            plateau_net.add(gluon.nn.BatchNorm())
            plateau_net.add(gluon.nn.Conv2DTranspose(channels=256,kernel_size=3,strides=(2,2),padding=(1,1),output_padding=(1,1)))
        return plateau_net            
                            
    def output_block(self,input_channels,hidden_channel,output_channels,kernel_size=3):
        x = gluon.nn.HybridSequential()
        with x.name_scope():
            x.add(gluon.nn.Conv2D(in_channels=input_channels,channels=hidden_channel,kernel_size=kernel_size,activation='relu'))
            x.add(gluon.nn.BatchNorm(in_channels=hidden_channel))
            x.add(gluon.nn.Conv2D(in_channels=hidden_channel,channels=hidden_channel,kernel_size=kernel_size,activation='relu'))
            x.add(gluon.nn.BatchNorm(in_channels=hidden_channel))
            x.add(gluon.nn.Conv2D(in_channels=hidden_channel,channels=output_channels,kernel_size=kernel_size,padding=(1,1),activation='relu'))
            x.add(gluon.nn.BatchNorm(in_channels=output_channels))
        return x
    
    def concatenate(self,upsampling_block,conv_block):
        padding = upsampling_block.shape[2]-conv_block.shape[2]
        mid_padding = padding//2
        padded_conv_block = mx.nd.pad(conv_block,mode="edge",pad_width=(0,0,0,0,mid_padding,mid_padding,mid_padding,mid_padding))
        return mx.nd.concat(upsampling_block,padded_conv_block,dim=1)

    
    def __init__(self,input_channels,output_channels,**kwargs):
        super(network,self).__init__(**kwargs)
        # convolving
        self.conv_depth0 = self.convoluting_part(input_channels,output_channels=64)
        self.conv_depth1 = self.convoluting_part(64,128)
        self.conv_depth2 = self.convoluting_part(128,256)
        
        # plateau 
        self.plateau = self.plateau_block(256,512)
        
        # deconvolving
        self.deconv_depth2 = self.deconvoluting_part(512,256,128)
        self.deconv_depth1 = self.deconvoluting_part(256,128,64)
        self.output_layer = self.output_block(128,64,output_channels)
    
    def hybrid_forward(self,F,X):
        conv_block_0 = self.conv_depth0(X)
        conv_block_1 = self.conv_depth1(conv_block_0)
        conv_block_2 = self.conv_depth2(conv_block_1)
        plateau_block_0 = self.plateau(conv_block_2)
        

        deconv_block_2 = self.concatenate(plateau_block_0,conv_block_2)
        concat_block_2 = self.deconv_depth2(deconv_block_2)
        
        deconv_block_1 = self.concatenate(concat_block_2,conv_block_1)
        concat_block_1 = self.deconv_depth1(deconv_block_1)
        
        deconv_block_0 = self.concatenate(concat_block_1,conv_block_0)
        output_layer = self.output_layer(deconv_block_0)
        return output_layer
net = network(input_channels=1,output_channels=2)
