# Segmentation of optical coherence tomography images with diabetic macular edema using UNets. 

Built from scratch using Apache MxNet and Gluon

![](images/trainoct.JPG)
                      
Plotting the training examples and the results
                        
 ![](images/valoct.JPG)
                       
Performance on our validation examples

![](images/network_structure.JPG)
                            
The UNet Structure 
                            
![](images/modelsummarypart1.JPG)
![](images/modelsummarypart2.JPG)
             
The model parameters when an ndarray of (5,1,284,284) is passed through it
              
              


# Dataset

Images for segmentation of optical coherence tomography images with diabetic macular edema. 
Obtained the dataset from https://www.kaggle.com/paultimothymooney/chiu-2015
I have included the unzipped version of the dataset in this repository

# Installing the requirements
```bash
pip3 install -r requirements.txt
```
# Clone the repository 
```bash
git clone https://github.com/sid0312/OCTImageSegmentation
cd OCTImageSegmentation
```
# To train the model, thereby getting the train and validation accuracies as well as losses
```bash
python train.py
```
# Results
```bash
python results.py
```

To get the intuition of the training process,
go to [unets.ipynb] (https://github.com/sid0312/OCTImageSegmentation/blob/master/unets.ipynb)

## Made with :heart: by Siddhant Baldota, project owner

