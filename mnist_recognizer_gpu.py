
# coding: utf-8

# ### Put all import statement here 

# In[22]:


import os

import imageio
import matplotlib.pyplot as plot
import numpy as np
import seaborn as sns
from keras.utils.np_utils import to_categorical 
from IPython.display import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import pandas as pd


# # 1. Prepare data 
# MNIST dataset has a good collection of handwritten digits. Train set has 60,000 rows of data and test set has 10,000 rows of data. The binary data is available at http://yann.lecun.com/exdb/mnist/. <br><br>
# However, to demostrate the real-life situation where the input is a real image, we have downloaded the png images from this link https://github.com/myleott/mnist_png/blob/master/mnist_png.tar.gz. It extracts the data from binary file and convert into 60,000 train images and 10,000 test images. The images are categorized according to their label

# Now we are going to use imageio and read on the image. The sample_image is a 28*28 numpy array. To show the image is labelled correctly, we will use matplotlib to visualize the array

# In[12]:


sample_image = imageio.imread('images/train/0/1.png')
print("The shape of the sample image: ", sample_image.shape)
g = plot.imshow(sample_image)
Image("images/train/0/1.png")


# The real image and graph show that this digit is a 0, matched with its label. Now we need to create a method to read all images and its label into numpy. <br> We normalize the data from [0..255] to [0..1]. It will also help CNN to coverg faster. 

# In[3]:


train_x = np.empty([60000,28,28])
train_y = np.empty([60000], dtype="int32")
test_x = np.empty([10000,28,28])
test_y = np.empty([10000], dtype="int32")

def store_image_to_train_test(index, is_train, image_array, label):
    if is_train:
        train_x[index] = image_array / 255
        train_y[index] = int(label)
    else:
        test_x[index] = image_array / 255
        test_y[index] = int(label)

def loop_train_test_image_file(is_train, parent_folder):
    index = 0
    for i in range (0,10):
        folder = parent_folder + "/" + str(i) + "/"
        for filename in os.listdir(folder):
            store_image_to_train_test(index, is_train, imageio.imread(folder + filename), i)
            index += 1
        print("Completed " + folder + ", accumulated " + str(index) + " images")

loop_train_test_image_file(True, "images/train")
loop_train_test_image_file(False, "images/test")


# Below charts show the distribution of class labels in training and test set. Both are them are nearly balanced 

# In[4]:


x_pos = ('0','1', '2', '3', '4', '5', '6', '7', '8', '9')
plot.bar(x_pos,  np.bincount(train_y), align='center', alpha=0.5)
plot.title('Training set')


# In[5]:


plot.bar(x_pos,  np.bincount(test_y), align='center', alpha=0.5)
plot.title('Test set')


# Encode the labels by using one hot encoder. It will convert into 10 binary labels

# In[6]:


train_y = to_categorical(train_y, num_classes = 10)
test_y = to_categorical(test_y, num_classes = 10)


# Keras requires last dimension to represent the channel. For RGB there will be 3 channels. MNIST is a grayscale so it uses only one channel. 

# In[7]:
train_x = np.around(train_x)
test_x = np.around(test_x)

train_x = np.expand_dims(train_x, axis=3)
test_x = np.expand_dims(test_x, axis=3)


# In[20]:


g = plot.imshow(train_x[9999][:,:,0])


# In[11]:


train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size = 0.1, random_state=2)


# # 2. Modeling

# Before go into CNN, data augmentation is a key idea to avoid overfitting problem. We can transform our data and make the existing dataset even larger. The purpose is to alter the training data with minor tweeks to simulate the real-life sitatuion because each person may write the same digit in different pattern. <br> <br>
# <b>For data augmentation, we choosed apply below options randomly to some training images : </b><br/>
# Randomly Zoom by 15%  <br>
# Randomly rotates by 10 degrees <br> 
# Randomly shift images horizontally by 15% of the width <br>
# Randomly shift images vertically by 15% of the height<br>

# In[8]:


datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,  
        featurewise_std_normalization=False,  
        samplewise_std_normalization=False,  
        zca_whitening=False, 
        rotation_range=10,  # randomly rotate images in the range
        zoom_range = 0.15, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  
        vertical_flip=False)  


datagen.fit(train_x)


# Now we going to define our CNN model. <br> <br>
# <b>1st and 2nd layer</b>: Convolutional layer (Conv2D). They are like a set of learnable filters. Each filter (or kernel) is an integral component of the layered architecture. In these layers, the kernel size is (5,5). It involves taking patches from the input images size (28x28) equal to that of the kernel (5x5) and convolving between the value in the patch and those in the kernal matrix.<br> <br>
# <b>3rd layer</b>: Downsampling (pooling) layer. It looks at the 2 neighboring pixels and picks the maximal value. Mainly to reduce the feature map dimensionality for computational efficiency, which can in turn improve actual performance. 
# <br> <br>
# <b>4th layer</b>: Dropout is a regularization technique to prevent overfitting<br> <br>
# <b>5th and 6th layer</b>: Convolutional layer with 64 bits filter and 3x3 kernal size<br> <br>
# <b> 7th layer </b>: Another downsampling (pooling) layer <br> <br>
# <b> 8th layer </b>: Another dropout layer  <br><br>
# <b> 9th layer </b>: Flatten layer. It is the process of converting all the resultant 2 dimensional arrays into a single long continuous linear vector.   <br><br>
# <b> 10th layer </b>: Dense layer for classification  <br><br>
# <b> 11th layer </b>: Another dropout layer  <br><br>
# <b> 12th layer </b>: Dense layer for classification, with softmax activation function  <br><br>

# In[9]:


model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))


# Next thing is to defines an optimizer, a score function and a loss function. RMSprop is a very effective optimizer. We use "categorical_crossentropy" as our loss function and "accuracy" as our score function

# In[10]:


optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])


# Annealer help to reduce learning rate to reach the global minimum of the loss function

# In[11]:


annealer = ReduceLROnPlateau(monitor='val_acc', 
                             patience=3, 
                             verbose=1, 
                             factor=0.5, 
                             min_lr=0.00001)


# Let run our model and train it! 

# In[ ]:


history = model.fit_generator(datagen.flow(train_x,train_y, batch_size=86),
                              epochs = 10, validation_data = (val_x,val_y),
                              verbose = 2, steps_per_epoch=train_x.shape[0] // 86
                              , callbacks=[annealer])


# In[ ]:

model.save("mnist.h5")


