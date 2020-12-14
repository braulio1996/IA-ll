## Inteligencia Artificial II

Enunciado:
Desarrollar una red neuronal que realice un proceso de clasificación de imágenes (íconos) que representan diferentes categorías.


### Introduction
    The following project is developed in Jupyter Python + TenserFlow and we call it the Keras api where
    we load folders containing images, thus obtaining optimal solutions for learning our neural network.

### KERAS
Keras is a deep learning API written in Python, running on top of the machine learning platform TensorFlow. It was developed with a focus on enabling fast experimentation. Being able to go from idea to result as fast as possible is key to doing good research.


## Objective
 
 ■ Determine which are the best parameters in order to obtain the most suitable alternative for the neural 
   network and activation function.

 ■ Implement the resources learned in class to carry out the training of a neural network.

 ■ Apply the use of Deep Learning tools for the development of the project

 ■ Create files where the process results are generate


## Development Process

### Requirements:
1. Python 3 ... with which we will make a Notebook Jupyter
2. A suite like Anaconda is recommended to facilitate the process
3. Keras and TensorFlow
4. Dataset


```KERAS
We import the necessary libraries

# Libraries

import numpy as np
import os, sys, re, keras, seaborn
from os import remove
import seaborn as sns
import matplotlib.pyplot as pp
from PIL import Image, ImageSequence, ImageFont, ImageDraw, ImageOps
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
from keras.utils import to_categorical
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten, Conv1D, GlobalMaxPooling1D,MaxPooling1D,GlobalAveragePooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import optimizers
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
%matplotlib inline
K.clear_session()

## We read our main directory
icons = np.load('Icons-50.npy', allow_pickle = True).item()

### We divide our dataset in training and test applying the train_test_split method
train_X,test_X,train_Y,test_Y = train_test_split(x,y,test_size=0.2)

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```


# 1. Implemented methods to load the images

For the processing of the images, a data set is available that contains images
What we do is divide the data set into 80-20 for training and testing.

Before we start we must load the data we need and use the ** plt.imread (filepath) 
** method that will load all the images in an array.

For more details see [IA2 PROYECT](https://github.com/braulio1996/IA2WEB).

### video

[Tutorial](https://www.youtube.com/watch?v=f3S5oCCYto8)

### Conclusion
Well ending, keras is a very powerful library that help us in many ways when creating a neural network, provides us with several classes to create, but despite this is very important to make a cleanup of the data before performing the analysis, because there may be empty data, for example we should apply the average for the data or the kmeans algorithm.

• Keras provides a high-level API for creating a deep neural network.

• Keras allows you to define a trigger function of your choice on each layer.

• Image recognition was carried out, with different layers and with good precision. RNA options were presented, with the best result being a precision of (PENDIENTE CUANTO MARCO EL RESULTADO) 83,   this is due to the activation function, the layers and times used.

• The combination of different trigger functions, the number of layers and the optimizer used will cause the precision result to vary considerably.


### Recomendation
• When working with images, it must be clear if they are in color or not, and they flatten the dimension for a better effect on the result.

• The amount of data must be significant, since RNA may not be as accurate with little data.

### Results
Artificial neural networks
```KERAS
!['RNA1'](https://github.com/braulio1996/IA-ll/blob/main/RNA1.png)
!['RNA2'](https://github.com/braulio1996/IA-ll/blob/main/RNA2.png)
!['RNA3'](https://github.com/braulio1996/IA-ll/blob/main/RNA3.png)
!['RNA3'](https://github.com/braulio1996/IA-ll/blob/main/RNA4.png)

```
