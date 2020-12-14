## Inteligencia Artificial II

Enunciado:
Desarrollar una red neuronal que realice un proceso de clasificación de imágenes (íconos) que representan diferentes categorías.

### KERAS
Keras is a deep learning API written in Python, running on top of the machine learning platform TensorFlow. It was developed with a focus on enabling fast experimentation. Being able to go from idea to result as fast as possible is key to doing good research.


```Code
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

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/braulio1996/IA2WEB/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
