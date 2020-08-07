
# coding: utf-8

# In[3]:


import numpy as np
import keras
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


# In[4]:


# reference
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
# https://sites.google.com/site/torajim/articles/pythontheanotensorflowkerasopencvinstallonwindows10


# #### remove false files

# In[5]:


#from PIL import Image
#import os
#from os import listdir
#
#train_path = '/home/ftatp/Documents/Studies/personality_consumer_types/Social/Train'
#test_path = '/home/ftatp/Documents/Studies/personality_consumer_types/Social'
#
#
## In[6]:
#
#
## remove photos from the list
#false_photos_file = open('false_emotion_photos.txt', 'r')
#false_photos = false_photos_file.readlines()
#
#
## In[7]:
#
#
## for train data set
#dirs = []
#for root, dirs, files in os.walk(train_path):
#    dirs = dirs
#    break
#    
#for dir_name in dirs:
#    for file_name in listdir(train_path + dir_name):
#        file_name = file_name.replace('.jpg', '')
#        existed = 0
#        for f_photo in false_photos:
#            if file_name == f_photo.strip():
#                existed = 1
#                break
#        if existed == 1:
#            os.remove(train_path + dir_name + '/' + file_name + '.jpg')
#            #print(train_path + dir_name + '/' + file_name + '.jpg')
#
#
## In[8]:
#
#
## for testing data set
#dirs = []
#for root, dirs, files in os.walk(test_path):
#    dirs = dirs
#    break
#    
#for dir_name in dirs:
#    for file_name in listdir(test_path + dir_name):
#        file_name = file_name.replace('.jpg', '')
#        existed = 0
#        for f_photo in false_photos:
#            if file_name == f_photo.strip():
#                existed = 1
#                break
#        if existed == 1:
#            os.remove(test_path + dir_name + '/' + file_name + '.jpg')
#            #print(test_path + dir_name + '/' + file_name + '.jpg')
#

# ### defining a model

# In[9]:


# 랜덤 시드 고정
np.random.seed(5)


# In[10]:


batch_size = 16


# In[9]:


train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=10,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
    #'C:/Users/user/Desktop/Data/Emotions/images/new_samples_positive_negative/train/',
    '/home/ftatp/Documents/Studies/personality_consumer_types/Social/Train/',
    target_size = (150,150),
    batch_size = batch_size,
    class_mode='categorical'
)


# In[10]:


test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    '/home/ftatp/Documents/Studies/personality_consumer_types/Social/Test/',
    target_size = (150,150),
    batch_size = batch_size,
    class_mode='categorical'
)


# ### Keras Model

# In[11]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils.training_utils import multi_gpu_model
import tensorflow as tf
# ref: http://3months.tistory.com/211

def CovNet():
    with tf.device('/cpu:0'):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=(150, 150, 3)))
        model.add(Activation('relu'))
        model.add(Dropout(0.4))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Dropout(0.4))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # the model so far outputs 3D feature maps (height, width, features)

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.4))
        model.add(Dense(6))
        model.add(Activation('sigmoid'))

        print(model.summary())
        
#    parallel_model = multi_gpu_model(model, gpus=2)
#parallel_
    model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])
    
    return model


# In[ ]:


model = CovNet()


# In[ ]:

model.fit_generator(
    train_generator,
    epochs=20,
    steps_per_epoch=100,
    validation_data=test_generator,
    validation_steps=128, # batch_Size
    shuffle=True,
    verbose=1
)


# In[ ]:


# 모델 평가하기
scores = model.evaluate_generator(test_generator)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# In[ ]:


#output = model.predict_generator(test_generator)


# In[ ]:


'''
index = 0
label = 0
correct_count = 0
for row in output:
    sorted_row = np.argsort(-row)
    
    if index >= 0 and index < 2000:
        label = 0
    if index >= 2000 and index < 4000:
        label = 1
    if index >= 4000 and index < 6000:
        label = 2
    if index >= 6000 and index < 8000:
        label = 3
    if index >= 8000 and index < 10000:
        label = 4
    if index >= 10000 and index < 12000:
        label = 5
    
    index += 1
    
    if label in sorted_row[0:1]:
        correct_count += 1
'''


# In[ ]:


# 모델 사용하기
#output = model.predict_generator(test_generator, steps=5)
#np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
#print(test_generator.class_indices)
#print(output)


# ## VVG16 ###
# 
# * https://deeplearningsandbox.com/how-to-use-transfer-learning-and-fine-tuning-in-keras-and-tensorflow-to-build-an-image-recognition-94b0b02444f2

# In[ ]:


from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16


# In[ ]:


#model = VGG16(weights='imagenet', include_top=False)
model = VGG16()


# In[ ]:


# 0. 사용할 패키지 불러오기
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.applications import imagenet_utils
import numpy as np

# 1. 모델 구성하기
model = VGG16(weights='imagenet')

# 2. 모델 사용하기 

# 임의의 이미지 불러오기
img_path = 'cat.jpg'
img = image.load_img(img_path, target_size=(224, 224))
xhat = image.img_to_array(img)
xhat = np.expand_dims(xhat, axis=0)
xhat = preprocess_input(xhat)

# 임의의 이미지로 분류 예측하기
yhat = model.predict(xhat)

# 예측 결과 확인하기
P = imagenet_utils.decode_predictions(yhat)

for (i, (imagenetID, label, prob)) in enumerate(P[0]):
    print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))


# ## Transfer Learning

# In[12]:


train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=10,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
    #'C:/Users/user/Desktop/Data/Emotions/images/new_samples_positive_negative/train/',
    '/home/ftatp/Documents/Studies/personality_consumer_types/Social/Train/',
    target_size = (150,150),
    batch_size = batch_size,
    class_mode='categorical'
)


# In[13]:


test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    '/home/ftatp/Documents/Studies/personality_consumer_types/Social/Test/',
    target_size = (150,150),
    batch_size = batch_size,
    class_mode='categorical'
)


# In[14]:


from keras.utils import np_utils

# Converting the labels to one-hot encoded matrix
train_labels = np_utils.to_categorical(train_generator.classes)
test_labels = np_utils.to_categorical(test_generator.classes)


# In[15]:


from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.xception import Xception
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50

#base_model = VGG16(weights='imagenet', include_top=False)
#base_model = InceptionV3(weights='imagenet', include_top=False) #include_top=False excludes final FC layer
#base_model = VGG19(weights='imagenet', include_top=False) #include_top=False excludes final FC layer
#base_model = ResNet50(weights='imagenet', include_top=False) #include_top=False excludes final FC layer
base_model = Xception(weights='imagenet', include_top=False) #include_top=False excludes final FC layer


# In[19]:


from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import SGD

# https://towardsdatascience.com/https-medium-com-manishchablani-useful-keras-features-4bac0724734c
# https://gist.github.com/fchollet/7eb39b44eb9e16e59632d25fb3119975
# https://deeplearningsandbox.com/how-to-use-transfer-learning-and-fine-tuning-in-keras-and-tensorflow-to-build-an-image-recognition-94b0b02444f2
# https://hackernoon.com/learning-keras-by-implementing-vgg16-from-scratch-d036733f2d5
# https://machinelearningmastery.com/use-pre-trained-vgg-model-classify-objects-photographs/

def add_new_last_layer(base_model, nb_classes):
    """Add last layer to the convnet
    Args:
    base_model: keras model excluding top
    nb_classes: # of classes
    Returns:
    new keras model with last layer
    """
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x) 
    predictions = Dense(nb_classes, activation='softmax')(x) 
    model = Model(input=base_model.input, output=predictions)
    
    return model


# In[20]:


def setup_to_transfer_learn(model, base_model):
    """Freeze all layers and compile the model"""
    
    for layer in base_model.layers:
        layer.trainable = False
    
    #sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer='adam',    
                loss='categorical_crossentropy', 
                metrics=['accuracy'])


# In[13]:


'''
# https://github.com/DeepLearningSandbox/DeepLearningSandbox/blob/master/transfer_learning/fine-tune.py
def setup_to_finetune(model):
    NB_IV3_LAYERS_TO_FREEZE = 172
    """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top 
      layers.
   note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in 
         the inceptionv3 architecture
   Args:
     model: keras model
   """
    for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
        layer.trainable = False
    for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
        layer.trainable = True
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),   
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
'''


# In[21]:


model = add_new_last_layer(base_model, 6)
model


# In[22]:


setup_to_transfer_learn(model, base_model)
#setup_to_finetune(model)


# In[23]:


history = model.fit_generator(
    train_generator,
    epochs=30,
    steps_per_epoch=300,
    validation_data=test_generator,
    validation_steps=128, # batch_Size
    class_weight='auto',
    shuffle=True,
    verbose=1
) 

