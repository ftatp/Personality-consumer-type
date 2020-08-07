
# coding: utf-8

# ## Transfer Learning Example

# In[2]:


# reference
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
# https://sites.google.com/site/torajim/articles/pythontheanotensorflowkerasopencvinstallonwindows10
# https://gogul09.github.io/software/flower-recognition-deep-learning


# In[150]:


import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


# #### Path

# In[1]:


#Target setting
gender = 'men' # men or women or all
gender_identity = 'neutrial'   # masculine or feminine or neutrial


#path
path = "/home/ftatp/Documents/Studies/personality_consumer_types/What is DNN"
delimiter = "/"

training_data_path = path + delimiter + gender + delimiter + 'train' + delimiter+ gender_identity
test_data_path = path + delimiter + gender + delimiter + 'test' + delimiter+ gender_identity


# In[158]:


# 랜덤 시드 고정
np.random.seed(5)

# batch_size 설정
batch_size = 16
nb_classes = 2


# ### Data preparation

# In[160]:


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
    training_data_path,
    target_size = (150,150),
    batch_size = batch_size,
    class_mode='categorical'
)


# In[161]:


test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_data_path,
    target_size = (150,150),
    batch_size = batch_size,
    class_mode='categorical'
)


# In[162]:


from keras.utils import np_utils

# Converting the labels to one-hot encoded matrix
train_labels = np_utils.to_categorical(train_generator.classes)
test_labels = np_utils.to_categorical(test_generator.classes)


# ### Model Setting

# In[163]:


from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.xception import Xception
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50

from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import SGD

#base_model = VGG16(weights='imagenet', include_top=False)
#base_model = InceptionV3(weights='imagenet', include_top=False) #include_top=False excludes final FC layer
base_model = VGG19(weights='imagenet', include_top=False) #include_top=False excludes final FC layer
#base_model = ResNet50(weights='imagenet', include_top=False) #include_top=False excludes final FC layer
#base_model = Xception(weights='imagenet', include_top=False) #include_top=False excludes final FC layer


# In[164]:


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
    x = Dense(64, activation='relu')(x) 
    predictions = Dense(nb_classes, activation='softmax')(x) 
    model = Model(input=base_model.input, output=predictions)
    
    return model


# In[165]:


def setup_to_transfer_learn(model, base_model):
    """Freeze all layers and compile the model"""
    
    for layer in base_model.layers:
        layer.trainable = False
    
    #sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer='adam',    
                loss='categorical_crossentropy', 
                metrics=['accuracy'])


# In[166]:


model = add_new_last_layer(base_model, nb_classes)
model


# In[167]:


setup_to_transfer_learn(model, base_model)
#setup_to_finetune(model)


# In[ ]:


history = model.fit_generator(
    train_generator,
    epochs=55,
    steps_per_epoch=100,
    validation_data=test_generator,
    validation_steps=128, # batch_Size
    class_weight='auto',
    shuffle=True,
    verbose=1
) 


# In[127]:


model.layers


# In[128]:

layer_name = []
for layer in model.layers:
    print(layer.name)
    layer_name.append(layer.name)	


# In[133]:


# 우리는 마지막 GlobalAveragePooling2D 하고난 뒤 vector space가 필요함

intermediate_layer_name = layer_name[-3]


# In[134]:


# step size ref: https://forums.fast.ai/t/how-do-i-understand-the-steps-parameters-in-predict-generator/3460
train_step_size = train_generator.samples / batch_size
test_step_size = test_generator.samples / batch_size


# In[135]:


from keras.models import Model

intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(intermediate_layer_name).output)
intermediate_output_train = intermediate_layer_model.predict_generator(train_generator, steps=train_step_size)
intermediate_output_test = intermediate_layer_model.predict_generator(test_generator, steps=test_step_size)


# In[137]:


# 본 training data 와 intermediate vector의 크기가 같음을 확인.
print(intermediate_output_train.shape)
print(train_generator.samples)
print(intermediate_output_test.shape)
print(test_generator.samples)


# In[140]:


import pandas as pd
from pandas import DataFrame
import os


# In[141]:


df = DataFrame(intermediate_output_train)
list_col = df.columns.values.tolist()

list_col = ["VGG"+"_"+str(i) for i in list_col]
df.columns = list_col


# In[142]:


training_data_path_re0 = training_data_path+delimiter+"0"
training_data_path_re1 = training_data_path+delimiter+"1"

train_picture0 = os.listdir(training_data_path_re0)
train_picture1 = os.listdir(training_data_path_re1)


train_picture_list0 = [i[:-4] for i in train_picture0]
train_picture_list1 = [i[:-4] for i in train_picture1]

train_picture_f =[]
train_picture_f.extend(train_picture_list0)
train_picture_f.extend(train_picture_list1)

train_picture_f = DataFrame(train_picture_f, columns = ['Picture_name'])
df_f_train = pd.concat([train_picture_f, df], axis=1)
df_f_train.to_csv(gender+"_"+gender_identity +"_"+"train_VGG.csv")


# In[144]:


df = DataFrame(intermediate_output_test)

list_col = df.columns.values.tolist()
list_col = ["VGG"+"_"+str(i) for i in list_col]
df.columns = list_col

test_data_path_re0 = test_data_path+delimiter+"0"
test_data_path_re1 = test_data_path+delimiter+"1"

test_picture0 = os.listdir(test_data_path_re0)
test_picture1 = os.listdir(test_data_path_re1)


test_picture_list0 = [i[:-4] for i in test_picture0]
test_picture_list1 = [i[:-4] for i in test_picture1]

test_picture_f =[]
test_picture_f.extend(test_picture_list0)
test_picture_f.extend(test_picture_list1)

test_picture_f = DataFrame(test_picture_f, columns = ['Picture_name'])
df_f_test = pd.concat([test_picture_f, df], axis=1)
df_f_test.to_csv(gender+"_"+gender_identity +"_"+"test_VGG.csv")

