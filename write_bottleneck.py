import os
import shutil
import pandas as pd

from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *

import h5py
import math

dir = "/ext/Data/Dog_Breed_Identification/Kaggle"

dog_imgs_list_csv = os.path.join(dir, "labels.csv")
df = pd.read_csv(dog_imgs_list_csv)
#driver_list = df.groupby('subject', as_index=False)['img'].count()
breed_list = df.groupby('breed')['id'].count()
classes = list()
for index, value in breed_list.iteritems():
    classes.append(index)
print(classes)

def write_gap(MODEL, image_size, lambda_func=None):
    width = image_size[0]
    height = image_size[1]
    input_tensor = Input((height, width, 3))
    x = input_tensor
    if lambda_func:
        x = Lambda(lambda_func)(x)
    base_model = MODEL(input_tensor=x, weights='imagenet', include_top=False)
    model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))

    print(MODEL.__name__)
    gen = ImageDataGenerator(
        featurewise_std_normalization=True,
        samplewise_std_normalization=False,
    )
    batch_size = 16
    train_generator = gen.flow_from_directory(os.path.join(dir, "train"), image_size, shuffle=False, batch_size=batch_size, classes=classes, class_mode="categorical")
    test_generator = gen.flow_from_directory(os.path.join(dir, "test"), image_size, shuffle=False, batch_size=batch_size, class_mode=None)

    print("predict_generator train {}".format(math.ceil(train_generator.samples/batch_size)))
    train = model.predict_generator(train_generator, math.ceil(train_generator.samples/batch_size))
    # train = model.predict_generator(train_generator, 1)
    print("train: {}".format(train.shape))
    print("predict_generator test {}".format(math.ceil(train_generator.samples/batch_size)))
    test = model.predict_generator(test_generator, math.ceil(test_generator.samples/batch_size))
    # test = model.predict_generator(test_generator, 1)
    print("test: {}".format(test.shape))
    print(train_generator.classes)
    print("label: {}".format(train_generator.classes.shape))

    # print("begin create database {}".format(Model.__name__))
    with h5py.File("models/premodel/gap_%s.h5" % MODEL.__name__) as h:
        h.create_dataset("train", data=train)
        h.create_dataset("test", data=test)
        h.create_dataset("label", data=train_generator.classes)
    print("write_gap {} successed".format(Model.__name__))

write_gap(ResNet50, (224, 224))
write_gap(Xception, (299, 299), xception.preprocess_input)
write_gap(InceptionV3, (299, 299), inception_v3.preprocess_input)
write_gap(VGG16, (224, 224))
write_gap(VGG19, (224, 224))