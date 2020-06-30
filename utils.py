import os
import shutil
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img

# Classification Problem
base_dir = './dogs-vs-cats/'
train_dir = os.path.join(base_dir, 'train/')
validation_dir = os.path.join(base_dir, 'validation/')
train_data_dir = os.path.join(train_dir,'train/')

# Directory with training cat pictures
train_cats_dir = os.path.join(train_dir, 'cats/')
train_dogs_dir = os.path.join(train_dir, 'dogs/')
validation_cats_dir = os.path.join(validation_dir, 'cats/')
validation_dogs_dir = os.path.join(validation_dir, 'dogs/')

def create_train_val_set():
    # if not exists, create one
    if not os.path.exists(validation_dir):
        os.mkdir(validation_dir)
    if not os.path.exists(train_cats_dir):
        os.mkdir(train_cats_dir)
    if not os.path.exists(train_dogs_dir):
        os.mkdir(train_dogs_dir)
    if not os.path.exists(validation_cats_dir):
        os.mkdir(validation_cats_dir)
    if not os.path.exists(validation_dogs_dir):
        os.mkdir(validation_dogs_dir)

    # create train_cats_dir
    if not os.listdir(train_cats_dir)and os.listdir(train_dogs_dir):
        for file in os.listdir(train_data_dir):
            if file.startswith('cat'):
                shutil.move(train_data_dir+file, train_cats_dir+file)
            elif file.startswith('dog'):
                shutil.move(train_data_dir + file, train_dogs_dir + file)

    # number of validation_cats_dir
    validation_split = 0.2
    train_cat_fnames = os.listdir(train_cats_dir)
    train_dog_fnames = os.listdir(train_dogs_dir)
    num_cats = len(train_cat_fnames)
    num_dogs = len(train_dog_fnames)
    num_cats_val = int(validation_split*num_cats)
    num_dogs_val = int(validation_split*num_dogs)

    print('*****************')
    print('The number of cats in total is {}.\nThe number of dogs in total is {}.\n'
          'The number of cats for validation is {}.\nThe number of dogs for validatioin is {}\n '.format(num_cats,
                                                                                                      num_dogs,
                                                                                                     num_cats_val,
                                                                                                     num_dogs_val))

    # create validation set
    if not os.listdir(validation_cats_dir):
        for file in train_cat_fnames[:num_cats_val]:
            shutil.move(train_cats_dir + file, validation_cats_dir + file)

    if not os.listdir(validation_dogs_dir):
        for file in train_dog_fnames[:num_dogs_val]:
            shutil.move(train_dogs_dir + file, validation_dogs_dir + file)

    print('*****************')
    print('The actual number of cats for validation is {}.\nThe actual number of dogs for validation is {}\n'.format(len(os.listdir(validation_cats_dir)),

                                                                                                                  len(os.listdir(validation_dogs_dir))))
def show_figure():

    train_cat_fnames = os.listdir(train_cats_dir)
    train_dog_fnames = os.listdir(train_dogs_dir)

    # present 16 figures, 8 for cats and 8 for dogs
    # parameters for our graph 4*4 configuration

    nrows = 4
    ncols = 4

    # Index for iterating over images
    pic_index = 0

    # Set up matplotlib fig
    fig = plt.gcf()
    fig.set_size_inches(ncols * 4, nrows * 4)

    pic_index += 8
    next_cat_pix = [os.path.join(train_cats_dir, fname)
                                 for fname in train_cat_fnames[pic_index-8: pic_index]]
    next_dog_pix = [os.path.join(train_dogs_dir, fname)
                                 for fname in train_dog_fnames[pic_index-8: pic_index]]

    for i, img_path in enumerate(next_cat_pix + next_dog_pix):
        sp = plt.subplot(nrows,ncols,i+1)
        sp.axis('Off')
        img = mpimg.imread(img_path)
        plt.imshow(img)

        print('*********************')
        print('The size of the img is {} '.format(img.shape))

    plt.show()


# Data preprocessing, convert the image size to 150x150, also add image augmentation

def preprocessing():
    create_train_val_set()
    # All images will be rescaled by 1./255. One for training data, one for validation set to prevent overfitting
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       )
    val_datagen = ImageDataGenerator(rescale=1./ 255) # validation data should not be augmented

    # Flow training/validation images in batches of 20 using train_datagen generator
    train_generator= train_datagen.flow_from_directory(
                    train_dir,
                    target_size=(150,150),
                    batch_size=20,
                    # Since we use binary_crossentropy loss, we need binary labels
                    class_mode='binary'
    )
    validation_generator= val_datagen.flow_from_directory(
                        validation_dir,
                        target_size=(150,150),
                        batch_size=20,
                        # Since we use binary_crossentropy loss, we need binary labels
                        class_mode='binary'
    )
    return train_generator, validation_generator

def plot_acc_loss(history):
    # Retrive accuracy results
    acc = history.history['acc']
    val_acc = history.history['val_acc']

    # Retrive loss results
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Number of epochs
    epochs = range(len(acc))

    #Plot
    plt.plot(epochs, acc)
    plt.plot(epochs, val_acc)
    plt.title('Training and validation accuracy')

    plt.figure()

    plt.plot(epochs, loss)
    plt.plot(epochs, val_loss)
    plt.title('Training and validation loss')

    plt.show()




