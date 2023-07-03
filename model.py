import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import platform
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

import keras
import tensorflow as tf
from keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from keras.layers import Activation, Add, AveragePooling2D, BatchNormalization, Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

if platform.system() == 'Windows':
    slash = '\\'
else:
    slash = '/'

def ImgGen(dataframe, img_size=(128,128), batch_size=32, brightness=[0.7, 1.3], vsplit=0.2, rrange=15, seed=2023, shuffle=True):
    """Function for the creation of Image Data Generators. 
    
    Parameters
    ----------
    dataframe: dataframe of images to be output by the generators
    img_size: desired output image size; default (128,128)
    batch_size: number of images to be returned by each call to the generator; default 32
    brightness: range of brightness for the images to be augmented by; default [0.7, 1.3]
    vsplit: percentage of images to be allocated to validation set; default 0.2
    rrange: range of rotation for augmented images; default 15
    seed: random seed; default 2023
    shuffle: boolean determining whether to shuffle the images; default True
    """

    dir = os.getcwd() + slash + 'B3FD' + slash

    train_datagen = ImageDataGenerator(validation_split=vsplit, rotation_range=rrange, fill_mode='nearest',
                                       brightness_range=brightness, rescale=1./255)
    
    val_datagen = ImageDataGenerator(validation_split=vsplit, rescale=1./255)
    
    train_gen = train_datagen.flow_from_dataframe(dataframe=dataframe, directory=dir, x_col='path', y_col=['age', 'gender'], class_mode='multi_output', batch_size=batch_size,
                                                  target_size=img_size, subset='training', shuffle=shuffle, seed=seed)
    
    val_gen = val_datagen.flow_from_dataframe(dataframe=dataframe, directory=dir, x_col='path', y_col=['age', 'gender'], class_mode='multi_output', batch_size=batch_size,
                                                  target_size=img_size, subset='validation', shuffle=shuffle, seed=seed)

    return train_gen, val_gen


def conv_block(x, filter):
    # copying input x
    x_resid = x

    # conv layer 1
    x = Conv2D(filter, kernel_size=(3,3), strides=(2,2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x) # Dropout layer

    # conv layer 2
    x = Conv2D(filter, kernel_size=(3,3), padding='same')(x)
    x = BatchNormalization()(x)
    
    # creating residual connection
    x_resid = Conv2D(filter, kernel_size=(3,3), strides=(2,2), padding='same')(x_resid)
    x = Add()([x, x_resid])
    x = Activation('relu')(x)

    return x 

def identity_block(x, filter):
    # copying input x
    x_resid = x

    # identity layer 1
    x = Conv2D(filter, kernel_size=(3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x) # Dropout layer

    # identity layer 2
    x = Conv2D(filter, (3,3), padding='same')(x)
    x = BatchNormalization()(x)

    # creating residual connection
    x = Add()([x, x_resid])
    x = Activation('relu')(x)

    return x



def make_model(shape=(256,256,3)):

    # first layer and pooling
    with tf.device('/gpu:0'):
        input_layer = keras.Input(shape=shape, name='input_layer')
        x = Conv2D(128, kernel_size=(3,3), padding='same')(input_layer)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2,2), strides=(2,2), padding='same')(x)

        block_loops = [2, 3, 4, 2]
        filter_size = 128

        # cycle through layer blocks
        for i in range(4):
            if i == 0:
                for j in range(block_loops[i]):
                    x = identity_block(x, filter_size)
            else:
                filter_size *= 2
                x = conv_block(x, filter_size)
                for j in range(block_loops[i] - 1):
                    x = identity_block(x, filter_size)
                
        
        # penultimate layer
        x = AveragePooling2D((2,2), padding='same')(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        # output layers
        age_output = Dense(6, activation='softmax', name='age')(x)
        gender_output = Dense(1, activation='sigmoid', name='gender')(x)

        model = keras.Model(inputs=input_layer, outputs=[age_output, gender_output])
        return model

# There exists much more pythonic ways to accomplish this,
# however for the sake of easy I created this function in a much more ugly way
def plot_history(history, file):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 7.5))

    ax1.set_title('Loss')
    sns.lineplot(x=history.epoch, y=history.history['loss'], ax=ax1, label='Train Loss')
    sns.lineplot(x=history.epoch, y=history.history['val_loss'], ax=ax1, label='Val Loss')

    ax2.set_title('Loss by Target')
    sns.lineplot(x=history.epoch, y=history.history['gender_loss'], ax=ax2, label='Train Loss: Gender')
    sns.lineplot(x=history.epoch, y=history.history['val_gender_loss'], ax=ax2, label='Val Loss: Gender')
    sns.lineplot(x=history.epoch, y=history.history['age_loss'], ax=ax2, label='Train Loss: Age')
    sns.lineplot(x=history.epoch, y=history.history['val_age_loss'], ax=ax2, label='Val Loss: Age')

    ax3.set_title('Accuracy')
    ax3.set_ylim([0, None])
    sns.lineplot(x=history.epoch, y=history.history['gender_accuracy'], ax=ax3, label='Train Accuracy: Gender')
    sns.lineplot(x=history.epoch, y=history.history['val_gender_accuracy'], ax=ax3, label='Val Accuracy: Gender')
    sns.lineplot(x=history.epoch, y=history.history['age_accuracy'], ax=ax3, label='Train Accuracy: Age')
    sns.lineplot(x=history.epoch, y=history.history['val_age_accuracy'], ax=ax3, label='Val Accuracy: Age')

    fig.savefig(file)

# Function to display sample images from generator
def gen_sample(generator, file):
    """Function which takes in an image generator and displays a sample of 9 images"""
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
    # Create list for labeling images
    age_groups = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
    genders = ['Female', 'Male']
    img, label = generator.next()
    for i, ax in enumerate(axes.flat):
        title = f'Age: {age_groups[label[0][i].argmax()]}, Gender: {genders[label[1][i]]}'
        ax.imshow(img[i])
        ax.set(title=f"{title}")
        ax.axis('off')
    fig.suptitle('Examples of Generated Images', fontsize=30)
    plt.tight_layout()
    fig.savefig(file, transparent=False)


if __name__ == '__main__':
    suffix = sys.argv[1]

    # loading in datasets
    imdb = pd.read_csv(f'B3FD_metadata{slash}B3FD-IMDB_age_gender.csv', delimiter=' ')
    wiki = pd.read_csv(f'B3FD_metadata{slash}B3FD-WIKI_age_gender.csv', delimiter=' ')

    # combining datatsets
    df = pd.concat([imdb, wiki])
    df = df.reset_index(drop=True)

    # function to group ages
    def age_split(age):
        if age < 18:
            return 'X'
        if age in range(18,25):
            return 'A'
        if age in range(25,35):
            return 'B'
        if age in range(35,45):
            return 'C'
        if age in range(45,55):
            return 'D'
        if age in range(55,65):
            return 'E'
        else:
            return 'F'
        
    df['age'] = df['age'].apply(age_split)

    # Dropping indiviudals under 18
    df = df.drop(df.loc[df['age'] == 'X'].index)
    df = df.reset_index(drop=True)

    ohe = OneHotEncoder(sparse_output=False)
    age = ohe.fit_transform(df[['age']]).tolist()
    df['age'] = age
    df['gender'] = df['gender'].str.get_dummies()['M']

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    BATCH_SIZE = 128

    train_gen, val_gen = ImgGen(df_train, img_size=(128,128), brightness=[0.5, 1.5], rrange=30, vsplit=0.2, batch_size=BATCH_SIZE)
    test_gen, null_gen = ImgGen(df_test, img_size=(128,128), batch_size=BATCH_SIZE, vsplit=0, brightness=None, rrange=0, shuffle=False)

    gen_sample(train_gen, f'images{slash}samples_{suffix}.png')

    model = make_model(shape=(128, 128, 3))
    
    model.compile(optimizer='adam', loss=[CategoricalCrossentropy(), BinaryCrossentropy()], loss_weights=[10,1], metrics='accuracy')

    callback = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=False)

    results = model.fit(train_gen, epochs=50, validation_data=val_gen, validation_steps=(len(val_gen.filenames)//BATCH_SIZE), callbacks=[callback])

    plot_history(results, f'images{slash}results_{suffix}.png')

    model.save(f'models{slash}model_no_opt_{suffix}', include_optimizer=False)
    model.save(f'models{slash}model_with_opt_{suffix}')

    model.evaluate(test_gen)