# Script to train the final model and test with test dataset
from IPython.display import display
import pandas as pd
import os
from pathlib import Path
import tensorflow.keras as keras
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.efficientnet import EfficientNetB2
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# CWD = Path.cwd() # This will give the working path, that is the same as the notebook.
# For non interactive scripts (i.e. running script with python), use: CWD = Path(__file__).parent
CWD = Path(__file__).parent

# General config
DATA_PATH = CWD / "../../data/Classification_dataset"
MODELS_PATH = "./models/"
SAMPLE_IMAGE = DATA_PATH / "Anwar Ratool" / "IMG_20210630_102920.jpg"

# Training config
TRAIN_SPLIT=0.8
VAL_SPLIT=0
CHANNELS = 3
WIDTH = 299
HEIGHT = 299
IMAGE_SIZE = (WIDTH,HEIGHT)
IMAGE_SHAPE = (WIDTH,HEIGHT,CHANNELS)
BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 0.001
INNER_LAYERS_CONFIG = [[{'size': 1000, 'drop_rate': 0.5}]]
BASE_MODEL = 'xception'

def create_folds (data_path: str, train_split=0.8, val_split=0.75) -> pd.DataFrame:

    classes = [subdir.name for subdir in os.scandir(data_path) if subdir.is_dir()]
    class_dirs = [subdir for subdir in os.scandir(data_path) if subdir.is_dir()]

    all_images = []
    all_classes = []

    for class_dir in class_dirs:
        for file in os.scandir(class_dir.path):
            if file.is_file():
                all_classes.append(class_dir.name)
                all_images.append(file.path)
    df_images = pd.DataFrame({'path': all_images, 'class': all_classes})

    df_full_train, df_test = train_test_split(df_images, train_size=train_split, random_state=1, shuffle=True)
    if val_split > 0:
        df_train, df_val = train_test_split(df_full_train, train_size=val_split, random_state=1, shuffle=True)
        print(f'Folds shapes: full_train={df_full_train.shape}, train={df_train.shape}, val={df_val.shape}, test={df_test.shape}')
    else:
        df_train = df_full_train
        df_val = None
        print(f'Folds shapes: full_train={df_full_train.shape}, train={df_train.shape}, val=None, test={df_test.shape}')
    print('Classes = ', classes)
    return df_full_train, df_train, df_val, df_test, classes


def create_datasets(preprocessing_function, target_size, batch_size, transformations, df_train=None, df_val=None, df_test=None):

    generator_train = ImageDataGenerator(preprocessing_function=preprocessing_function, **transformations)
    generator_val = ImageDataGenerator(preprocessing_function=preprocessing_function)

    if df_train is not None:
        train_dataset = generator_train.flow_from_dataframe(
            df_train,
            x_col = 'path',
            y_col = 'class',
            target_size = target_size, 
            class_mode = 'categorical',
            batch_size = batch_size, 
        )
    else:
        train_dataset=None

    if df_val is not None:
        val_dataset = generator_val.flow_from_dataframe(
            df_val,
            x_col = 'path',
            y_col = 'class',
            target_size = target_size, 
            class_mode = 'categorical', 
            batch_size = batch_size, 
            shuffle = False
        )
    else:
        val_dataset=None

    if df_test is not None:
        test_dataset = generator_val.flow_from_dataframe(
            df_test,
            x_col = 'path',
            y_col = 'class',
            target_size = target_size, 
            class_mode = 'categorical',
            batch_size = batch_size,
            shuffle = False
        )
    else:
        test_dataset=None

    return train_dataset, val_dataset, test_dataset


def make_model(input_shape=IMAGE_SHAPE, number_of_classes=None, learning_rate=0.01, inner_layers=None, base_model = 'xception'):
      
    if base_model == 'xception':
        
        base_model = Xception(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape,
            pooling='avg'
        )
    else:
        
        base_model = EfficientNetB2(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape,
            pooling='avg'
        )
    
    base_model.trainable = False

    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs)

    x = keras.layers.BatchNormalization()(x)
    
    if inner_layers and len(inner_layers) > 0:
        for layer in inner_layers:
            print(layer)
            for param, value in layer.items():
                if param == 'size':
                    x = keras.layers.Dense(units=value, activation='relu')(x)
                if param == 'drop_rate':
                    x = keras.layers.Dropout(value, seed=1)(x)

    outputs = keras.layers.Dense(units=number_of_classes, activation='softmax')(x)  # Softmax activationb, check https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
    model = keras.Model(inputs, outputs)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy(from_logits=False)  # We use softmax in the output layer
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    print(model.summary())

    return model

def create_checkpoint(name:str='xception-v1', path=MODELS_PATH, delete_files=True)->keras.callbacks.ModelCheckpoint: 

    if delete_files:
        for file in os.listdir(path):
            if name in file:
                os.remove(f'{path}{file}')

    return keras.callbacks.ModelCheckpoint(
        f'{path}{name}' + '.{epoch:02d}_{val_accuracy:.3f}.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )

def run_train(checkpoint=None, inner_layers_tests=[[]], epochs=20, learning_rates=[], input_shape=(150,150,3), base_model='xception', train_dataset=None, val_dataset=None):
    
    callbacks = [checkpoint] if checkpoint else []
    
    scores = {}
    for learning_rate in learning_rates:
        for inner_layers in inner_layers_tests:
            model = make_model(input_shape=input_shape, number_of_classes=number_of_classes, learning_rate=learning_rate, inner_layers=inner_layers, base_model=base_model)
            history = model.fit(x=train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=callbacks)
            params = f'lr={learning_rate}' + '-'.join([f'{param}={value}' for layer in inner_layers for param, value in layer.items()])
            scores[params]=history
    return scores, model

if __name__ == "__main__":

    df_full_train, df_train, df_val, df_test, classes = create_folds(
        DATA_PATH,
        train_split=TRAIN_SPLIT,
        val_split=VAL_SPLIT
    )

    number_of_classes = len(classes)
    print(f'Number of classes = {number_of_classes}')

    transformations = transformations={'horizontal_flip':True}

    train_dataset, val_dataset, test_dataset = create_datasets(
        preprocessing_function=preprocess_input,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        transformations=transformations,
        df_train=df_full_train,
        df_val=None,
        df_test=df_test
    )

    print(f'Classes index = {train_dataset.class_indices}')

    # Delete all development models
    for file in os.listdir(MODELS_PATH):
        os.remove(f'{MODELS_PATH}{file}')
    checkpoint = create_checkpoint('best-model', MODELS_PATH)

    inner_layers_config = INNER_LAYERS_CONFIG
    scores, model = run_train(checkpoint, inner_layers_config, epochs = EPOCHS, learning_rates=[LEARNING_RATE], input_shape=IMAGE_SHAPE, base_model=BASE_MODEL, train_dataset=train_dataset, val_dataset=test_dataset)

    models = [file.name for file in os.scandir(MODELS_PATH) if file.is_file()]
    accuracies = [float(file.split('.h5')[0].split('_')[1]) for file in models]
    best_model = models[accuracies.index(max(accuracies))]
    print(f'Best Model: {best_model}')
    model = keras.models.load_model(MODELS_PATH + best_model)
    test_loss, test_accuracy = model.evaluate(test_dataset)
    print(f'Test accuracy = {test_accuracy:0.4f}')
    model_path = f'{MODELS_PATH}{best_model}'
    command = f'python convert_model.py --modelpath {model_path}'
    os.system(command)



