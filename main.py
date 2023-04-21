"""
Main file to run the tasks
"""

# generic imports
import sys
import logging
import os

# keras and plot imports
from matplotlib import pyplot
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD

# import Hydra
import hydra
from omegaconf import OmegaConf

logging.basicConfig(filename='output.log', level=logging.DEBUG)

# global hydra config
app_cfg = None

@hydra.main(version_base=None, config_path="conf", config_name="baseline_test")
def main(config):
    # set global hydra conf file
    global app_cfg
    app_cfg = config

    # log global hydra configuration
    logging.info(f"Running with following configuration:\n{OmegaConf.to_yaml(app_cfg)}")

    # create run directory
    try:
        os.stat(config.run_path)
    except:
        print("Creating directory for the run tasks at: {}".format(config.run_path))
        os.makedirs(config.run_path)

    # entry point, run the test harness
    run_test_harness()

# function to terminate program with message
def exit_program(reason):
    print(f"Exiting program: {reason}")
    sys.exit(1)

# load train and test dataset
def load_dataset():
    # load dataset
    (trainX, trainY), (testX, testY) = cifar10.load_data()
    # one hot encode target values
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY
 
# scale pixels
def prep_pixels(train, test):
    # convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # return normalized images
    return train_norm, test_norm
 
# define cnn model with 3-blocks VGG
def define_model():
    global app_cfg

    # check defined model and architecture
    if app_cfg.model.model_type != 'sequential':
        exit_program(f"Model type assigned not supported, write a routine for {app_cfg.model.model_type}")
    elif app_cfg.model.architecture != 'vgg':
        exit_program(f"Model architecture assigned not supported, write a routine for {app_cfg.model.architecture}")

    # Params initialization from Hydra config file
    blocks = app_cfg.model.num_blocks
    activ_funct = app_cfg.model.activation_function
    kernel_init = app_cfg.model.kernel_initializer
    optimizer_name = app_cfg.optimizer.name
    learn_rate = app_cfg.optimizer.lr
    opt_momentum = app_cfg.optimizer.momentum

    # Init value for blocks on VGG
    depth = 32

    if blocks <= 0:
        exit_program("Minimum number of blocks for VGG need to be 1")
    
    logging.info("== Stacking convolutional layers with small 3×3 filters ==")

    # VGG MODEL CREATION
    model = Sequential()
    for i in range(blocks):
        if blocks == 1:
            logging.info(f"Block number: {i+1}\nIteration {i+1}: depth = {depth}")
            model.add(Conv2D(depth, (3, 3), activation=activ_funct, kernel_initializer=kernel_init, padding='same', input_shape=(32, 32, 3)))
            model.add(Conv2D(depth, (3, 3), activation=activ_funct, kernel_initializer=kernel_init, padding='same'))
            model.add(MaxPooling2D((2, 2)))
            depth *= 2
        else:
            logging.info(f"Block number: {i+1}\nIteration {i+1}: depth = {depth}")
            model.add(Conv2D(depth, (3, 3), activation=activ_funct, kernel_initializer=kernel_init, padding='same'))
            model.add(Conv2D(depth, (3, 3), activation=activ_funct, kernel_initializer=kernel_init, padding='same'))
            model.add(MaxPooling2D((2, 2)))
            depth *= 2
    model.add(Flatten())
    model.add(Dense(128, activation=activ_funct, kernel_initializer=kernel_init))
    model.add(Dense(10, activation='softmax'))

    # check defined optimizer
    if optimizer_name != 'sdg':
        exit_program(f"Optimizer assigned not supported, write a routine for {optimizer_name}")
    # compile model
    opt = SGD(lr=learn_rate, momentum=opt_momentum)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
 
# plot diagnostic learning curves
def summarize_diagnostics(history):
    global app_cfg

    # plot loss
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
    # save plot to file
    filename = sys.argv[0].split('/')[-1]
    pyplot.savefig(app_cfg.run_path + '/' + filename + '_plot.png')
    pyplot.close()
 
# run the test harness for evaluating a model
def run_test_harness():
    global app_cfg

    # Setting parameters
    conf_epochs = app_cfg.model.epochs
    conf_batch_sz = app_cfg.model.batch_size

    # load dataset
    trainX, trainY, testX, testY = load_dataset()
    # prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)
    # define model
    model = define_model()
    # fit model
    history = model.fit(trainX, trainY, epochs=conf_epochs, batch_size=conf_batch_sz, validation_data=(testX, testY), verbose=0)
    # evaluate model
    _, acc = model.evaluate(testX, testY, verbose=0)
    print('> %.3f' % (acc * 100.0))
    # learning curves
    summarize_diagnostics(history)


if __name__ == "__main__":
    main()