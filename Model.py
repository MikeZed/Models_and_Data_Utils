import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from Data import Data

import matplotlib.pyplot as plt
from tensorflow import keras
import hickle
import os


class Model:
    # used for making the whole process of building, training and evaluating of the model more organized
    # also can save and load the the model with its training history

    def __init__(self, use_generator=False, img_channels=1, img_res=100, img_mode='pad'):
        self.use_generator = use_generator

        self.img_settings = {'img_res': img_res, 'img_channels': img_channels, 'img_mode': img_mode}

        self.epochs = []
        self.batch_size = []

        self.history = {}
        self.evaluation = None
        self.status = 'unbuilt'  # {'built': False, "trained": False}
        self.model = None

    def __str__(self):
        # return model's info
        if self.model is None:
            return "The model is not ready!"

        opt = self.model.optimizer.get_config()['name']
        lr = self.model.optimizer.get_config()['learning_rate']
        loss = self.model.loss

        epochs_and_batch_size = list(zip(self.epochs, self.batch_size))

        img_res = self.img_settings['img_res']

        struct = []
        self.model.summary(print_fn=lambda x: struct.append(x))
        struct = '\n'.join(struct)

        model_info_dict = {'opt': opt, 'lr': lr, 'loss': loss, 'struct': struct,
                           'epochs_and_batch_size': epochs_and_batch_size, 'img_res': img_res}

        model_str = 'MODEL INFO: \n' \
                    'Image resolution: {img_res}x{img_res}\n' \
                    'Model optimizer, learning rate: {opt}, {lr:.5f}\n' \
                    'Model loss function: {loss}\n' \
                    'Number of epochs and batch sizes: {epochs_and_batch_size}\n' \
                    'Model structure: {struct}\n' \
            .format(**model_info_dict)

        if self.evaluation is not None:
            model_str += "Test dataset evaluation: {}\n".format(self.evaluation)

        return model_str

    ################################################################################################################

    def construct(self, optimizer, loss, struct, epochs, batch_size, url, data_path, data_file, split=(80, 10, 10),
                  save_path=None, metrics=None):
        # builds model, loads data, trains and evaluates model

        # ---------------------------------------
        #            build model
        # ---------------------------------------

        self.build_model(optimizer, loss, struct, metrics)

        # ---------------------------------------
        #    load data, train and evaluate
        # ---------------------------------------

        self.continue_training(url=url, data_path=data_path, data_file=data_file, epochs=epochs, batch_size=batch_size, split=split)

        # ---------------------------------------
        #            save model
        # ---------------------------------------

        if save_path is not None:
            self.save_model(save_path)
            self.update_info(save_path)

        # ---------------------------------------
        #            plot results
        # ---------------------------------------

        self.plot_results(save_path=save_path)

    # ------------------------------------------------------------------------------------------------------------ #

    def continue_training(self, url, data_path, data_file, epochs=100, batch_size=32, split=(80, 20, 0)):
        # loads the data, trains the model and evaluates it

        # ---------------------------------------
        #               load data
        # ---------------------------------------

        data_loader = Data(use_generator=self.use_generator, split=split, img_settings=self.img_settings,
                           url=url, data_path=data_path, data_file=data_file)

        data_loader.load_data()

        self.epochs.append(epochs)
        self.batch_size.append(batch_size)

        data_dict = data_loader.prepare_data(self.batch_size[-1])

        # ---------------------------------------
        #        continue training the model
        # ---------------------------------------

        self.train_model(train=data_dict['train'], val=data_dict['val'])

        self.model.summary()

        # ---------------------------------------
        #            evaluating
        # ---------------------------------------

        if not split[2] == 0:
            self.evaluate_model(data_dict['test'])

    ###########################################   Building   #######################################################

    def build_model(self, optimizer, loss, struct, metrics=None):
        # builds the model either layer by layer or by using transfer learning

        # ---------------------------------------
        #             build model
        # ---------------------------------------

        print("Building model...")

        model = self.build_layers(struct)

        # ---------------------------------------
        #           compile model
        # ---------------------------------------

        if metrics is None:
            metrics = []

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        model.summary()

        self.model = model
        self.status = 'untrained'

        print("Model is built!\n")

    def build_layers(self, model_struct, layers_to_train=3):
        # builds the model layer by layer

        using_transfer = False
        base_layer = model_struct[0]

        # ----------------------------------------------
        # correct channels if using transfer learning
        # ----------------------------------------------

        if not base_layer['name'] == 'In':
            if not self.img_settings['img_channels'] == 3:
                print('using transfer learning -> number of channels was changed to 3')
                self.img_settings['img_channels'] = 3

        # -----------------------------------------------------------
        # build base layer - regular input or use pre-trained model
        # -----------------------------------------------------------
        img_res = self.img_settings['img_res']
        channels = self.img_settings['img_channels']

        input_shape = (img_res, img_res, channels)

        if base_layer['name'] == 'In':
            # building a regular model
            x = keras.layers.Input(shape=input_shape)
            In = x

        else:
            # using transfer learning

            settings = {'weights': 'imagenet', 'include_top': False, 'input_shape': input_shape}

            if base_layer['name'] == 'VGG16':
                x = keras.applications.VGG16(**settings)

            elif base_layer['name'] == 'VGG19':
                x = keras.applications.VGG19(**settings)

            elif base_layer['name'] == 'ResNet50':
                x = keras.applications.ResNet50(**settings)

            else:
                raise NameError('Unknown Base!')

            In = x.input
            x = x.output

            using_transfer = True

        # ---------------------
        # add remaining layers
        # ---------------------

        for layer in model_struct[1:]:
            if layer['name'] == 'Dense':
                x = keras.layers.Dense(layer['size'])(x)

            elif layer['name'] == 'Activation':
                x = keras.layers.Activation(layer['type'])(x)

            elif layer['name'] == 'Conv2D':
                x = keras.layers.Conv2D(layer['filters'], layer['kernel_size'], padding='same')(x)

            elif layer['name'] == 'MaxPooling2D':
                x = keras.layers.MaxPooling2D(layer['size'])(x)

            elif layer['name'] == 'AveragePooling2D':
                x = keras.layers.AveragePooling2D(layer['size'])(x)

            elif layer['name'] == 'BN':
                x = keras.layers.BatchNormalization()(x)

            elif layer['name'] == 'DO':
                x = keras.layers.Dropout(layer['rate'])(x)

            elif layer['name'] == 'Flatten':
                x = keras.layers.Flatten()(x)

            elif layer['name'] == 'Lambda':
                x = keras.layers.Lambda(layer['func'])(x)

            else:
                raise NameError('Unknown Layer!')

        model = keras.models.Model(inputs=In, outputs=x)

        # ---------------------------------------
        # freeze layers if using transfer model
        # ---------------------------------------

        if using_transfer:
            layers_num = len(model.layers)

            for layer in model.layers[:layers_num - layers_to_train]:
                layer.trainable = False

            for layer in model.layers[layers_num - layers_to_train:]:
                layer.trainable = True

        return model

    ###########################################   Training   #######################################################

    def train_model(self, train, val, epochs=None, batch_size=None):
        # trains the model and updates its training history
        # recommended to use only through continue_training, because the epochs and batch_size are determined there
        print("Training model...")

        if epochs is not None:
            self.epochs.append(epochs)

        if batch_size is not None:
            self.batch_size.append(batch_size)

        epochs = sum(self.epochs)

        initial_epoch = epochs - self.epochs[-1]
        batch_size = self.batch_size[-1]

        settings = {'epochs': epochs, 'shuffle': True, 'verbose': 1, 'initial_epoch': initial_epoch}

        if self.use_generator:
            history = self.model.fit_generator(generator=train, validation_data=val, **settings)

        else:
            history = self.model.fit(train[0], train[1], validation_data=val, batch_size=batch_size, **settings)

        history = history.history

        for metric, metric_train_history in history.items():
            self.history.setdefault(metric, []).extend(metric_train_history)

        self.status = 'ready'
        self.evaluation = None

        print("Model is ready!\n")

    ###########################################   Evaluating   #####################################################

    def evaluate_model(self, test):
        # evaluates the model

        if self.use_generator:
            eval = self.model.evaluate_generator(test)
        else:
            eval = self.model.evaluate(test[0], test[1])

        eval = list(zip(self.model.metrics_names, eval))

        print("Test dataset evaluation: {}".format(eval))

        self.evaluation = eval

    #########################################   Saving and Loading   ###############################################

    def save_model(self, save_path):
        # saves the model and its history and settings

        os.mkdir(save_path)

        self.model.save('{}\\model.hdf5'.format(save_path))

        self_list = [self.epochs, self.batch_size, self.history, self.img_settings]

        hickle.dump(self_list, "{}\\model_data.hkl".format(save_path))

    @staticmethod
    def load_model(load_path):
        # loads the model and its history and settings, and then returns it

        new_model = Model()

        new_model.model = keras.models.load_model(load_path + "\\model.hdf5")

        new_model.epochs, new_model.batch_size, new_model.history, new_model.img_settings = \
            hickle.load(load_path + "\\model_data.hkl")

        new_model.status = 'ready'
        new_model.use_generator = False

        return new_model

    def update_info(self, save_path):
        # updates the model's info file

        info = str(self)

        with open(save_path + "\\model_info.txt", "w") as f:
            f.write(info)

    ##########################################   Plotting Results   ################################################

    def plot_results(self, save_path=None):
        # plots the training and validation process

        epochs_range = range(sum(self.epochs))

        plt.figure(figsize=(11, 8))

        metrics_num = len(self.model.metrics_names)

        for i, metric in enumerate(self.model.metrics_names, 1):
            plt.subplot(1, metrics_num, i)

            metric_values = self.history[metric]
            val_metric_values = self.history['val_' + metric]

            if metric == 'loss':
                metric = ' '.join(self.model.loss.split('_')) + ' (loss)'
            else:
                metric = ' '.join(metric.split('_'))

            metric = metric.title()

            plt.plot(epochs_range, metric_values, label='Training ' + metric)
            plt.plot(epochs_range, val_metric_values, label='Validation ' + metric)
            plt.title('Training and Validation ' + metric)
            plt.grid(True)
            plt.legend()

        # plt.subplots_adjust(top=0.75)

        if save_path is not None:
            plt.savefig(save_path + "\\Training and Validation Metrics.png")

        plt.show()
