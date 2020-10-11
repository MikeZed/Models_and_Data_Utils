import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from Data import Data

import matplotlib.pyplot as plt
from tensorflow import keras
import sklearn.metrics 
import hickle
import os


class Model:
    # used for making the whole process of building, training and evaluating of the model more organized
    # also can save and load the the model with its training history

    def __init__(self, img_settings=None):

        self.img_settings = img_settings

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
        loss = self.model.loss if type(self.model.loss) is str \
                else str(self.model.loss).split(' ')[1]

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

    def construct(self, optimizer, loss, struct, epochs, batch_size, data_dict, layers_to_train=3,
                  save_path=None, metrics=None, use_generator=False):
        # builds model, loads data, trains and evaluates model

        # ---------------------------------------
        #            build model
        # ---------------------------------------
        if self.model is None:
            self.build_model(optimizer, loss, struct, layers_to_train, metrics)

        # ---------------------------------------
        #    load data, train and evaluate
        # ---------------------------------------

        self.continue_training(data_dict=data_dict, epochs=epochs, batch_size=batch_size,
                               use_generator=use_generator)

        # ---------------------------------------
        #            save model
        # ---------------------------------------

        if save_path is not None:
            self.save_model(save_path)
            self.update_info(save_path)

    # ------------------------------------------------------------------------------------------------------------ #

    def continue_training(self, data_dict, epochs=100, batch_size=32, use_generator=False):
        # gets data dictionary, trains the model and evaluates it

        self.epochs.append(epochs)
        self.batch_size.append(batch_size)

        # ---------------------------------------
        #        continue training the model
        # ---------------------------------------

        self.train_model(train=data_dict['train'], val=data_dict['val'], use_generator=use_generator)

        # self.model.summary()

        # ---------------------------------------
        #            evaluating
        # ---------------------------------------

        if 'test' in data_dict: #if not split[2] == 0:
            self.evaluate_model(data_dict['test'], use_generator=use_generator)

    ###################################################################################################################
    #                                             Building                                                            #
    ###################################################################################################################

    def build_model(self, optimizer, loss, struct, layers_to_train=3, metrics=None):
        # builds the model either layer by layer or by using transfer learning

        # ---------------------------------------
        #             build model
        # ---------------------------------------

        print("Building model...")

        model = self.build_layers(struct, layers_to_train=layers_to_train)

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
        layers = [None] * len(model_struct)

        # -----------------------------------------------------------
        # build base layer - regular input or use pre-trained model
        # -----------------------------------------------------------
        img_res = self.img_settings['img_res']
        channels = self.img_settings['img_channels']

        input_shape = (img_res, img_res, channels)

        if base_layer['name'] == 'input':
            # building a regular model
            x = keras.layers.Input(shape=input_shape)
            model_input = x
            layers[0] = x

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

            model_input = x.input  # fix it num of layers
            layers[0] = x.output

            using_transfer = True

        # ---------------------
        # add remaining layers
        # ---------------------

        for i, layer in enumerate(model_struct[1:len(model_struct)-1], 1):

            connected_to = layer.get('connected_to', i-1)

            if layer['name'] == 'Dense':
                layers[i] = keras.layers.Dense(layer['size'])(layers[connected_to])

            elif layer['name'] == 'Activation':
                layers[i] = keras.layers.Activation(layer['type'])(layers[connected_to])

            elif layer['name'] == 'Conv2D':
                layers[i] = keras.layers.Conv2D(layer['filters'], layer['kernel_size'], padding='same')(layers[connected_to])

            elif layer['name'] == 'MaxPooling2D':
                layers[i] = keras.layers.MaxPooling2D(layer['size'])(layers[connected_to])

            elif layer['name'] == 'AveragePooling2D':
                layers[i] = keras.layers.AveragePooling2D(layer['size'])(layers[connected_to])

            elif layer['name'] == 'BN':
                layers[i] = keras.layers.BatchNormalization()(layers[connected_to])

            elif layer['name'] == 'DO':
                layers[i] = keras.layers.Dropout(layer['rate'])(layers[connected_to])

            elif layer['name'] == 'Flatten':
                layers[i] = keras.layers.Flatten()(layers[connected_to])

            elif layer['name'] == 'Lambda':
                layers[i] = keras.layers.Lambda(layer['func'])(layers[connected_to])

            else:
                raise NameError('Unknown Layer!')

        outputs = [layers[i] for i in (model_struct[-1])['outputs']]
        model = keras.models.Model(inputs=model_input, outputs=outputs)

        # ---------------------------------------
        # freeze layers if using transfer model
        # ---------------------------------------

        if using_transfer:
            layers_num = len(model.layers)
            
            if layers_to_train == 'all':
                layers_to_train = layers_num
                
            for layer in model.layers[:layers_num - layers_to_train]:
                layer.trainable = False

            for layer in model.layers[layers_num - layers_to_train:]:
                layer.trainable = True

        return model

    ###################################################################################################################
    #                                             Training                                                            #
    ###################################################################################################################

    def train_model(self, train, val, epochs=None, batch_size=None, use_generator=False):
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

        if use_generator:
            history = self.model.fit(train, validation_data=val, **settings)

        else:
            history = self.model.fit(train[0], train[1], validation_data=val, batch_size=batch_size, **settings)

        history = history.history

        for metric, metric_train_history in history.items():
            self.history.setdefault(metric, []).extend(metric_train_history)

        self.status = 'ready'
        self.evaluation = None

        print("Model is ready!\n")

    ###################################################################################################################
    #                                             Evaluating                                                          #
    ###################################################################################################################

    def evaluate_model(self, test, use_generator=False):
        # evaluates the model

        if use_generator:
            evaluation = self.model.evaluate(test)
        else:
            evaluation = self.model.evaluate(test[0], test[1])

        evaluation = list(zip(self.model.metrics_names, evaluation))

        print("Test dataset evaluation: {}".format(evaluation))

        self.evaluation = eval


    ###################################################################################################################
    #                                             Predicting                                                          # 
    ###################################################################################################################
    
    def predict(self, data):
        # predicts labels by using the model 
        
        if isinstance(data, dict):
        
            predictions={}
            
            for key, value in data.items(): 
                predictions[key]=list(zip(*self.model.predict(value)))
                
            return predictions
            
        
        else:
            return self.model.predict(data)
            

    ###################################################################################################################
    #                                             Saving and Loading                                                  #
    ###################################################################################################################

    def save_model(self, save_path):
        # saves the model and its history and settings

        os.makedirs(save_path, exist_ok=True)

        self.model.save('{}/model.hdf5'.format(save_path))

        self_list = [self.epochs, self.batch_size, self.history, self.img_settings]

        hickle.dump(self_list, "{}/model_data.hkl".format(save_path))

    @staticmethod
    def load_model(load_path):
        # loads the model and its history and settings, and then returns it

        new_model = Model()

        new_model.model = keras.models.load_model(load_path + "/model.hdf5")

        new_model.epochs, new_model.batch_size, new_model.history, new_model.img_settings = \
            hickle.load(load_path + "/model_data.hkl")

        new_model.status = 'ready'
        new_model.use_generator = False

        return new_model

    def update_info(self, save_path):
        # updates the model's info file

        info = str(self)

        with open(save_path + "/model_info.txt", "w") as f:
            f.write(info)

    ###################################################################################################################
    #                                            Plotting Results                                                     #
    ###################################################################################################################

    def plot_train_val_history(self,  plots_in_row=3, save_path=None):
        # plots the training and validation process

        epochs_range = range(sum(self.epochs))

        plt.figure(figsize=(11, 8))

        plt.subplots_adjust(hspace=0.5)

        metrics_num = len(self.model.metrics_names)
        
        output_shape = self.model.output_shape if isinstance(self.model.output_shape,list) else [self.model.output_shape]
        outputs_num = len(output_shape)

        plots_num = (metrics_num-1)//outputs_num + 1
          
        metrics = self.model.metrics_names

        metrics = [[metrics[0]]] + [metrics[1 + n:1 + n + outputs_num] for n in range(0, len(metrics) - 1, outputs_num)]

        loss_func = self.model.loss if type(self.model.loss) is str \
            else str(self.model.loss).split(' ')[1]

        if plots_num == 4:
            plots_in_row = 4

        if plots_num < plots_in_row:
            plots_in_row = plots_num

        num_plot_rows = plots_num // plots_in_row + (plots_num % plots_in_row > 0)

        for i, metric in enumerate(metrics, 1):
            plt.subplot(num_plot_rows, plots_in_row, i)

            for m in metric:
                metric_values = self.history[m]
                val_metric_values = self.history['val_' + m]

                if m == 'loss':
                    m = ' '.join(loss_func.split('_')) + ' (loss)'

                else:
                    m = ' '.join(m.split('_'))

                m = m.title()
                m = ' '.join(m.split('_')[:2])
                plt.plot(epochs_range, metric_values, label='Train ' + m)
                plt.plot(epochs_range, val_metric_values, label='Val ' + m)

            plt_title = ' '.join(loss_func.split('_')) + ' (loss)' if metric[0] == 'loss' \
                else ' '.join(metric[0].split('_')[2:])

            plt.title('Train and Val ' + plt_title)
            plt.grid(True)
            plt.legend()

        # plt.subplots_adjust(top=0.75)

        if save_path is not None:
            plt.savefig(save_path + "/Training and Validation Metrics.png")

        plt.show()


    def evaluate_classifier(self, predictions, labels, mode='roc', plots_in_row=3, save_path=None): # TODO 
        # evaluates the classifier by plotting the ROC or Precision Recall Curve

    
        plt.figure(figsize=(11, 8))

        plt.subplots_adjust(hspace=0.35, wspace=0.35)
        
        keys = ["train", "val", "test"]
        
        output_shape = self.model.output_shape if isinstance(self.model.output_shape,list) else [self.model.output_shape]
        outputs_num = len(output_shape)
        
        if outputs_num < plots_in_row:
            plots_in_row = outputs_num
            
        num_plot_rows = outputs_num // plots_in_row + (outputs_num % plots_in_row > 0)
        
        
        print(outputs_num, num_plot_rows, plots_in_row)
   
        if mode == 'roc':
            # plot ROC 
            evaulate_func=sklearn.metrics.roc_curve
            score_func=sklearn.metrics.roc_auc_score
            score_label = "AUC"
            xlabel='False positives'
            ylabel= 'True positives'
            plt_title="ROC"
            
        else:
            # plot Precision Recall
            evaulate_func=sklearn.metrics.precision_recall_curve
            score_func=sklearn.metrics.average_precision_score
            score_label = "AP"
            xlabel='Recall'
            ylabel= 'Precision'
            plt_title="PR"
            
            
        out_labels, out_predictions = {}, {} 
        
        for key in keys:
            if key not in predictions or key not in labels: 
                continue
                
            out_labels[key] = list(zip(*labels[key]))
            out_predictions[key] = list(zip(*predictions[key]))
            
            if outputs_num == 1:
                out_labels[key] = [[i[0] for i in out_labels[key]]]
                out_predictions[key] = [[i[0] for i in out_predictions[key]]]
        
        for i in range(1, outputs_num + 1): 
            plt.subplot(num_plot_rows, plots_in_row, i)
            
            i-=1
            for key in keys: 
                if key not in predictions or key not in labels: 
                    continue 
                x, y, _ = evaulate_func(out_labels[key][i], out_predictions[key][i])
                
                score = score_func(out_labels[key][i], out_predictions[key][i])
                
                plt.plot(x, y, label="{}, {}: {:.3f}".format(key, score_label, score))
                
            plt.xlabel(xlabel,fontsize=14)
            plt.ylabel(ylabel,fontsize=14)
            plt.xlim([-0.005, 1.005])
            plt.ylim([-0.005, 1.005])
            plt.grid(True)
            plt.legend()

        
            plt.title("Output {} {}".format(i, plt_title),fontsize=18)
                
       
                
            """     
             for i, out_predictions in enumerate(predictions[key+"_predictions"])
                  
                  fp, tp, _ = sklearn.metrics.roc_curve(labels[key+"_labels"][i], out_predictions) 
                        
        for output in outputs
        
            train_fp, train_tp, _ = sklearn.metrics.roc_curve(train_labels, train_predictions)
            train_fp, train_tp, _ = sklearn.metrics.roc_curve(train_labels, train_predictions)
            train_fp, train_tp, _ = sklearn.metrics.roc_curve(train_labels, train_predictions)
            
        
           """  
        
        if save_path is not None:
            plt.savefig("{}/{} Curves.png".format(save_path,plt_title))

        plt.show()
        
        
        
        
        