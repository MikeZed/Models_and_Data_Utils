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
       
        layers = [None] * len(model_struct)
        
        layers_labels = {}  
        inputs=[]
        outputs=[]
        
        i = -1
        
        for layer_description in model_struct[0:len(model_struct)]:
        
            i+=1
            
            # ---------------------------------------
            #        determine previous layer
            # ---------------------------------------
            
            if not (layer_description['name'] == 'input' or layer_description['name'] == 'Transfer'): 
                # if not input layer, get previous layer
                
               prev_layer = layer_description.get('connected_to', None) # get label if exists
                
               if prev_layer is None: 
                   # get layer to connect to according to label, 
                   prev_layer = layers[i-1]
               else:   
                   # if there is no label take the previous layer 
                   prev_layer = layers_labels[prev_layer]             
                                                                        

            layer_amount = 1 # if using transfer model --> the amount of layers added would be the depth of the model, 
                             #                             otherwise we add only 1 layer
               
            # ---------------------------------------
            #     create and connect current layer 
            # --------------------------------------- 
            
            if layer_description['name'] == 'Dense':
                layers[i] = keras.layers.Dense(layer_description['size'])(prev_layer)

            elif layer_description['name'] == 'Activation':
                layers[i] = keras.layers.Activation(layer_description['type'])(prev_layer)

            elif layer_description['name'] == 'Conv2D':
                layers[i] = keras.layers.Conv2D(layer_description['filters'], layer_description['kernel_size'], padding='same')(prev_layer)

            elif layer_description['name'] == 'MaxPooling2D':
                layers[i] = keras.layers.MaxPooling2D(layer_description['size'])(prev_layer)

            elif layer_description['name'] == 'AveragePooling2D':
                layers[i] = keras.layers.AveragePooling2D(layer_description['size'])(prev_layer)

            elif layer_description['name'] == 'BN':
                layers[i] = keras.layers.BatchNormalization()(prev_layer)

            elif layer_description['name'] == 'DO':
                layers[i] = keras.layers.Dropout(layer_description['rate'])(prev_layer)

            elif layer_description['name'] == 'Flatten':
                layers[i] = keras.layers.Flatten()(prev_layer)                   
                
            elif layer_description['name'] == 'Lambda':
                layers[i] = keras.layers.Lambda(layer_description['func'])(prev_layer)
                
            elif layer_description['name'] == 'input':
                # building a regular model
                layers[i] = keras.layers.Input(shape=layer_description['input_shape'])
            
            elif layer_description['name'] == 'Transfer':
                # using transfer learning          
                layer_amount = self.add_transfer_model(layers, layer_description, i)
                i+=layer_amount-1    
                
            else:
                raise NameError('Unknown Layer!')     
                
            # -----------------------------------------
            #   add layer to labels dict if labelled, 
            #   add layer to input / output list if IO 
            # ----------------------------------------- 
               
            if 'label' in layer_description:
                layers_labels[layer_description['label']] = layers[i] # label points to output of a layers
            
            if 'IO' not in layer_description: 
                continue 
                
            if layer_description['IO'] == 'input': 
                inputs.append(layers[i+1-layer_amount])    
            
            if layer_description['IO'] == 'output': 
                outputs.append(layers[i]) 
                
        model = keras.models.Model(inputs=inputs, outputs=outputs)

        return model
        
              
    def add_transfer_model(self, layers, transfer_dict, index): 
    
        settings = {'weights': 'imagenet', 'include_top': False, 'input_shape': transfer_dict['input_shape']}
        
        # ---------------------------------------
        #      determine and load model
        # ---------------------------------------
        
        if transfer_dict['type'] == 'VGG16':
            x = keras.applications.VGG16(**settings)

        elif transfer_dict['type'] == 'VGG19':
            x = keras.applications.VGG19(**settings)
    
        elif transfer_dict['type'] == 'ResNet50':
            x = keras.applications.ResNet50(**settings)

        else:
            raise NameError('Unknown Transfer Model!')
        
        # -----------------------------------------------
        #    add transfer model's layers to list and
        #    freeze layers according to layers_to_train
        # -----------------------------------------------
        
        layer_amount = len(x.layers)
        layers += [None] * layer_amount

        layers_to_train = layer_amount if transfer_dict['layers_to_train'] == 'all' else transfer_dict['layers_to_train']
                
        layers[index] = x.input 
        index+=1
                
        for j, layer in enumerate(x.layers[1:], 1): 
        
            layers[index] = layer 
            index+=1
            
            if j+1 <= layer_amount-layers_to_train: 
                layer.trainable=False
                               
        layers[index-1] = x.output
        
        return layer_amount
                                                     
                                                              
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
        
        
        
        
        