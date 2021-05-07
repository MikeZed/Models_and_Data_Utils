import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from Data import Data

import matplotlib.pyplot as plt

from tensorflow import keras
import sklearn.metrics 
import hickle
import os


#----------------------
import numpy as np 
#----------------------

class ModelEvaluator:

    def __init__(self, model_object):
        self.model_object = model_object

    ###################################################################################################################
    #                                            Plotting Results                                                     #
    ###################################################################################################################

    def plot_train_val_history(self, plots_in_row=3, save_path=None):
        # plots the training and validation process

        epochs_range = range(sum(self.model_object.epochs))

        plt.figure(figsize=(11, 8))

        plt.subplots_adjust(hspace=0.5)

        metrics_num = len(self.model_object.model.metrics_names)
        
        output_shape = self.model_object.model.output_shape if isinstance(self.model_object.model.output_shape,list) else [self.model_object.model.output_shape]
        outputs_num = len(output_shape)

        plots_num = (metrics_num-1)//outputs_num + 1
          
        metrics = self.model_object.model.metrics_names

        metrics = [[metrics[0]]] + [metrics[1 + n:1 + n + outputs_num] for n in range(0, len(metrics) - 1, outputs_num)]

        loss_func = self.model_object.model.loss if type(self.model_object.model.loss) is str \
            else str(self.model_object.model.loss).split(' ')[1]

        if plots_num == 4:
            plots_in_row = 4

        if plots_num < plots_in_row:
            plots_in_row = plots_num

        num_plot_rows = plots_num // plots_in_row + (plots_num % plots_in_row > 0)

        for i, metric in enumerate(metrics, 1):
            plt.subplot(num_plot_rows, plots_in_row, i)

            for m in metric:
                metric_values = self.model_object.history[m]
                val_metric_values = self.model_object.history['val_' + m]

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
            plt.savefig(os.path.join(save_path, "Training and Validation Metrics.png"))

        plt.show()


    def evaluate_classifier(self, predictions, labels, labels_name, mode='roc', plots_in_row=3, save_path=None): # TODO 
        # evaluates the classifier by plotting the ROC or Precision Recall Curve
       # print(predictions)
        x=np.array(predictions['train'])
        x=((x>0.5).astype(int)).tolist()
       # x=[y[0] for y in x]
        
        plt.figure(figsize=(11, 8))

        plt.subplots_adjust(hspace=0.35, wspace=0.35)
        
        keys = ["train", "val", "test"]
        
        output_shape = self.model_object.model.output_shape if isinstance(self.model_object.model.output_shape,list) else [self.model_object.model.output_shape]
        outputs_num = len(output_shape)

        print(self.model_object.model.output_shape, [self.model_object.model.output_shape])
        print(" --- {} --- ".format(str(outputs_num)))


        if outputs_num < plots_in_row:
            plots_in_row = outputs_num
            
        num_plot_rows = outputs_num // plots_in_row + (outputs_num % plots_in_row > 0)
         
         
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
                
    
        """        
        x, y, _ = evaulate_func(out_labels[keys[0]][0], out_predictions[keys[0]][0])
             
    
        score = score_func(out_labels[keys[0]][0], out_predictions[keys[0]][0])
        
        plt.plot(x, y, label="{}, {}: {:.3f}".format(keys[0], score_label, score))
        
        plt.xlabel(xlabel,fontsize=14)
        plt.ylabel(ylabel,fontsize=14)
        plt.xlim([-0.005, 1.005])
        plt.ylim([-0.005, 1.005])
        plt.grid(True)
        plt.legend()
        """

        for i in range(1, outputs_num + 1): 
            plt.subplot(num_plot_rows, plots_in_row, i)
            
            i-=1
            for key in keys: 
                if key not in out_predictions or key not in out_labels: 
                    continue 

                print(i)
                print(out_labels[key][i][:10])
                x, y, _ = evaulate_func(out_labels[key][i], out_predictions[key][i])

                """
                out_predictions[key][i] = np.array(out_predictions[key][i])
                out_predictions[key][i][out_predictions[key][i]>0.5]=1 
                out_predictions[key][i][out_predictions[key][i]<=0.5]=0
                """
                score = score_func(out_labels[key][i], out_predictions[key][i])
                
                plt.plot(x, y, label="{}, {}: {:.3f}".format(key, score_label, score))
           
            plt.xlabel(xlabel,fontsize=14)
            plt.ylabel(ylabel,fontsize=14)
            plt.xlim([-0.005, 1.005])
            plt.ylim([-0.005, 1.005])
            plt.grid(True)
            plt.legend()

        
            plt.title("{} {}".format(labels_name[i], plt_title),fontsize=18)
                
       
            
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
        
        
        
        
        