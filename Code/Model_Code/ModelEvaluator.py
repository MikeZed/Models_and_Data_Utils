
import sklearn.metrics
import os
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


# ----------------------
# ----------------------


class ModelEvaluator:

    def __init__(self, model_object):
        pass
    ###################################################################################################################
    #                                            Plotting Results                                                     #
    ###################################################################################################################

    @staticmethod
    def plot_train_val_history(model, plots_in_row=3, save_path=None):
        # plots the training and validation process

        epochs_range = range(sum(model.epochs))

       # plt.figure(figsize=(11, 8))

      #  plt.subplots_adjust(hspace=0.5)

        metrics_num = len(model.model.metrics_names)

        output_shape = model.model.output_shape if isinstance(
            model.model.output_shape, list) else [model.model.output_shape]
        outputs_num = len(output_shape)

        plots_num = (metrics_num-1)//outputs_num + 1

        metrics = model.model.metrics_names

        metrics = [[metrics[0]]] + [metrics[1 + n:1 + n + outputs_num]
                                    for n in range(0, len(metrics) - 1, outputs_num)]

        loss_func = model.model.loss if type(model.model.loss) is str \
            else str(model.model.loss).split(' ')[1]

        if plots_num == 4:
            plots_in_row = 4

        if plots_num < plots_in_row:
            plots_in_row = plots_num

        num_plot_rows = plots_num // plots_in_row + \
            (plots_num % plots_in_row > 0)

        for i, metric in enumerate(metrics, 1):
           # plt.subplot(num_plot_rows, plots_in_row, i)
            plt.figure(figsize=(11, 8))

            for m in metric:
                metric_values = model.history[m]
                val_metric_values = model.history['val_' + m]
                print(m)
                """
                if m == 'loss':
                    m = ' '.join(loss_func.split('_')) + ' (loss)'

                else:
                    m = ' '.join(m.split('_'))

                m = m.title()
                """
                m = ' '.join(m.split('_')[:2])
                plt.plot(epochs_range, metric_values, label='Train ' + m)
                plt.plot(epochs_range, val_metric_values, label='Val ' + m)

            plt_title = ' '.join(loss_func.split('_')) + ' (loss)' if metric[0] == 'loss' \
                else ' '.join(metric[0].split('_')[2:])

            if(plt_title == ''):
                plt_title = m

            plt.title('Train and Val ' + plt_title)
            plt.grid(True)
            # plt.legend()

            plt.legend(loc='upper center', bbox_to_anchor=(
                0.5, -0.05), fancybox=True, shadow=True, ncol=5)
            if save_path is not None:
                plt.savefig(os.path.join(
                    save_path, "Training and Validation {}.png".format(plt_title)))

            plt.show()

        # plt.subplots_adjust(top=0.75)

    @staticmethod
    def evaluate_classifier(predictions, labels, labels_name, mode='roc', plots_in_row=3, save_path=None):  # TODO
        # evaluates the classifier by plotting the ROC or Precision Recall Curve

        keys = ["train", "val", "test"]

        outputs_num = len(labels_name)

        if outputs_num < plots_in_row:
            plots_in_row = outputs_num

        num_plot_rows = outputs_num // plots_in_row + \
            (outputs_num % plots_in_row > 0)

        if mode == 'roc':
            # plot ROC
            evaulate_func = sklearn.metrics.roc_curve
            score_func = sklearn.metrics.roc_auc_score
            score_label = "AUC"
            xlabel = 'False positives'
            ylabel = 'True positives'
            plt_title = "ROC"

        else:
            # plot Precision Recall
            evaulate_func = sklearn.metrics.precision_recall_curve
            score_func = sklearn.metrics.average_precision_score
            score_label = "AP"
            xlabel = 'Recall'
            ylabel = 'Precision'
            plt_title = "PR"

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
            # plt.subplot(num_plot_rows, plots_in_row, i)
            plt.figure(figsize=(11, 8))
            i -= 1
            for key in keys:
                if key not in out_predictions or key not in out_labels:
                    continue

                # print(i)
                # print(out_labels[key][i][:10])
                x, y, _ = evaulate_func(
                    out_labels[key][i], out_predictions[key][i])

                """
                out_predictions[key][i] = np.array(out_predictions[key][i])
                out_predictions[key][i][out_predictions[key][i]>0.5]=1 
                out_predictions[key][i][out_predictions[key][i]<=0.5]=0
                """
                score = score_func(out_labels[key][i], out_predictions[key][i])

                plt.plot(x, y, label="{}, {}: {:.3f}".format(
                    key, score_label, score))

            plt.xlabel(xlabel, fontsize=15)
            plt.ylabel(ylabel, fontsize=15)
            plt.xlim([-0.01, 1.01])
            plt.ylim([-0.01, 1.01])
            plt.grid(True)
            plt.legend(fontsize=14)

            plt.title("{} {}".format(labels_name[i], plt_title), fontsize=18)

            if save_path is not None:
                plt.savefig("{}/{} {} Curve.png".format(save_path,
                            labels_name[i], plt_title))

            plt.show()

        """     
             for i, out_predictions in enumerate(predictions[key+"_predictions"])
                  
                  fp, tp, _ = sklearn.metrics.roc_curve(labels[key+"_labels"][i], out_predictions) 
                        
        for output in outputs
        
            train_fp, train_tp, _ = sklearn.metrics.roc_curve(train_labels, train_predictions)
            train_fp, train_tp, _ = sklearn.metrics.roc_curve(train_labels, train_predictions)
            train_fp, train_tp, _ = sklearn.metrics.roc_curve(train_labels, train_predictions)
            
        
        """
