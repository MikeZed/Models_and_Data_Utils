
# fmt: off
import numpy as np 
from functools import reduce
import matplotlib.pyplot as plt
import cv2
import os

from Model_Code.Model import Model
from Model_Code.Data import Data 
from Model_Code.ModelEvaluator import ModelEvaluator
from Model_Comparison import * 

# fmt: on
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}


def main():
    model_dirs = sorted((os.listdir(MODELS_PATH)))

    models = [Model.load_model(os.path.join(MODELS_PATH, path))
              for path in model_dirs]

    # load data
    data_loader = Data(data_path=DATA_PATH,
                       data_file=DATA_FILE, img_settings=IMG_SETTINGS)

    # load dataframe
    data_loader.load_dataframe()

    # load the data to the memory / create data generators
    data_dict = data_loader.load_data(
        batch_size=BATCH_SIZE, split=TRAIN_VAL_TEST_SPLIT, mode='val')

    # iterate thorugh predictions and multiply them in a new variable called combined.
    #
    labels = data_loader.get_labels(split=TRAIN_VAL_TEST_SPLIT)
    predictions_train = [model.evaluate_model(
        data_dict['train']) for model in models]
    predictions_val = [model.evaluate_model(
        data_dict['val']) for model in models]

    for i in range(len(predictions_val)):
        print(model_dirs[i], predictions_val[i])

    '''
    predictions_train = [model.predict(data_dict['train']) for model in models]
    predictions_val = [model.predict(data_dict['val']) for model in models]

    combined_train = np.array(reduce(lambda x, y: x*y, predictions_train))
    combined_val = np.array(reduce(lambda x, y: x*y, predictions_val))

    tempLabels = labels["val"][0]
    tempCombined = np.around(combined_val)
    sum = 0
    for i in range(len(tempLabels)):
        sum += (tempLabels[i] == tempCombined[i][0])

    print("123")
    sum = sum/len(tempLabels)
    print(sum)
    print("567")
    combined_train = np.ndarray.tolist(combined_train ** (1/3))
    combined_val = np.ndarray.tolist(combined_val ** (1/3))

    combined_dict = {
        "train": list(zip(*combined_train)), "val": list(zip(*combined_val))}

    ModelEvaluator.evaluate_classifier(predictions=combined_dict, labels=labels,
                                       labels_name=DATA_FILE["used_labels"], mode='roc', save_path=GRAPHS_SAVE_PATH)

    ModelEvaluator.evaluate_classifier(predictions=combined_dict, labels=labels,
                                       labels_name=DATA_FILE["used_labels"], mode='pr',  save_path=GRAPHS_SAVE_PATH)
    '''

    print("Finishing program...")


if __name__ == "__main__":
    main()
