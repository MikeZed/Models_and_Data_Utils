

from Data import Data
from Model import Model
import pandas as pd
import os

YES = ['y', 'Y', '1', ' ']


def create_model(models_dir, model_file, model_dict=None, save_model=True, img_settings=None, use_generator=True, 
                 data_path=None, data_file=None, training_dict=None, train_val_test_split=(80,10,10), plots_in_row=3): 
                 
                 
    # checks if there is an existing model,
    # can create a new one by downloading and loading the data, creating a new model, training it and then
    # plots the results and saves them

    # ---------------------------------------
    #            check existing model
    # ---------------------------------------

    # print(MODEL_DICT)

    use_existing, path = prep_and_check_existing(models_dir=models_dir, model_file=model_file) 
    
    # -------------------------------------------------
    #     load existing model or create a new one
    # -------------------------------------------------
           
    settings = {'model': None, 'model_dict': model_dict,  'save_model': save_model, 'img_settings': img_settings,
                'training_dict': training_dict, 'train_val_test_split': train_val_test_split, 'use_generator': use_generator,
                'data_path': data_path, 'data_file': data_file, 'save_path': path, 'plots_in_row': plots_in_row}
                 
    if not use_existing:
        # build and train new model

        model = Model()
        settings['model'] = model
        
        load_data_and_construct_model(**settings)

    else:
        # use existing model

        model = Model.load_model(path)
        settings['model'] = model 
        
        continue_training = input("Continue training model? (Y/[N]): ")
        continue_training = True if continue_training in YES else False

        if continue_training:
            load_data_and_construct_model(**settings)

        print(model.model.metrics_names)
        

    return model

# ------------------------------------------------------------------------------------------------------------------- #

def load_data_and_construct_model(model, model_dict, save_model, img_settings, training_dict, train_val_test_split, 
                                  use_generator, data_path, data_file, save_path, plots_in_row):
        
    if not save_model:
        save_path = None  
          
    # load data
    data_loader = Data(use_generator=use_generator, data_path=data_path, data_file=data_file, img_settings=img_settings)
                      
    # load dataframe
    data_loader.load_dataframe()
  
    # load the data to the memory / create data generators
    data_dict = data_loader.load_data(batch_size=training_dict['batch_size'], split=train_val_test_split)
    
    # create and train model
    model.construct(**model_dict, **training_dict, use_generator=use_generator, data_dict=data_dict, save_path=save_path)
    
    # plot results
    model.plot_train_val_history(save_path=save_path, plots_in_row=plots_in_row)
    
    data_dict, labels = data_loader.load_data(get_labels=True, batch_size=training_dict['batch_size'], split=train_val_test_split)
   
    predictions = model.predict(data_dict)
    
    model.evaluate_classifier(predictions=predictions, labels=labels, mode='roc', save_path=save_path, plots_in_row=plots_in_row)
    #model.evaluate_classifier(predictions=predictions, labels=labels, mode='pr',  save_path=save_path, plots_in_row=plots_in_row)

def prep_and_check_existing(models_dir, model_file):
    # check if there is already an existing model at path
    use_existing = False
    
    path = os.path.join(models_dir, model_file)

    if not os.path.exists(models_dir):
        os.mkdir(models_dir)

    # if not os.path.exists(GRAPHS_DIR):
    #     os.mkdir(GRAPHS_DIR)

    if os.path.exists(path):
        use_existing = input("A trained model already exists: \nUse existing one? (Y/[N]): ")
        use_existing = True if use_existing in YES else False

        if not use_existing:
            i = 1
            while os.path.exists(path + ' ({})'.format(i)):
                i += 1

            os.rename(path, path + ' ({})'.format(i))

    return use_existing, path 
      
      
    






