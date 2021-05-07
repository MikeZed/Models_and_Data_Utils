


"""
	Model Training Order:

	 1. only imgs, VGG19,    lr = 0.0005, 
	   with {}
	   no   {weight init, regulizers, DO, BN, rotation / flipping, whitened}


	 2. only imgs, VGG19,    lr = 0.00001, - Change layers to train 
	   with {weight init} 
	   no {regulizers, DO, BN, rotation / flipping, whitened}


	 3. only imgs, VGG19,    lr = 0.00001, 
	   with {weight init, regulizers, DO, BN} 
	   no {rotation / flipping, whitened}


	 4. only imgs, ResNet50, lr = 0.00001, 
	   with {weight init, regulizers, DO, BN} 
	   no {rotation / flipping, whitened}


	 5. only imgs, ResNet50, lr = 0.00001, 
	   with {weight init, regulizers, DO, BN, rotation / flipping, whitened} 
	   no {}


   6. mixed (standard), ResNet50, lr = 0.00001, 
	   with {weight init, regulizers, DO, BN, rotation / flipping, whitened} 
	   no {}


	 7. mixed (robust), ResNet50, lr = 0.00001, 
	   with {weight init, regulizers, DO, BN, rotation / flipping, whitened} 
	   no {}


	 8. mixed (min-max), ResNet50, lr = 0.00001, 
	   with {weight init, regulizers, DO, BN, rotation / flipping, whitened} 
	   no {}
 
   8.1. mixed (min-max), VGG19, lr = 0.00001, 
	   with {weight init, regulizers, DO, BN, rotation / flipping, whitened} 
	   no {}


	 9. mixed (min-max), ResNet50, lr = 0.00001, - Change Architecture 				
	   with {weight init, regulizers, DO, BN, rotation / flipping, whitened} 
	   no {}
 
 	 10. one label at a time mixed (min-max), ResNet50, lr = 0.00001, 
	   with {weight init, regulizers, DO, BN, rotation / flipping, whitened} 
	   no {} head0->head1->mid0->head2->head3
 
   10.1. one label at a time mixed (min-max), VGG19, lr = 0.00001, 
	   with {weight init, regulizers, DO, BN, rotation / flipping, whitened} 
	   no {} head0->head1->mid0->head2->head3
  





                                                     TO-DO                                                             
												  -----------
			* add support for numeric and mixed data in DataSequence:
					* img generator works 
					* check numeric generator 
					* check mixed data generator 
									  
			* take out get_labels part in load_data_generators to another function (add data split as property of Data)

			* change config files to .ini files for multiple configs support 

			* move evaluation to another class 

			* rearrange model evaulation and data code  


												Possible Additions
											 -----------------------
			* general data handling in DataSequence (multiple images, img name not in first column)

"""
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}


from ModelManager import create_model

from Model_and_Data_Configuration import *


import cv2
import matplotlib.pyplot as plt
# ---------------------------------------------------------------------
IMG_SETTINGS = {'img_res': IMAGE_RES, 'img_channels': IMG_CHANNELS, 'img_preprocessing': IMG_PREPROCESSING}           

# ---------------------------------------------------------------------

MODEL_DICT = {'optimizer': OPTIMIZER, 'loss': LOSS_FUNC, 'metrics': METRICS,
              'struct': MODEL_STRUCT if not USE_TRANSFER else TRANSFER_MODEL, 'layers_to_train': LAYERS_TO_TRAIN}

TRAINING_DICT = {'epochs': EPOCHS, 'batch_size': BATCH_SIZE}



# ------------
MODEL_SETTINGS = {'model_dict': MODEL_DICT,'save_model': SAVE_MODEL}

                                
DATA_SETTINGS = {'training_dict': TRAINING_DICT, 'train_val_test_split': TRAIN_VAL_TEST_SPLIT, 'plots_in_row': PLOTS_IN_ROW,
                'data_path': DATA_PATH, 'data_file': DATA_FILE, 'img_settings': IMG_SETTINGS}
       
       
def main():
 
    #img=cv2.imread("/home/michael/Cell_Classification/Code/Data_Preprocessing/Small_Windows_Whitened/D102_F001_C02.png",0)
    #plt.imshow(img*255, cmap='gray')
    
   # plt.show()
    
    Classifier = create_model(MODELS_DIR, MODEL_FILE, **MODEL_SETTINGS, **DATA_SETTINGS)

    print("Finishing program...")


if __name__ == "__main__":
    main()
