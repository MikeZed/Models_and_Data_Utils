# Models and Data Utils
Previously was part of the Skew_Detection project.
As a result of - Deep Learning models creation and training - being a repeated task of Machine Learning Projects,
this small repository was created for:
- Downloading the data set 
- Basic data preproccesing
- Creating, training and testing a Deep Learning model 

A brief of each file purpose: 
- Data - used for downloading, preparing and loading of the data.

- DataSequence - used for creating data generators, useful when dealing with large data sets 
                 that can't be entirely loaded into the memory.
                 
- image_utils - offers basic image processing such as image padding and resizing, 
                extracting patches from the image.
         
- Model - used to create, train and test models. can also load and save trained models, can plot training process, 
          and supports data generators and transfer learning 

- Model_and_Data_Configuration - holds all the parameters of the model such as the loss function, metrics, optimizer, structure. 
                                 also contains the necessary paths to access the data.  
                                 
- model_creation - receives all the nessary information from Model_and_Data_Configuration.
                   loads the dataset by using the Data Class and feeds it to a new model according to Model_and_Data_Configuration. 
          
                
                 




