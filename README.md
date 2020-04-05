# Models and Data Utils
Previously was part of the Skew_Detection project.
As a result of Deep Learning models creation and training being a repeated task of Machine Learning Projects,
this small repository was created for:
- Downloading the data set 
- Basic data preproccesing
- Creating, training and testing a Deep Learning model 

A small brief of each file purpose: 
- Data - used for downloading, preparing and loading of the data.

- DataSequence - used for creating data generators, useful when dealing with large data sets 
                 that can't be entirely loaded into the memory.
                 
- image_utils - offers basic image processing such as image padding and resizing, 
                extracting patches of the image.
         
- Model - used to create, train and test models. can also load and save trained models, can plot training process, 
          and supports data generators and transfer learning 
          
          
 Note - this project was tested only with the Skew_Detection project, may require additional tweaks in order to work with other data. 
                
                 




