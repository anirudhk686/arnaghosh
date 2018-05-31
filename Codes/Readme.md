# Step-by-step instructions to run the codes in this project

## Data prep
1. Get the mat files corresponding to each subject and each sleep stage
2. Run Data/prepData.m file. This shall generate required hdf5 files after downsampling data from 1k Hz to 200 Hz and splitting each 30 sec sleep trial into 5 sec sleep segments
3. The new data files should be hdf5 format. These files are needed by the training files. 

## Deep Learning
1. The model description can be found and changed in model.lua file. The file presently contains multiple model architectures that were used at some time during the project. 
2. To test the training paradigm and to understand the hyperparameter selection, users may run trainModel.lua file. This file selects 80% of all sleep segments as training set and the rest 20% as validation for dreamer vs non-dreamer classification.
3. To run a cross-validation with 34 subjects in training set and 2 subjects in validation set for dreamer vs non-dreamer classification, use the trainModel_subj.lua file.
4. For the subject ID problem, use the trainModel_identifySubj.lua file.

## LogFiles and Images
1. All the results obtained are stored as Logfiles from various experiments and relevant figures are located in the Images folder
