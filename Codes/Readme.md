# Step-by-step instructions to run the codes in this project

## Data prep
1. Get the mat files corresponding to each subject and each sleep stage
2. Run Data/prepData.m file. This shall generate required hdf5 files after downsampling data from 1k Hz to 200 Hz and splitting each 30 sec sleep trial into 5 sec sleep segments
3. The new data files should be hdf5 format. These files are needed by the training files. 
