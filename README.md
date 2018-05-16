# Learning to dream from dreamers
This is an attempt to classify people who had a high dream recall rate from those who didn't based on their EEG data recorded during sleep using a deep convolutional network architecture. <br/>
[More about me](https://arnaghosh.github.io/)

## Aims
* Train a deep convolutional architecture to discriminate people with high dream recall rate (dreamers) from those with low dream recall rate (non-dreamers).
* Observe the decoding (classification) accuracies for different sleep stages to identify key sleep stage containing differences
* Visualize the features extracted by the CNN to classify the two groups
* Try to train on identifying individual subjects - maybe used as person identification tool

## Dataset
Courtesy - [CoCo Lab](http://www.karimjerbi.com/index.html)
* 36 subjects - 18 dreamers and 18 non-dreamers
* Sleep trials of 30 sec interval sampled at 1000 Hz
* 19 EEG electrodes
* Sleep trials segregated based on sleep stage (S1,S2,REM and SWS) as annotated by an expert

## Tools/Language used
* MATLAB
* Torch (Lua)

## Data Processing and Structuring
- [X] Data downsampled to 200 Hz
- [X] Check for anti-aliasing
- [X] Sleep trials split into trials of 5 sec long each
- [X] Data dimension structured to be 1 X 19 X 1000 to pass to the CNN

## Deep Learning architecture
- [X] Train network on SWS sleep data. 80% training data from bag of sleep segments, 20% for validaton
- [ ] Train network on 34 subjects and test on remaining 2 subjects
- [ ] Use REM and S2 data. S1 data may not be enough!! :/
- [X] Train a subject identifier
- [ ] Choose parameters for good subject classifier

## Visualizing features
- [ ] Use GAP
- [ ] Use gradCAM/deep dream
- [ ] Use ccCAM (method being developed)

## Current issues
* Good accuracy at sleep segment level, but not at subject level
* Write code for identifying subjects

## Further analysis
- [ ] Interpret neurological basis of extracted features
- [ ] Learning how people dream or what affects dream recall
