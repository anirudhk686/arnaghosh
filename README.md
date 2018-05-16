# Learning to dream from dreamers
This is an attempt to classify people who had a high dream recall rate from those who didn't based on their EEG data recorded during sleep using a deep convolutional network architecture. <br/>

## Aims
* Train a deep convolutional architecture to discriminate people with high dream recall rate (dreamers) from those with low dream recall rate (non-dreamers).
* Observe the decoding (classification) accuracies for different sleep stages to identify key sleep stage containing differences
* Visualize the features extracted by the CNN to classify the two groups
* Try to train on identifying individual subjects - maybe used as person identification tool

## Data
Courtesy - [CoCo Lab](http://www.karimjerbi.com/index.html)
* 36 subjects - 18 dreamers and 18 non-dreamers
* Sleep trials of 30 sec interval sampled at 1000 Hz
* Sleep trials segregated based on sleep stage (S1,S2,REM and SWS) as annotated by an expert
