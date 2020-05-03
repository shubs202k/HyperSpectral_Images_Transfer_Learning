Folder contains code for transfer learning models for all possible combinations of source and target dataset amongst pavia, indian pines and salinas hyper spectral datasets. Same dataset is also used as both source and target. Thus there are 9 combinations of source and target datasets.

Code for each combination of source and target outputs a table with test accuracies for different overlapping ratios for source and target as the final results

50,75,85,95 are the different overlapping ratios used.

The folder also contains a Transfer_Model_Utils python file which has
all the helper functions to extract samples, create cubes, assign labels for each cube, prepare data for training and the transfer function which does the transfer learning between source and target dataset and outputs the test accuracies for all combinations of overlapping ratios.
