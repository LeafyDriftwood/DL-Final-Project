# DL-Final-Project
CS 1470 final project

This model aims to detect the presence of suicidal intent in tweets by analyzing past activity to arrive at a decision. The train.py file contains all the code to 
train and evaluate the model, which is implemented in model.py. The preprocessed data is in the data folder, which was given alongside the project. 

Our final model has a recall of 1, a precision of 0.73, and an F1 score of 0.84. In order to replicate the results, please run the train.py file. NOTE: we have followed the procedure for selecting train and eval data exactly as the actual paper, where there are sometimes overlaps. If you would like to ensure the eval data is completely unique, you can restrict the bounds of the train data to (0, 80) instead of (0, 100) and (80, 100) instead of (0, 100) for the eval data in the train_loop and eval_loop functions respectively. 
