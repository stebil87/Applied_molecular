## BBB prediction
This repository contains our final project in Applied Case Studies of ML and DL in Key Areas 2 (prof. Gianvito Grasso), BSc in Data Science and AI, SUPSI AY 2023-24.
The folders data, result and utils contain the functions used in the analysis, along with some results, while the executable jupyter notebook is named "BBB classification", where in the last cell the user can interact with the model, getting explainable prediction.
The code is also equipped with a util named "rfcv" which provides a visual analysis on the entire dataset about feature importance, is can be also used for feature selection. Its usage is left to the interested user, the function is already implemented and only needs a dataframe as an input. 
The user can also change the features generated, jus changing the parameters in "feature_types" or change the type of NN used with exmol, just changing the "drug_encoding" as per exmol documentation and code, in our showcase a MLP was used.
A report can also be found.

We hope you can have fun with it! 

