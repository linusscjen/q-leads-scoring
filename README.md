# Leads Scoring

This repo contains analysis and modeling for [this Kaggle dataset](https://www.kaggle.com/datasets/amritachatterjee09/lead-scoring-dataset?select=Lead+Scoring.csv). For context, X Education (an education company selling online courses to industry professionals) is hoping to increase its conversion rate from ~30% to ~80%. Our goal is to create a prospect list of people likely to convert.

## Setup

Data should be stored under the `src/data` directory and titled `Leads Scoring.csv`

Packages can be downloaded from the `requirements.txt` file.

## Logic

Data exploration was done under `eda/feature_eda.ipynb`. Modeling and minor analysis of the model was done under `model_data.ipynb`.

From the Kaggle website (and hints provided by Bo), the goal of the company was to improve the conversion rate from ~30% to ~80%, making precision the key metric here. However, we used ROC as our main metric to optimize, and planned to tweak the threshold for what probabilities would be required for the model to predict an observation as `Converted`.

In general, we found ~13 columns that did not contain any meaningful information and dropped those. Many categorical variables were found to have categories with little observations, and they were aggregated together. To deal with class imabalances, we used `SMOTE` and undersampling on the training data, and train our model(s) on that dataset. I used `XGBoost` given how successful this method tends to be with ML problems. The first model had an accuracy and precision of 95%. Because one-hot encoding of categorical variables led to a total of 107 variables, we used `RFECV` to reduce this number to about 82 total variables.

Key variables include `Tags`, `Total Time Spent on Website`, `Last Notable Activity` and `Last Activity`. This aligns strongly with our findings in the `feature_eda.ipynb`.

## Presentation

TODO:

A PDF version of our Powerpoint can be found here in the repo under ________.