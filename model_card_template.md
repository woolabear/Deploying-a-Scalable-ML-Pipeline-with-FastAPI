# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model was developed by Christina Brooman in October of 2026. It is a Random Forest Classifier model using the default hyperparameters in scikit-learn 1.5.1. The model analyzes data from 8 categorical columns (workclass, education, marital-status, occupation, relationship, race, sex, and native-country), 6 numerical columns (age, fnlwgt, education-num, capital-gain, capital-loss, and hours-per-week), and one binarized salary column (salary). Categorical features were transformed via one-hot-encoding, and the salary column used a label binarizer.

## Intended Use
This model predicts whether a person's income will be over $50,000 based on data obtained from the U.S. Census Adult Income dataset. It is for educational use and analysis, not for real hiring, lending, or benefits decisions.

## Training Data
The dataset contains 32,561 records and is split into 80% training and 20% testing.

## Evaluation Data
Evaluation was performed on the 20% test split.

## Metrics
Precision: 0.7391 | Recall: 0.6384 | F1: 0.6851

## Ethical Considerations
There are other categories that have not been evaluated that can contribute to a person's income. Also, because this is from another year, it might not be as relevant today. There are also possible fairness bias or factual concerns involving historical census data as well.

## Caveats and Recommendations
Looking at datasets from other years, or more current years, could build a bigger picture, because the trends we find might not even be relevant in more recent years. Also, considering other factors like in which state a person lives, the population, or the unemployment rate in that year could provide more understanding.
