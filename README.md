# CSCI3329_HW3_Mushroom
Homework 3 Mushroom Classification
CSCI 3329 - Homework 3 Report

1. Dataset

I used the Mushroom dataset from the UCI Machine Learning Repository.

- Original Samples: 8124
- Samples after cleaning: 5644
- Features: 22
- Classes: 2 (Poisonous and Edible)

The goal of this assignment was to classify mushrooms as edible (e) or poisonous (p) based on their physical characteristics.

---

2. Preprocessing

I had to manually add column names since they were not originally included in the dataset documentation. 

Missing values were represented by '?' and were removed by replacing them with null  values and dropping these rows.

All features in the dataset were categorical, so I used LabelEncoder to convert each category into numerical values.

I also used StandardScaler to scale the features. This was crucial because models like KNN and Neural Network work better when the data is scaled.

---

3. Part 2 Algorithm Comparison
| Algorithm | Mean Accuracy | Std |
| --------- | ------------- | --- |
| Linear Classifier | 0.9857 | 0.0166 |
| Logistic Regression | 0.9887 | 0.0045 |
| KNN | 1.0000 | 0.0000 |
| Gaussian NB | 0.6277 | 0.0176 |
| Neural Network | 1.0000 | 0.0000 |

Discussion

The best performing models were KNN and Neural Network, both of these models were 100% accurate. This indicates that they were perfectly able to classify all mushrooms across all validation runs.

This suggests that the dataset contained very strong patterns that could clearly separate edible and poisonous mushrooms.

Logistic Regression and the Linear Classifier also performed well, with accuracies above 98%. This indicated that the dataset was possibly somewhat linearly separable.

Gaussian Naive Bayes performed significantly worse, with an accuracy of about 62%.This is because the algorithm assumes that all features are independent, which is not the case for this dataset. 

In order to reduce runtime, I used RepeatedKFold with 10 splits and 10 repeats instead of 100 repeats.

---

4. Part 3 - Feature Selection

I used forward feature selection to determine the most important features for each model.

Due to runtime limitation with Google Colab, I used a faster version of forward selection by selecting the best 3 features for each model and using 3-fold cross validation.

---

| Algorithm | Best Feature Subset | Mean Accuracy | Std |
| --------- | ------------------- | ------------- | --- |
| Linear Classifier | bruises, stalk-shape, spore-print-color | 0.8554 | 0.0540 |
| Logistic Regression | gill-size, ring number, spore-print-color | 0.9584 | 0.0090 |
| KNN | cap-color, odor, spore-print-color | 1.000 | 0.000 |
| Gaussian NB | gill-spacing, gill-size, spore-print-color | 0.9440 | 0.0060 |
| Neural Network | cap-color, odor, spore-print-color | 1.0000 | 0.0000 |

Discussion 

Feature selection showed high accuracy despite using only a small number of features.

Gaussian Naive Bayes improved significantly compared to Part 2, which suggests that the removal of less useful features helped its assumptions of feature independence.

In conclusion, feature selection helped reduce the complexity of the model while maintaining strong performance.

---

5. Reproduction 

Python version: 3.10+

Libraries used:
- pandas
- numpy
- scikit-learn

To run the project:

```bash
python main.py 
