# Glaucoma Image Classification Project

## Overview  
This project aims to classify fundus images (retinal images) into two distinct categories (Glaucoma Positive or Glaucoma Negative) to aid in early detection of glaucoma. The dataset consists of labeled images for training, validation, and testing. The goal was to build an effective model for predicting glaucoma.

## Problem Statement  
Glaucoma early detection is a major challenge in low-resource countries, such as those in Sub-Saharan Africa. This project aims to create a Glaucoma Early Prediction Model with high accuracy, as early diagnosis can significantly improve patient treatment and outcomes.

## Dataset  
The dataset contains retinal images stored in three directories: train, test, and val. These images are labeled according to the condition they represent. The dataset was sourced from Fundus Image Dataset on Kaggle : https://www.kaggle.com/datasets/sabari50312/fundus-pytorch

## Video Link  
_(Click Here)_

## Findings  

| Instance | Model               | Optimizer | Regularizer | Epochs | Early Stopping | Layers | Learning Rate | Accuracy | Loss  | Precision | Recall | F1-score | ROC-AUC |
|----------|---------------------|-----------|-------------|--------|----------------|--------|---------------|----------|-------|----------|--------|----------|---------|
| 1        | CNN Model 1 (Base)  | None      | None        | 10     | No             | 8      | None          | 0.7888   | 0.4578| 0.7878   | 0.7888 | 0.7882   | N/A     |
| 2        | CNN Model 2 (L1)    | Adam      | L1          | 10     | No             | 8      | None          | 0.6103   | 0.8636| 0.3725   | 0.6103 | 0.4626   | N/A     |
| 3        | CNN Model 3 (L2)    | Adam      | L2          | 10     | No             | 8      | None          | 0.6684   | 0.6230| 0.6592   | 0.6684 | 0.6467   | N/A     |
| 4        | CNN Model 4 (Full)  | Adam      | L2          | 10     | Yes            | 8      | 0.3           | 0.6768   | 0.6192| 0.6703   | 0.6768 | 0.6532   | N/A     |
| 5        | Logistic Regression | -         | L2 (default)| -      | -              | -      | -             | 0.9882   | 0.0995| 0.9927   | 0.9768 | 0.9847   | 0.9995  |

## Summary  

### **Best Model Among CNN Models :**  
The **CNN Model 1 (Base)** performed the best among the CNN models, with an accuracy of **78.88%**. This model did not use any regularization, optimization, or advanced techniques such as early stopping, which contributed to its ability to learn more effectively from the data. The simplicity allowed it to better fit the dataset and achieve higher accuracy compared to other CNNs.

### **Overall Best Model:**  
The **Logistic Regression model** outperformed all CNN models, achieving **98.82% accuracy** and **99.27% precision**. This result highlights the power of simpler models in specific cases, particularly in healthcare applications where accuracy and precision are critical to avoid false positives and negatives.

### **Neural Networks vs. ML Algorithm:**  
- **Logistic Regression** achieved excellent accuracy and precision, making it a more reliable choice for healthcare classification tasks in this case. Even though CNNs generally excel in image classification tasks, Logistic Regression was more reliable here due to its higher precision.
- The **fully optimized CNN model (CNN Model 4)** did not perform as well as expected, with an accuracy of **67.68%**. Despite using regularization and early stopping techniques, the model’s accuracy was lower than **CNN Model 1**. This suggests that the added complexity of optimizations may have led to underfitting in this case.
- **CNN Model 2 (L1)** and **CNN Model 3 (L2)** also underperformed compared to the base model, likely due to insufficient training time and overfitting in some cases.

## **Hyperparameters of Logistic Regression**  
- **Solver:** lbfgs (Limited-memory Broyden-Fletcher-Goldfarb-Shanno, suitable for smaller datasets)  
- **Max Iterations:** 1000 (enough to ensure the model converges and learns well)  
- **Regularization:** L2 (default, C=1.0) - Regularization prevents overfitting and ensures the model generalizes well.  
- **Loss Function:** Cross-entropy loss (log-loss) - This is typically used for binary classification problems like this one, where the goal is to minimize the difference between the predicted and actual probabilities.

## **Instructions on How to Run the Best Saved Model**  
1. Clone my repository
2. Navigate to `/intro_to_ml_summative/saved_models/`
3. Load the model using `pickle` 
4. Prepare data and load it
5. Make predictions

```python
import pickle

# Load the logistic regression model
with open('logistic_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)
````
## Conclusion

Despite the expected better results of CNNs in image classification, the Logistic Regression model performed best in my project. This suggests that, for this specific dataset, a simple model generalized better than deeper and complex architectures. The second-best model, CNN Model 1, achieved the highest accuracy but lacked regularization or advanced techniques. It showed me  that for this dataset, simpler models may outperform complex neural networks. 

## What I Will Do Better Next Time:
- ✅ Increase the number of epochs for CNN models to allow better learning.
- ✅ Use a larger dataset instead of using the  subset l used for the project to see if CNNs can outperform Logistic Regression.
- ✅ Optimize CNN architectures further with techniques like data augmentation .

### Thank You!
