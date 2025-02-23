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

| Model               | Optimizer | Regularizer | Epochs | Early Stopping | Layers | Learning Rate | Accuracy | Loss  | Precision | Recall | F1-score | ROC-AUC |
|---------------------|-----------|-------------|--------|---------------|--------|--------------|----------|-------|----------|--------|----------|---------|
| CNN Model 1 (Base)  | None      | None        | 10     | No            | 8      | None         | 0.7888   | 0.4578 | 0.7878   | 0.7888 | 0.7882   | N/A     |
| CNN Model 2 (L1)    | Adam      | L1          | 10     | No            | 8      | None         | 0.6103   | 0.8636 | 0.3725   | 0.6103 | 0.4626   | N/A     |
| CNN Model 3 (L2)    | Adam      | L2          | 10     | No            | 8      | None         | 0.6684   | 0.6230 | 0.6592   | 0.6684 | 0.6467   | N/A     |
| CNN Model 4 (Full)  | Adam      | L2          | 10     | Yes           | 8      | 0.3          | 0.6768   | 0.6192 | 0.6703   | 0.6768 | 0.6532   | N/A     |
| Logistic Regression | -         | L2 (default)| -      | -             | -      | -            | 0.9882   | 0.0995 | 0.9927   | 0.9768 | 0.9847   | 0.9995  |

## Summary  

### **Best Model Among CNN Models :**  
The fully optimized **CNN Model 4 (Full)** performed better than the other CNNs, likely due to the use of **regularization and early stopping**.

### **Overall Best Model:**  
The **Logistic Regression model** outperformed all CNN models, achieving **98.82% accuracy** and **99.27% precision**. Given the importance of accuracy in healthcare applications, this result was prioritized.

### **Neural Networks vs. ML Algorithm:**  
- **Logistic Regression** achieved excellent accuracy and precision, making it more reliable for healthcare classification tasks. Although CNNs typically perform better on image classification, Logistic Regression's higher precision was crucial in avoiding false positives and negatives.
- The **fully optimized CNN model (CNN Model 4)** was the second-best performing model, showing that **optimization techniques** contributed to better performance.
- **Other CNN models struggled**, likely due to insufficient training epochs and possible overfitting. More training time and computational resources might have improved their performance.

## **Hyperparameters of Logistic Regression**  
- **Solver:** lbfgs  
- **Max Iterations:** 1000  
- **Regularization:** L2 (default, C=1.0)  
- **Loss Function:** Cross-entropy loss (log-loss)  

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

```
## Conclusion
Despite the expected dominance of CNNs in image classification, the **Logistic Regression model** performed best in this project. This suggests that, for this specific dataset, a simple model generalized better than deeper architectures. The second-best model, **CNN Model 4**, performed relatively well, but additional training resources (more epochs, a larger dataset) could have improved CNN performance.

## **What I Will Do Better Next Time:**
- ✅ Increase the number of epochs for CNN models to allow better learning.
- ✅ Use a **larger dataset** to see if CNNs can outperform Logistic Regression.
- ✅ Optimize CNN architectures further with techniques like **data augmentation and dropout layers**.

### **Thank You!**
