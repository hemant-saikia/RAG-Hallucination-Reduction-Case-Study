# Machine Learning Basics

## Supervised Learning
Supervised learning uses labeled training data to learn a mapping from inputs to outputs. Common algorithms include linear regression, logistic regression, decision trees, random forests, and support vector machines. The goal is to minimize the error between predictions and actual labels on the training data while maintaining good performance on unseen data.

## Unsupervised Learning
Unsupervised learning finds patterns in unlabeled data. Techniques include clustering (K-means, hierarchical), dimensionality reduction (PCA, t-SNE), and anomaly detection. These algorithms are used when the data has no predefined categories.

## Bias-Variance Tradeoff
The bias-variance tradeoff is a fundamental concept in machine learning. High bias leads to underfitting where the model is too simple to capture the underlying patterns. High variance leads to overfitting where the model memorizes the training data including noise. The goal is to find the optimal model complexity that minimizes total error. 

Regularization techniques like L1 (Lasso) and L2 (Ridge) help control overfitting by adding penalty terms to the loss function. Cross-validation is used to tune these hyperparameters effectively.

## Decision Trees
Decision trees split data recursively based on feature values. They are prone to overfitting and are often used as ensemble components. Random forests average multiple decision trees to reduce variance and improve generalization. Gradient boosting builds trees sequentially where each new tree corrects errors from previous ones.

## Support Vector Machines
SVMs find the hyperplane that maximally separates classes. The kernel trick allows SVMs to handle non-linear decision boundaries by projecting data into higher-dimensional spaces. Common kernels include linear, polynomial, and radial basis function (RBF). SVMs work well with high-dimensional data but can be computationally expensive for large datasets.

## Model Evaluation
Common evaluation metrics include accuracy, precision, recall, F1-score, and ROC-AUC. Confusion matrices provide a detailed breakdown of prediction errors. The choice of metric depends on the problem domain and the costs of different types of errors.

## Feature Engineering
Feature engineering transforms raw data into better representations for learning. Techniques include normalization, standardization, one-hot encoding, and feature creation. Proper feature engineering often matters more than the choice of algorithm for model performance.

## The Curse of Dimensionality
As the number of features increases, the data becomes sparse in high-dimensional space. This makes distance metrics less meaningful and requires exponentially more data to achieve the same level of statistical significance. Dimensionality reduction techniques help mitigate this problem.

## Ensemble Methods
Ensemble methods combine multiple models to achieve better performance than any individual model. Bagging (Bootstrap Aggregating) trains models in parallel on different bootstrap samples. Boosting trains models sequentially, focusing on previously misclassified examples. Stacking trains a meta-learner to combine predictions from base models.

## Transfer Learning
Transfer learning leverages knowledge from a pre-trained model and adapts it to a new but related task. This is especially useful when labeled data for the target task is scarce. Fine-tuning adjusts the pre-trained model weights on the new data while preserving previously learned features.
