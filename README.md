# Text-level-classification

## Project Overview
This project develops a machine learning model to predict the difficulty level of French texts, enabling English speakers learning French to access reading materials that are appropriate for their proficiency level. Hosted as part of a [Kaggle competition](https://www.kaggle.com/competitions/predicting-the-difficulty-of-a-french-text-e4s/overview), my model classifies texts into CEFR levels from A1 to C2 to support personalized learning experiences.

## Project Objective
The objective of this project is to provide learners with an automated tool that identifies the difficulty level of French texts. By doing so, learners can engage with materials that are neither too easy nor too hard, enhancing both learning efficiency and engagement.

## Data Utilization
I use labeled training data and (sometimes) the unlabeled test data provided by the competition organizers to train and validate our models. These models are designed to streamline the assessment of text difficulty for educational purposes.

## Technical Development
My approach includes:
- **Initial Use of Classical Machine Learning Algorithms**: Before employing more complex models, I first used traditional machine learning algorithms such as Logistic Regression, K-Nearest Neighbors (KNN), Decision Trees, and Random Forests to establish a baseline for performance. These algorithms are well-known for their effectiveness in various classification tasks and provided initial insights into the challenge of classifying text difficulty.
- **Multi-input Neural Network**: Analyzes linguistic features to predict text difficulty.
- **Fine-tuning of the Camembert Model**: Adapts this robust language model to better understand the nuances of French texts, improving its predictive accuracy.

## Methodology Overview

My project applies several machine learning algorithms to predict the difficulty level of French texts. The methodologies I adopted include Logistic Regression, K-Nearest Neighbors (KNN), Decision Trees, and Random Forest. Each model was evaluated based on its accuracy and further analyzed using confusion matrices. Here’s a step-by-step breakdown of my approach:

### Data Preprocessing

Before applying the models, I first analyzed the distribution of data. This was important to ensure that the training data did not favor certain classes over others, which could bias the models' performance.

**Bar Plot Visualization**:
I generated a bar plot to visually confirm that the data across the different classes (difficulty levels from A1 to C2) is evenly distributed. This plot is crucial as it underpins my confidence in the subsequent model training and validation phases.

![Bar Plot of Data Distribution](path/to/bar_plot_image.png)

### Model Training and Validation

I employed four different machine learning models to understand their efficacy in classifying text difficulties. Here is how each model was applied:

#### Logistic Regression
- **Approach**: A simple yet powerful linear model used for classification tasks.
- **Rationale**: To establish a baseline for performance due to its ease of implementation and interpretation.

#### K-Nearest Neighbors (KNN)
- **Approach**: A non-parametric method that classifies data points based on the majority label of its nearest neighbors.
- **Rationale**: To leverage the similarity of text feature vectors for classification, assuming that similar difficulty texts lie close to each other in feature space.

#### Decision Tree
- **Approach**: A flowchart-like tree structure where each internal node represents a "test" on an attribute, each branch represents the outcome of the test, and each leaf node represents a class label.
- **Rationale**: To capture non-linear patterns in the data, which could be indicative of varying text complexities.

#### Random Forest
- **Approach**: An ensemble method using multiple decision trees to improve classification accuracy and control over-fitting.
- **Rationale**: To enhance the robustness of decision trees by averaging multiple trees that individually consider random subsets of features and samples.

### Model Evaluation

Each model's performance was quantitatively assessed using accuracy metrics. Accuracy measures the proportion of total correct predictions (both true positives and true negatives) relative to the total dataset.

**Confusion Matrix Visualization**:
For each model, a confusion matrix was generated to provide insights into the type and frequency of classification errors. This matrix helps in understanding model performance across different classes, highlighting potential biases or weaknesses in classification.

![Confusion Matrix for Logistic Regression](path/to/confusion_matrix_lr.png)
![Confusion Matrix for KNN](path/to/confusion_matrix_knn.png)
![Confusion Matrix for Decision Tree](path/to/confusion_matrix_dt.png)
![Confusion Matrix for Random Forest](path/to/confusion_matrix_rf.png)

The methodology outlined ensures a thorough understanding and comparison of how different algorithms perform on the task of predicting text difficulty. This structured approach helps in identifying the most effective model and in making informed decisions to further improve the model performance.
