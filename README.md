# Text-level-classification

## Project Overview
This project develops a machine learning model to predict the difficulty level of French texts, enabling English speakers learning French to access reading materials that are appropriate for their proficiency level. Hosted as part of a [Kaggle competition](https://www.kaggle.com/competitions/predicting-the-difficulty-of-a-french-text-e4s/overview), my model classifies texts into CEFR levels from A1 to C2 to support personalized learning experiences.

## Project Objective
The objective of this project is to provide learners with an automated tool that identifies the difficulty level of French texts. By doing so, learners can engage with materials that are neither too easy nor too hard, enhancing both learning efficiency and engagement.

## Data Utilization
I use labeled training data and (sometimes) the unlabeled test data provided by the competition organizers to train and validate our models. These models are designed to streamline the assessment of text difficulty for educational purposes.

### Data Augmentation Attempt
I experimented with data augmentation to increase the diversity and amount of training data, hoping to enhance the model's ability to generalize across unseen texts. However, these efforts did not yield the expected improvements in performance. This outcome indicates that there is substantial room for developing more effective augmentation strategies that could better address the challenges of text difficulty classification.

## Technical Development
My approach includes:
- **Initial Use of Classical Machine Learning Algorithms**: Before employing more complex models, I first used traditional machine learning algorithms such as Logistic Regression, K-Nearest Neighbors (KNN), Decision Trees, and Random Forests to establish a baseline for performance. These algorithms are well-known for their effectiveness in various classification tasks and provided initial insights into the challenge of classifying text difficulty.
- **Multi-input Neural Network**: Analyzes linguistic features to predict text difficulty.
- **Fine-tuning of the Camembert Model**: Adapts this robust language model to better understand the nuances of French texts, improving its predictive accuracy.

## Method 1 - ML algorithms

My project applies several machine learning algorithms to predict the difficulty level of French texts. The methodologies I adopted include Logistic Regression, K-Nearest Neighbors (KNN), Decision Trees, and Random Forest. Each model was evaluated based on its accuracy and further analyzed using confusion matrices. Here’s a step-by-step breakdown of my approach:

### Data Preprocessing

Before applying the models, I first analyzed the distribution of data. This was important to ensure that the training data did not favor certain classes over others, which could bias the models' performance.

**Bar Plot Visualization**:
I generated a bar plot to visually confirm that the data across the different classes (difficulty levels from A1 to C2) is evenly distributed. This plot is crucial as it underpins my confidence in the subsequent model training and validation phases.

![Bar Plot of Data Distribution](path/to/bar_plot_image.png)

### Model Training and Validation

#### Logistic Regression
- **Approach**: A simple yet powerful linear model used for classification tasks.
- **Evaluation Score**: [Include evaluation score here]
- **Confusion Matrix Visualization**:
-  ![Confusion Matrix for Logistic Regression](path/to/confusion_matrix_lr.png)

#### K-Nearest Neighbors (KNN)
- **Approach**: A non-parametric method that classifies data points based on the majority label of its nearest neighbors.
- **Evaluation Score**: [Include evaluation score here]
- **Confusion Matrix Visualization**:
-  ![Confusion Matrix for KNN](path/to/confusion_matrix_knn.png)

#### Decision Tree
- **Approach**: A flowchart-like tree structure where each internal node represents a "test" on an attribute, each branch represents the outcome of the test, and each leaf node represents a class label.
- **Evaluation Score**: [Include evaluation score here]
- **Confusion Matrix Visualization**:
-  ![Confusion Matrix for Decision Tree](path/to/confusion_matrix_dt.png)

#### Random Forest
- **Approach**: An ensemble method using multiple decision trees to improve classification accuracy and control over-fitting.
- **Evaluation Score**: [Include evaluation score here]
- **Confusion Matrix Visualization**:
-  ![Confusion Matrix for Random Forest](path/to/confusion_matrix_rf.png)

The methodology outlined ensures a thorough understanding and comparison of how different algorithms perform on the task of predicting text difficulty. This structured approach helps in identifying the most effective model and in making informed decisions to further improve the model performance.
The methodology outlined ensures a thorough understanding and comparison of how different algorithms perform on the task of predicting text difficulty. This structured approach helps in identifying the most effective model and in making informed decisions to further improve the model performance.

## Method 2 - Neural Network with Feature Engineering and Optimization

#### Step 1: Tokenization and Embedding
- **Tokenization Techniques**: I employed the Mistral API to tokenize the French texts, which is specifically designed for the French language, ensuring that the nuances and morphological aspects of French are accurately captured. Following tokenization, I used the Camembert embedding models to convert text tokens into vector representations. Camembert is a powerful language model based on the RoBERTa architecture, adapted and pre-trained specifically for the French language, making it ideal for this task.
- **Evaluation of Embedding Techniques**: Both the Mistral tokenization and Camembert embeddings were evaluated to determine their effectiveness in capturing linguistic features. The evaluations showed that both methods provided similarly robust results, indicating that our embedding process was capable of capturing comprehensive semantic and syntactic properties of the input texts.

#### Step 2: Feature Extraction and Selection
- **Extraction of Linguistic Features**: Post-embedding, I proceeded to extract a wide range of linguistic features from the sentences. These features included syntactic dependencies, part-of-speech tags, and other text-based features that could potentially signal text complexity.
- **Feature Selection Using Random Forest**: To refine the feature set, I utilized a Random Forest classifier, a robust ensemble learning method known for its high accuracy and control over overfitting. I used the classifier to identify the most predictive features, focusing on selecting the top three features that had the most significant impact on model performance. This selective approach helped in reducing dimensionality and improving the efficiency of the subsequent training process.

#### Step 3: Initial PCA Attempt and Reassessment
- **Dimensionality Reduction via PCA**: In an effort to simplify the model, Principal Component Analysis (PCA) is applied to reduce the dimensionality of the feature set.
- **Reevaluation of PCA**: After assessing the impact of PCA on model performance, I find it detrimental to prediction accuracy as critical information is lost. Consequently, PCA is removed from the process.

#### Step 4: Neural Network Configuration and Training
- **Constructing the Neural Network**: Utilizing TensorFlow’s Keras library, I build a neural network designed to integrate both tokenized numerical features and embedded text efficiently.

#### Step 5: Hyperparameter Tuning and Optimization
- **Hypermodel Training**: To achieve the best possible model performance, I utilized hypermodel training to systematically test a range of hyperparameters. This included adjustments to learning rate, weight decay, batch size, embedding unit sizes, numeral features units, and the dimensions of the final output layers. By automating the search for the optimal configuration, I was able to significantly enhance the learning process and overall model accuracy.

#### Step 6: Model Evaluation
- **Accuracy Measurement**: The performance of the final model is quantified through its accuracy metric, reflecting how well the model predicts the difficulty levels of new texts.
- **Confusion Matrix for Insight**: A confusion matrix is generated to visualize the model's performance across different difficulty classes, highlighting successes and areas for potential improvement.

![Confusion Matrix for Neural Network Model](path/to/confusion_matrix_nn.png)

This comprehensive, multi-stage approach not only optimized the predictive accuracy but also provided deep insights into the textual features most indicative of French text difficulty. By iterating over various techniques and optimizations, the methodology refined the predictive capabilities of the neural network, setting a robust foundation for practical application in educational technologies.

## Method 3 - Fine-Tuning Camembert Model with Transformers Library

#### Overview
In this approach, I leverage the power of the Camembert model, a transformer model adapted specifically for the French language, to predict the difficulty of French texts. This method involves fine-tuning the pre-trained Camembert on our dataset to tailor it more closely to the specific task of text difficulty classification.

#### Training Process
- **Model Selection**: I chose the Camembert model due to its robust performance on French language tasks and its capability to understand nuanced linguistic features.
- **Fine-Tuning with Trainer**: I utilized the Trainer from the Hugging Face Transformers library to fine-tune the Camembert model on our dataset. The Trainer is a feature within the library that simplifies the training process by handling many of the routine tasks involved in training transformer models.

#### Model Optimization
- **Trial and Error Method**: The process of achieving good accuracy involved trial and error, where I manually adjusted hyperparameters and model configurations. While this method is not the most efficient form of optimization, it allowed for hands-on adjustments that gradually improved the model's performance based on empirical results.

#### Model Evaluation
- **Evaluation Accuracy**: After several iterations, the fine-tuned model achieved an evaluation accuracy of [Insert evaluation accuracy here]. This accuracy indicates the model's proficiency in classifying the difficulty levels of French texts.
- **Confusion Matrix Visualization**: The confusion matrix provides a visual representation of the model's performance across various text difficulty categories, identifying strengths and pinpointing areas where the model may struggle.

![Confusion Matrix for Camembert Model](path/to/confusion_matrix_camembert.png)

This method highlights the practical application of advanced NLP techniques like transformers in specialized tasks such as classifying text difficulty. Through a hands-on, iterative approach to fine-tuning, the model was adapted to perform effectively in real-world scenarios.

## Best Method and Further Improvements

#### Best Performing Method
The best results were achieved through the fine-tuning of the Camembert model. This approach proved highly effective for the specific task of classifying the difficulty of French texts. The advanced capabilities of the transformer-based Camembert model allowed it to capture and utilize the complex nuances of the French language more efficiently than the other methods.

#### Strategies for Improvement
- **Further Hyperparameter Tuning**: More systematic and extensive hyperparameter tuning could yield better results. Utilizing tools like grid search or Bayesian optimization could help in finding the optimal set of parameters more efficiently than the trial and error method used.
- **Refine Data Augmentation Techniques**: Although initial attempts at data augmentation did not significantly improve results, this area holds potential for enhancement. Exploring more sophisticated text augmentation techniques such as synonym replacement, back-translation, or using contextual word replacements provided by models like BERT could lead to a more effective increase in data variability and model robustness.
- **Ensemble Methods**: Combining the predictions from multiple models, including those fine-tuned with different subsets of data or using different NLP techniques, could enhance accuracy and reliability. Ensemble methods often lead to better generalization on complex tasks like text classification.

### Conclusion
In conclusion, while the fine-tuning of the Camembert model delivered the best results among the methods I tried, there remains significant scope for enhancement. Systematic hyperparameter optimization, refining data augmentation strategies, and exploring ensemble methods are potential strategies that could further boost the model’s performance. Each step taken in this project builds on our understanding of applying machine learning to language processing, highlighting both the potential and the challenges of NLP tasks. Moving forward, implementing these improvements could lead to even more accurate and robust models for classifying text difficulty, ultimately making this tool more useful for learners of French.
