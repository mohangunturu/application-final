### What leads to divorce?

MG

### Executive summary
This research is going to help couples proactively to prevent/fix/take necessary steps to avoid divorce based on their answers to the questions in the dataset.

### Rationale

Families serve as the foundational units of society, playing a crucial role in the well-being and development of individuals. Here are several reasons why supporting and nurturing families is essential for the stability and success of society:

* Social Stability:Families contribute to social stability by providing a stable environment for individuals, fostering a sense of belonging and security.
* Emotional Support:Families offer emotional support, care, and love, helping individuals cope with life's challenges and build resilience.
* Child Development:Families are primary in shaping the early development of children. A supportive family environment positively influences a child's cognitive, emotional, and social development.
* Values and Morality:Families often serve as the primary source for transmitting values, ethics, and cultural traditions. They play a crucial role in instilling a sense of morality and responsibility in individuals.
* Education and Learning:Families are the first educators, providing a foundation for learning and critical thinking. A supportive family environment encourages curiosity and a love for learning.
* Economic Stability:Strong families contribute to economic stability. Supportive family structures can provide financial stability, allowing individuals to focus on personal and professional development.
* Community Building:Families are building blocks for communities. Strong families contribute to the development of healthy and supportive communities.

### Research Question
This research will help couples to predict the success rate of their marriage and help them proactively.

### Data Sources

This the dataset that was collected from Kaggle from the the below url.

https://www.kaggle.com/datasets/andrewmvd/divorce-prediction

### Methodology
I have used data exploration, data analysis, data cleaning, visualizations, handling missing and duplicate values, splitting the train and test data, and finally modeling by using classification and deep learning models.

#### Outline of project
The following steps are performed on this data.

1. Data Exploration
    *  Check for Data Quality
    *  Missing Values Check
    *  Zero Values Check
    *  Unique Values Check
    *  Duplicate Values Check
    *  Outlier Values Check
2. Data Visualization & Analysis
    * Visualizing Outliers
    * Visualizing Correlations of Features
3. Data Preparation
    * Data Cleanup
    * Handling Missing Values
    * Handling Outliers
4. Feature Engineering
    * Correlation method
    * Random Forest Classifier method
5. Modeling
    * Data split for traing and testing
    * Classification Models
    * Cross Validation
    * Grid Search Hyper Parameters Tuning
6. Ensemble Techniques
   * Voting Classifier and Multiple Classifiers
7. Deep Learning
   * Sequential
   * LSTM and Dense

#### Summary and Interpretation of Models:
* Features: I have applied correlation and Random forest classifier to select the features that compares target variable with all the features. I have picked top 10 features that predicts that explains at least 90% variance. Using these top 10 features, I have analyzed the data with several models and compared the performance.

### Classification Models
  * I have applied Logistic, Random forest, Decision tree and SV Classification algorithms.
  ### Results
    
<p>Logistic Regression Accuracy: 0.9655172413793104
<pre><code>Classification Report:
precision    recall  f1-score   support</p>
<pre><code>       0       0.94      1.00      0.97        17
       1       1.00      0.92      0.96        12

accuracy                           0.97        29
</code></pre>
<p>macro avg       0.97      0.96      0.96        29
weighted avg       0.97      0.97      0.97        29</p>
<p>Random Forest Accuracy: 0.9655172413793104
Classification Report:
precision    recall  f1-score   support</p>
<pre><code>       0       0.94      1.00      0.97        17
       1       1.00      0.92      0.96        12

accuracy                           0.97        29
</code></pre>
<p>macro avg       0.97      0.96      0.96        29
weighted avg       0.97      0.97      0.97        29</p>
<p>Decision Tree Accuracy: 1.0
Classification Report:
precision    recall  f1-score   support</p>
<pre><code>       0       1.00      1.00      1.00        17
       1       1.00      1.00      1.00        12

accuracy                           1.00        29
</code></pre>
<p>macro avg       1.00      1.00      1.00        29
weighted avg       1.00      1.00      1.00        29</p>
<p>SVC Accuracy: 0.9655172413793104
Classification Report:
precision    recall  f1-score   support</p>
<pre><code>       0       0.94      1.00      0.97        17
       1       1.00      0.92      0.96        12

accuracy                           0.97        29
</code></pre>
<p>macro avg       0.97      0.96      0.96        29
weighted avg       0.97      0.97      0.97        29</p>
</code></pre>

*  <p>Precision is the ratio of correctly predicted positive observations to the total predicted positives, recall is predicted observations on total actual postives. Decision tree performed much better compared to other modesl with accuracy of 100% and follwed by the rest of the models with accuracy of 97%. </p>

</code></pre>




### Cross Validation
<p>Classification Results:</br>
Model  Accuracy (Mean)  Accuracy (Std)</br>
0  Logistic Regression Classification         0.982609        0.021300</br>
1        Random Forest Classification         0.973913        0.034783</br>
2        Decision Tree Classification         0.965217        0.032536</br>
3                  SVC Classification         0.982609        0.021300</p>

<p>SVC and Logistic Regression performed better with highest accuracy of predicting the outcomes than other models, suggests that this is a better classification model for this data. 98% of the time, the predictions are correct.</p>

### Grid Search Hyper Parameters Tuning
There are several paramters that were tried and finally, the best parameters for both regression and classifications are below.
<p>Best Hyperparameters (Logistic Regression): {'C': 1}</br>
Best Hyperparameters (Random Forest Classification): {'max_depth': None, 'n_estimators': 50}</br>
Best Hyperparameters (Decision Tree Classification): {'criterion': 'gini', 'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2}</br>
Best Hyperparameters (SVC Classification): {'C': 0.1, 'degree': 2, 'gamma': 'auto', 'kernel': 'rbf'}</p>

### Ensemble Techniques
Classifiers:
Accuracy: 0.97
<p>Voter Classifier has a accuracy of 97%, able to predict correctly 97% of the time</p>

Classification Report:</br>
              precision    recall  f1-score   support</br>

           0       0.94      1.00      0.97        17</br>
           1       1.00      0.92      0.96        12</br>

    accuracy                           0.97        29</br>
   macro avg       0.97      0.96      0.96        29</br>
weighted avg       0.97      0.97      0.97        29</br>


Confusion Matrix:</br>
[[17  0]</br>
 [ 1 11]]</br>
               Classifier  Accuracy  Precision    Recall  Training Time (s)</br>
0      LogisticRegression  0.965517   0.967433  0.965517           0.002027</br>
1  RandomForestClassifier  0.965517   0.967433  0.965517           0.042244</br>
2  DecisionTreeClassifier  1.000000   1.000000  1.000000           0.000637</br>
3                     SVC  0.965517   0.967433  0.965517           0.000804</br>
<p> I have also performed time calculations on these models, Logistic regression aand SVC classifiers are the better models with less time and close to actual values in predictions </p>

### Deep Learning
<p> I have applied Sequential and LSTM networks, LSTM performed better with MSE 0% with more than 99% of explainable of variance. This network 3 layers, 243 neurons and, 27049 parameters.</p>
1/1 [==============================] - 0s 57ms/step
Mean Squared Error (MSE): 0.00
R-squared (R2): 0.98
Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 lstm (LSTM)                 (None, 50)                12200     
                                                                 
 dense_3 (Dense)             (None, 128)               6528      
                                                                 
 dense_4 (Dense)             (None, 64)                8256      
                                                                 
 dense_5 (Dense)             (None, 1)                 65        
                                                                 
=================================================================
Total params: 27049 (105.66 KB)
Trainable params: 27049 (105.66 KB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________

### Results: 
<p> LSTM , SVC,and Logistic Regression are the best models for predicting the divorce close to 98% of the time with the 10 features that were selected. </p>
Below are 10 features. Select one option for each question to attain a predictive accuracy of 98% in predicting divorces using two widely-used and models that we explored out of many algorithms: Logistic Regression and LSTM network.

All responses were collected on a 5 point scale (0=Never, 1=Seldom, 2=Averagely, 3=Frequently, 4=Always).

1.My spouse and I have similar ideas about how marriage should be</br>
2.We're just starting a discussion before I know what's going on.</br>
 3.think that one day in the future, when I look back, I see that my spouse and I have been in harmony with each other</br>
4.We share the same views about being happy in our life with my spouse</br>
5.My spouse and I have similar values in trust</br>
6.My spouse and I have similar ideas about how roles should be in marriage</br>
7.I know my spouse's hopes and wishes</br>
8.I know my spouse's basic anxieties</br>
9.I can be humiliating when we discussions</br>
10.Most of our goals for people (children, friends, etc.) are the same.</br>

### Next Steps and Recommendations

* Visualizations:
    * I like to visualize more on overfitting, underfitting, bias and variance, other techniques to understand better

* Evaluate Model Performance:

    * I want to assess the  performance of these predictive models (Logistic Regression and LSTM network). Use metrics such as accuracy, precision, recall, and F1 score to understand how well the models are performing on predicting divorces.

* Feature Importance Analysis:

    * I like to explore through feature importance analysis to understand which features have the most significant impact on the predictions. This can provide insights into the factors influencing divorce predictions.

* Model Interpretability:

    * I want to enhance the interpretability of these models to understand the reasons behind specific predictions.
      
* Error Analysis:

    * I like to analyze model errors and misclassifications. Identify patterns or trends in instances where the models are struggling to make accurate predictions, it may offer insights to get more data points.
* Fine-Tuning and Optimization:

    * I like to fine-tune these models based on the insights gained from the evaluation and analysis.(Ex: adjusting hyperparameters, experimenting with different feature sets, or exploring advanced techniques to optimize model performance).

* Validation on New Data:

    * I want to validate these models on new, unseen data to ensure that they generalize well, so that we can make sure the model is not overfitting.
* Deployment Strategy:

    * I want to create deployment strategy of my models. May be AWS or something else, I will leverage for this
* Continuous Monitoring:

    * I want to establish continuous monitoring of these models performance.
      
* Feedback Loop:

    * I want establish a feedback loop with end-users :) . 


#### Please find the link below to codebase.
- [Note book](https://github.com/mohangunturu/application-final/blob/main/prompt.ipynb)


#### Contact and Further Information
Contact information not provided for privacy reasons.