### What leads to divorce?

MG

#### Executive summary
This research is going to help couples proactively to prevent/fix/take necessary steps to avoid divorce based on their answers to the questions in the dataset.

#### Rationale

Families serve as the foundational units of society, playing a crucial role in the well-being and development of individuals. Here are several reasons why supporting and nurturing families is essential for the stability and success of society:

* Social Stability:Families contribute to social stability by providing a stable environment for individuals, fostering a sense of belonging and security.
* Emotional Support:Families offer emotional support, care, and love, helping individuals cope with life's challenges and build resilience.
* Child Development:Families are primary in shaping the early development of children. A supportive family environment positively influences a child's cognitive, emotional, and social development.
* Values and Morality:Families often serve as the primary source for transmitting values, ethics, and cultural traditions. They play a crucial role in instilling a sense of morality and responsibility in individuals.
* Education and Learning:Families are the first educators, providing a foundation for learning and critical thinking. A supportive family environment encourages curiosity and a love for learning.
* Economic Stability:Strong families contribute to economic stability. Supportive family structures can provide financial stability, allowing individuals to focus on personal and professional development.
* Community Building:Families are building blocks for communities. Strong families contribute to the development of healthy and supportive communities.

#### Research Question
This research will help couples to predict the success rate of their marriage and help them proactively.

#### Data Sources

This the dataset that was collected from Kaggle from the the below url.

https://www.kaggle.com/datasets/andrewmvd/divorce-prediction

#### Methodology
I have used data exploration, data analysis, data cleaning, visualizations, handling missing and duplicate values, splitting the train and test data, and finally modeling by using regression models and classification models.

#### Results
My research finds that data helps to predict the divorces with above 90% of accuracy given a spouse answers the questions unbiased.

#### Next steps
Analyze this with more visualizations on overfitting, underfitting, bias and variance, other techniques to achieve best accuracy

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
    * Regression Models
    * Classification Models
    * Cross Validation
    * Grid Search Hyper Parameters Tuning
6. Ensemble Techniques
   * Voting Regressor and Multiple Regressors
   * Voting Classifier and Multiple Classifiers
7. Deep Learning
   * Sequential
   * LSTM and Dense

#### Summary and Interpretation of Models:
* Features: I have applied correlation and Random forest classifier to select the features that compares target variable with all the features. I have picked top 10 features that predicts that explains atleast 90% variance. Using these top 10 features, I have analyzed the data with several models and compared the performance.
### Regression Models
  * I have applied Linear, Random forest, Polynomial and SV Regression algorithms.
  ### Results
    * Mean Squared Error (Linear Regression): 0.029111439108339715
    * Mean Squared Error (Random Forest Regression): 0.023634482758620692
    * Polynomial Regression (Degree 3) MSE: 0.4662792904085586
    * SVR MSE: 0.03277362141855743
  * To calculate MSE, you take the squared difference between each predicted and actual value, sum these squared differences, and then divide by the number of observations.MSE of zero is a perfect model which predicts all values correct. Random Forest Regressor has low MSE compare to other regression models, MSE of 0.0236 suggests relatively good performance on the training data. The next best model is Linear regression. Polynomial with Degree 3 suggets that model is overfit with Degree 3 with error rate of 46%. This is a bad model with degree 3.

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
<p>
*  Precision is the ratio of correctly predicted positive observations to the total predicted positives, recall is predicted observations on total actual postives. Decision tree performed much better compared to other modesl with accuracy of 100% and follwed by the rest of the models with accuracy of 97%. </p>

### Cross Validation

<p>Regression Results:
Model  R2 (Mean)  R2 (Std)
0         Linear Regression   0.918330  0.018782
1  Random Forest Regression   0.921108  0.054572
2     Polynomial Regression   0.711600  0.120574
3            SVR Regression   0.907840  0.026842</p>
Random Forest is able to explain 92% of variance, and it has lowest standard deviation suggests the model is better than all other regressions.
<p>Classification Results:
Model  Accuracy (Mean)  Accuracy (Std)
0  Logistic Regression Classification         0.982609        0.021300
1        Random Forest Classification         0.973913        0.034783
2        Decision Tree Classification         0.965217        0.032536
3                  SVC Classification         0.982609        0.021300</p>

<p>SVC and Logistic Regression performed better with highest accuracy of predicting the outcomes than other models, suggests that this is a better classification model for this data. 98% of the time, the predictions are correct.</p>

### Grid Search Hyper Parameters Tuning
There are several paramters that were tried and finally, the best parameters for both regression and classifications are below.
<p>Best Hyperparameters (Linear Regression): {'fit_intercept': True}
Best Hyperparameters (Random Forest Regression): {'max_depth': 20, 'n_estimators': 200}
Best Hyperparameters (Polynomial Regression): {'max_depth': 20, 'n_estimators': 200}
Best Hyperparameters (SVR Regression): {'max_depth': 20, 'n_estimators': 200}
Best Hyperparameters (Logistic Regression): {'C': 1}
Best Hyperparameters (Random Forest Classification): {'max_depth': None, 'n_estimators': 50}
Best Hyperparameters (Decision Tree Classification): {'criterion': 'gini', 'max_depth': None, 'min_samples_leaf': 4, 'min_samples_split': 5}
Best Hyperparameters (SVC Classification): {'C': 0.1, 'degree': 2, 'gamma': 'auto', 'kernel': 'rbf'}</p>

### Ensemble Techniques
Regressors:
Mean Squared Error (MSE): 0.07
R-squared (R2): 0.73
<p>Voter Regressor has a MSE of 7% with 73% of explained variance</p>
               Regressor       MSE        R2  Training Time (s)
0       LinearRegression  0.029111  0.879987           0.000559
1  RandomForestRegressor  0.020214  0.916668           0.037138
2               Pipeline  0.519156 -1.140244           0.001531
3                    SVR  0.032752  0.864978           0.002296

Classifiers:
Accuracy: 0.97
<p>Voter Classifer has a accuracy of 97%, able to predict correctly 97% of the time</p>

Classification Report:
              precision    recall  f1-score   support

           0       0.94      1.00      0.97        17
           1       1.00      0.92      0.96        12

    accuracy                           0.97        29
   macro avg       0.97      0.96      0.96        29
weighted avg       0.97      0.97      0.97        29


Confusion Matrix:
[[17  0]
 [ 1 11]]
               Classifier  Accuracy  Precision    Recall  Training Time (s)
0      LogisticRegression  0.965517   0.967433  0.965517           0.001508
1  RandomForestClassifier  0.965517   0.967433  0.965517           0.043153
2  DecisionTreeClassifier  1.000000   1.000000  1.000000           0.000567
3                     SVC  0.965517   0.967433  0.965517           0.000832

<p> I have also performed time calculations on these models, Random Forest Regressor and Decision tree classifier are the better models with less time and close to actual values in predictions </p>

### Deep Learning
<p> I have applied Sequential and LSTM networks, LSTM performed better with MSE 0% with 98% of explainablity of variance.</p>

### Winner: 
<p> LSTM and Logistic Regression are the best models for predicting the divorce close to 98% of the time with the 10 features that were selected. </p>
The 10 features are below with answering one of the choice for each question.

All responses were collected on a 5 point scale (0=Never, 1=Seldom, 2=Averagely, 3=Frequently, 4=Always).

1.My spouse and I have similar ideas about how marriage should be
2.We're just starting a discussion before I know what's going on.
 3.think that one day in the future, when I look back, I see that my spouse and I have been in harmony with each other
4.We share the same views about being happy in our life with my spouse
5.My spouse and I have similar values in trust
6.My spouse and I have similar ideas about how roles should be in marriage
7.I know my spouse's hopes and wishes
8.I know my spouse's basic anxieties
9.I can be humiliating when we discussions
10.Most of our goals for people (children, friends, etc.) are the same.



#### Please find the link below to codebase.
- [Note book](https://github.com/mohangunturu/application-final/blob/main/prompt.ipynb)


#### Contact and Further Information
Contact information not provided for privacy reasons.