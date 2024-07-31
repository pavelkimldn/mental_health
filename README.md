# mental_health

**Mental Health Prediction Using Decision Trees**

**Introduction*

The objective of this project is to build a machine learning model using decision trees to predict whether a person will seek treatment for mental health issues based on various work environment factors. By leveraging a dataset from a mental health survey, I aimed to understand the correlations between work-related factors and the likelihood of seeking mental health treatment.

*Methodology**

Exploratory Data Analysis (EDA)
To begin, I conducted an exploratory data analysis to understand the dataset's structure and the relationships between different variables. The initial step involved data cleaning, where I filtered out non-binary gender responses and handled missing values using the forward fill method. For visualization, I created a bar chart showing the age distribution of respondents. Additionally, pie charts were generated to depict the distribution of respondents by gender and country. According to these charts, 83.6% of respondents were male and 16.4% were female, with most participants hailing from the USA, followed by the UK and Canada. A correlation heatmap was also created to visualize the correlations between different variables after encoding categorical variables numerically. Furthermore, I checked for class imbalance in the treatment variable and found that the classes were balanced.

![github](https://github.com/pavelkimldn/mental_health/blob/main/1001.png)

![github](https://github.com/pavelkimldn/mental_health/blob/main/1002.png)

![github](https://github.com/pavelkimldn/mental_health/blob/main/1003.png)

![github](https://github.com/pavelkimldn/mental_health/blob/main/1004.png)

![github](https://github.com/pavelkimldn/mental_health/blob/main/1005.png)

*Model Building*

For model building, I used a decision tree classifier to predict the treatment variable. All variables except treatment were used as features, with treatment serving as the target variable. Data preprocessing involved one-hot encoding for categorical variables and standard scaling. I created a pipeline with a column transformer, scaler, feature selector, and decision tree classifier. A grid search with 5-fold cross-validation was performed to find the best hyperparameters for the model, including 1440 iterations. The best model was then trained on the training data and evaluated on the test data.

*Findings*

From the correlation heatmap, the strongest correlations with seeking mental health treatment were identified as whether the company is a tech company, whether the respondent believes discussing a physical health issue with their employer would have negative consequences, and whether the respondent is willing to discuss a mental health issue with their direct supervisor. The decision tree model was evaluated using several metrics. The confusion matrix indicated a relatively balanced performance with both classes, with true positives (TP) at 59, true negatives (TN) at 55, false positives (FP) at 18, and false negatives (FN) at 16. The ROC curve showed an area under the curve (AUC) of 0.84, indicating a good level of discrimination between the classes. Cross-validation scores showed some variability across folds, with accuracy ranging from approximately 0.72 to 0.79, highlighting the model's robustness.

![github](https://github.com/pavelkimldn/mental_health/blob/main/1006.png)

![github](https://github.com/pavelkimldn/mental_health/blob/main/1007.png)

![github](https://github.com/pavelkimldn/mental_health/blob/main/1008.png)

*Conclusion*

This project demonstrates the use of decision tree classifiers to predict mental health treatment-seeking behavior based on work environment factors. Key findings include the significant impact of the work environment, particularly in tech companies, and the perceived consequences of discussing health issues on seeking mental health treatment. The model's performance, as evidenced by the confusion matrix, ROC curve, and cross-validation scores, indicates that it can be a useful tool for predicting mental health treatment-seeking behavior.


![github](https://github.com/pavelkimldn/Computer_Vision_Live_Emotion_Detection/blob/main/1.png)
