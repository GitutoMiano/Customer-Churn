#!/usr/bin/env python
# coding: utf-8

# # Customer Churn Prediction Model
# 

# In[ ]:


import pandas as pd


# In[19]:


from sklearn.model_selection import train_test_split


# In[20]:


data = pd.read_csv('C:/Users/user/Desktop/sprint_data1.csv')


# In[21]:


data.head()


# # Data Processing:
# 
# Dealing with:
#  
# 1. Missing values
# 2. Duplicate records
# 3. Outliers
# 4. Encoding categorical variables (Employment status, Gender)
# 5. Normalizing numerical variables
# #Normalization is the process of scaling numeric variables to a standard range (usually between 0 and 1) to ensure that they have a similar influence on machine learning algorithms.
# 6. Creating a time-based dataset based on available data.
# 

# In[22]:


# 1. Find Missing Values
missing_values = data.isnull().sum()
print("Missing Values:")
print(missing_values)


# In[23]:


# 2. Find Duplicate Records
duplicate_records = data[data.duplicated()]
print("\nDuplicate Records:")
print(duplicate_records)


# In[24]:


# 3. find outliers (using box plot)
# first import relevant libraries
import seaborn as sns
import matplotlib.pyplot as plt
# Create a boxplot
sns.set(style="whitegrid")  # Set the style of the plot
plt.figure(figsize=(8, 6))  # Set the figure size
# Create subplots for each boxplot
fig, axes = plt.subplots(2, 3, figsize=(15, 8))  # Create a 2x3 grid of subplots

# Create boxplots for each variable
sns.boxplot(y='Monthly Data Usage (GB)', data=data, ax=axes[0, 0])
sns.boxplot(y='Contract Duration (months)', data=data, ax=axes[0, 1])
sns.boxplot(y='Billing Information (Monthly Bill, $)', data=data, ax=axes[1, 0])
sns.boxplot(y='Customer Complaints (No. of Complaints)', data=data, ax=axes[1, 1])
sns.boxplot(y='Competitor Pricing Group by Area', data=data, ax=axes[1, 2])

# Adjust subplot layout
plt.tight_layout()

# Show the subplots
plt.show()


# In[25]:


# Encoding categorical variables (Employment status, Gender)
# Perform one-hot encoding for 'Employment Status' and 'Gender'
data_encoded = pd.get_dummies(data, columns=['Employment Status', 'Gender','Customer Churn'])
data_encoded.head()


# In[26]:


# 5. Normalizing numerical variables
from sklearn.preprocessing import MinMaxScaler

# Select the columns to normalize
columns_to_normalize = [
    'Billing Information (Monthly Bill, $)',
    'Customer Complaints (No. of Complaints)',
    'Contract Duration (months)',
    'Monthly Data Usage (GB)','Competitor Pricing Group by Area',
    'Employment Rate by Area of Residence']

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Apply normalization to the selected columns
data_encoded[columns_to_normalize] = scaler.fit_transform(data_encoded[columns_to_normalize])
data_encoded.head()


# In[27]:


# 6. Creating a time-based dataset based on available data.

import numpy as np

# Define the start and end dates for your time-based dataset
start_month = '2015-09'
end_month = '2023-09'

# Generate a sequence of dates between the start and end dates
month_range = pd.date_range(start=start_month, end=end_month, freq='M')  # 'M' for monthly frequency

# Create a DataFrame with the generated dates
time_df = pd.DataFrame({'Month': month_range})

# Repeat existing data for each month
existing_data = data_encoded.sample(n=len(month_range), replace=True)  
#Since I am using len(month_range) as the number of samples to draw, and month_range contains 96 unique months (from '2015-09' to '2023-09'), the sample method will randomly select 96 rows from data_encoded, and some rows may be selected multiple times due to replacement.
# Concatenate the time-based DataFrame with the existing data
final_dataset = pd.concat([time_df, existing_data.reset_index(drop=True)], axis=1)

summary = final_dataset.describe()
# Display the summary statistics
print(summary)
final_dataset.head(100)


# # Creating the model
# 1. split the data into train and test sets
# 2. Initialize and train the XGBoost model
# 3. Make predictions
# 4. Evaluate the model
# 

# In[28]:


# first find the correlating variables
import matplotlib.pyplot as plt

# Calculate the correlation matrix
correlation_matrix = final_dataset.corr()

# Create a heatmap to visualize the correlations
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Heatmap')
plt.show()

variables = final_dataset.columns
print(variables)


# In[38]:


#1. split the data into train and test sets
get_ipython().system('pip install xgboost')
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

X = final_dataset[[
       'Customer Complaints (No. of Complaints)',
       'Competitor Pricing Group by Area',
       'Employment Rate by Area of Residence',
       'Employment Status_Self-employed','Employment Status_Employed','Employment Status_Unemployed',
       'Gender_Agender','Gender_Male','Gender_Polygender']]

Y = final_dataset['Customer Churn_True']
# Split the data into test and train
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#2. Initialize and train the XGBoost model
model = xgb.XGBClassifier()
model.fit(X_train, Y_train)

#3. make predictions
y_pred=model.predict(X_test)



# In[39]:


#4. Evaluating the model
accuracy = accuracy_score(Y_test,y_pred)
precision = precision_score(Y_test,y_pred)
recall = recall_score(Y_test,y_pred)
f1 = f1_score(Y_test,y_pred)
roc_auc = roc_auc_score(Y_test,model.predict_proba(X_test)[:,1])
print(accuracy)
print(precision)
print(recall)
print(f1)
print(roc_auc)


# # Conclusion
# 
# The accepted values for accuracy, precision, ROC AUC, F1-score, and recall can vary depending on the specific problem, dataset, and business objectives. 
# 
# Accuracy:
# Accuracy measures the overall correctness of predictions.
# Typically, higher accuracy is desirable, but it should be evaluated in the context of the problem.
# The acceptable range for accuracy depends on the specific application but is often above 80% for many binary classification tasks.
# 
# Precision:
# Precision focuses on minimizing false positives.
# A higher precision indicates fewer false alarms.
# The acceptable value for precision depends on the cost associated with false positives. In some cases, a precision of 90% or higher may be desired.
# 
# ROC AUC (Receiver Operating Characteristic Area Under the Curve):
# ROC AUC measures the model's ability to distinguish between positive and negative classes.
# A higher ROC AUC score suggests better discrimination.
# ROC AUC values can range from 0.5 (random classifier) to 1 (perfect classifier). Values above 0.7 are often considered acceptable.
# 
# F1-Score:
# The F1-score is the harmonic mean of precision and recall.
# It balances precision and recall.
# An F1-score close to 1 is desirable, indicating a good balance between precision and recall.
# Recall (Sensitivity or True Positive Rate):
# 
# Recall focuses on minimizing false negatives.
# A higher recall indicates that the model captures more true positive cases.
# The acceptable value for recall depends on the importance of minimizing false negatives. In some cases, a recall of 90% or higher may be desired.
# It's important to note that these metrics are often interrelated. Improving one metric may come at the expense of another. The choice of which metrics to prioritize and what values are acceptable should be guided by the specific goals and constraints of your project.
# 
# # Sprint conclusion
# The model indicates that variables such as ('Customer Complaints (No. of Complaints)',
#        'Competitor Pricing Group by Area',
#        'Employment Rate by Area of Residence',
#        'Employment Status_Self-employed','Employment Status_Employed','Employment Status_Unemployed',
#        'Gender_Agender','Gender_Male','Gender_Polygender') are crucial in determining the probability of a customer terminating their service relations with Sprint company. 
