#!/usr/bin/env python
# coding: utf-8

# # Import Library

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
import joblib
import pickle


# # Read dataset

# In[10]:


df = pd.read_csv('flightdata.csv')
df.head()


# # Read Info

# In[11]:


df.info()


# #    Univariate analysis using Pie chart

# In[8]:


df['YEAR'].value_counts().plot(kind='pie', autopct='%.0f')
plt.show()


# #    Bivariate analysis using scatterplot

# In[15]:


plt.scatter(df.DEP_DELAY, df.DEP_DEL15)
plt.title('Departure Delay Analysis')
plt.xlabel('DEP_DELAY')
plt.ylabel('DEP_DEL15')
plt.show()


plt.scatter(df.DEP_DELAY, df.DEP_DEL15)
plt.title('Arrival Delay Analysis')
plt.xlabel('ARR_DELAY')
plt.ylabel('ARR_DEL15')
plt.show()


# # Bivariate analysis using lineplot

# In[16]:


fig, ax = plt.subplots(figsize=(10, 3))
plt.subplot(1, 2, 1)
plt.title('CRS_DEP_TIME')
plt.plot(df.CRS_DEP_TIME)
plt.subplot(1, 2, 2)
plt.title('DEP_TIME')
plt.plot(df.DEP_TIME)
plt.show()



# In[17]:


fig, ax = plt.subplots(figsize=(10, 3))
plt.subplot(1, 2, 1)
plt.title('CRS_ARR_TIME')
plt.plot(df.CRS_ARR_TIME)
plt.subplot(1, 2, 2)
plt.title('ARR_TIME')
plt.plot(df.ARR_TIME)
plt.show()


# In[19]:


fig, ax = plt.subplots(figsize=(10, 3))
plt.subplot(1, 2, 1)
plt.title('CRS_ELAPSED_TIME')
plt.plot(df.CRS_ELAPSED_TIME)
plt.subplot(1, 2, 2)
plt.title('ACTUAL_ELAPSED_TIME')
plt.plot(df.ACTUAL_ELAPSED_TIME)
plt.show()


# # Performing multivariate analysis

# using pairplot

# In[23]:


sb.pairplot(df.iloc[:, 12:])
plt.show()


# Using heatmap

# In[24]:


fig, ax = plt.subplots(figsize=(15, 10))
sb.heatmap(df.iloc[:, 12:].corr(), annot=True, ax=ax)
plt.show()


# # Performing Descriptive Analysis

# In[25]:


df.describe()


# # Dropping Unnecessary Columns

# In[26]:


df = df[['FL_NUM', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'ORIGIN', 'DEST', 'DEP_DEL15', 'CRS_ARR_TIME', 'ARR_DEL15']]
df.head()


# # Handling Missing Values

# Checking for Null Values

# In[27]:


df.isnull().any()


# Replacing Null Values

# In[29]:


df.fillna(df['DEP_DEL15'].mode()[0], inplace=True)
df.fillna(df['ARR_DEL15'].mode()[0], inplace=True)


# Checking if the replacement is made

# In[30]:


df.isnull().any()


# # Handling Outliers

# In[31]:


fig, ax = plt.subplots(figsize=(5, 6))
sb.boxplot(data=df['CRS_ARR_TIME'])
plt.show()


# # Encoding

# One hot encoding

# In[32]:


df = pd.get_dummies(df, columns=['ORIGIN', 'DEST'])
df.head()

df.columns


# # Splitting dataset into Independent and Dependent Variables

# In[33]:


X = df.drop(columns=['ARR_DEL15'])
Y = df[['ARR_DEL15']]


# # Converting the Independent and Dependent Variables to 1D Arrays

# In[34]:


X = X.values
Y = Y.values


# # Splitting dataset into Train and Test datasets

# In[35]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


# In[36]:


X_train.shape, X_test.shape, Y_train.shape, Y_test.shape


# Logistic Regression

# In[37]:


from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(max_iter=800)
log_reg.fit(X_train, Y_train.ravel())


# Decision Tree Classifier

# In[38]:


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X_train, Y_train.ravel())


# KNN Classifier

# In[39]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train.ravel())


# Random Forest Classifier

# In[40]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=15, max_depth=3)
rf.fit(X_train, Y_train.ravel())


# # Testing the Models

# Logistic Regression

# In[41]:


Y_pred_log_train = log_reg.predict(X_train)
Y_pred_log_test = log_reg.predict(X_test)


# In[42]:


pd.DataFrame(Y_pred_log_train).value_counts()


# In[43]:


pd.DataFrame(Y_pred_log_test).value_counts()


# Decision Tree Classifier

# In[44]:


Y_pred_clf_train = clf.predict(X_train)


# In[45]:


Y_pred_clf_test = clf.predict(X_test)


# In[46]:


pd.DataFrame(Y_pred_clf_train).value_counts()


# In[47]:


pd.DataFrame(Y_pred_clf_test).value_counts()


# KNN Classifier

# In[48]:


Y_pred_knn_train = knn.predict(X_train)
Y_pred_knn_test = knn.predict(X_test)


# In[49]:


pd.DataFrame(Y_pred_knn_train).value_counts()


# In[50]:


pd.DataFrame(Y_pred_knn_test).value_counts()


# Random Forest Classifier

# In[51]:


Y_pred_rf_train = rf.predict(X_train)
Y_pred_rf_test = rf.predict(X_test)


# In[52]:


pd.DataFrame(Y_pred_rf_train).value_counts()


# In[53]:


pd.DataFrame(Y_pred_rf_test).value_counts()


# # Evaluating the ML Models using Metrics

# Logistic Regression
# 
# Classification Report
# 

# In[54]:


print(classification_report(Y_test, Y_pred_log_test))


# Accuracy, Precision, Recall, F1 Score

# In[57]:


acc_log = accuracy_score(Y_test, Y_pred_log_test)
prec_log, rec_log, f1_log, sup_log = precision_recall_fscore_support(Y_test, Y_pred_log_test)
print('Accuracy Score =', acc_log)
print('Precision =', prec_log[0])
print('Recall =', rec_log[0])
print('F1 Score =', f1_log[0])


# In[58]:


log_train_acc = accuracy_score(Y_train, Y_pred_log_train)
log_test_acc = accuracy_score(Y_test, Y_pred_log_test)
print('Training Accuracy =', log_train_acc)
print('Testing Accuracy =', log_test_acc)


# Confusion Matrix

# In[59]:


pd.crosstab(Y_test.ravel(), Y_pred_log_test)


# Decision Tree Classifier
# 
# Classification Report
# 

# In[60]:


print(classification_report(Y_test, Y_pred_clf_test))


# Accuracy, Precision, Recall, F1 Score

# In[61]:


acc_clf = accuracy_score(Y_test, Y_pred_clf_test)
prec_clf, rec_clf, f1_clf, sup_clf = precision_recall_fscore_support(Y_test, Y_pred_clf_test)
print('Accuracy Score =', acc_clf)
print('Precision =', prec_clf[0])
print('Recall =', rec_clf[0])
print('F1 Score =', f1_clf[0])


# Checking for Overfitting and Underfitting

# In[62]:


clf_train_acc = accuracy_score(Y_train, Y_pred_clf_train)
clf_test_acc = accuracy_score(Y_test, Y_pred_clf_test)
print('Training Accuracy =', clf_train_acc)
print('Testing Accuracy =', clf_test_acc)


# There is significant variation in the training and testing accuracy. The training accuracy is greater when compared to the testing accuracy. Therefore, the Decision Tree Classifier model is overfit.

# Confusion Matrix

# In[63]:


pd.crosstab(Y_test.ravel(), Y_pred_clf_test)


# Classification Report

# In[64]:


print(classification_report(Y_test, Y_pred_knn_test))


# Accuracy, Precision, Recall, F1 Score

# In[65]:


acc_knn = accuracy_score(Y_test, Y_pred_knn_test)
prec_knn, rec_knn, f1_knn, sup_knn = precision_recall_fscore_support(Y_test, Y_pred_knn_test)
print('Accuracy Score =', acc_knn)
print('Precision =', prec_knn[0])
print('Recall =', rec_knn[0])
print('F1 Score =', f1_knn[0])


# Checking for Overfitting and Underfitting

# In[66]:


knn_train_acc = accuracy_score(Y_train, Y_pred_knn_train)
knn_test_acc = accuracy_score(Y_test, Y_pred_knn_test)
print('Training Accuracy =', knn_train_acc)
print('Testing Accuracy =', knn_test_acc)


# There is no big variation in the training and testing accuracy. Therefore, the KNN Classifier model is not overfit or underfit

# Confusion Matrix

# In[67]:


pd.crosstab(Y_test.ravel(), Y_pred_knn_test)


# Random Forest Classifier
# 
# Classification Report
# 

# In[69]:


print(classification_report(Y_test, Y_pred_rf_test))


# Accuracy, Precision, Recall, F1 Score

# In[70]:


acc_rf = accuracy_score(Y_test, Y_pred_rf_test)
prec_rf, rec_rf, f1_rf, sup_rf = precision_recall_fscore_support(Y_test, Y_pred_rf_test)
print('Accuracy Score =', acc_rf)
print('Precision =', prec_rf[0])
print('Recall =', rec_rf[0])
print('F1 Score =', f1_rf[0])


# Checking for Overfitting and Underfitting

# In[71]:


rf_train_acc = accuracy_score(Y_train, Y_pred_rf_train)
rf_test_acc = accuracy_score(Y_test, Y_pred_rf_test)
print('Training Accuracy =', rf_train_acc)
print('Testing Accuracy =', rf_test_acc)


# There is no big variation in the training and testing accuracy. Therefore, the Random Forest Classifier model is not overfit or underfit.

# Confusion Matrix

# In[72]:


pd.crosstab(Y_test.ravel(), Y_pred_rf_test)


# On comparing the four models built, based on the performance metrics it is clear that Logistic Regression Model gives the highest   Hence, that model is chosen for deployment

# # Dumping the Chosen Model into pkl file
# 

# In[73]:


joblib.dump(log_reg, 'model.pkl')


# In[ ]:




