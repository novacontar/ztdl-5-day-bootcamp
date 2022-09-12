#!/usr/bin/env python
# coding: utf-8

# # Improving performance

# In[25]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[57]:


RANDOM_STATE = 100


# In[26]:


# Load the data
df = pd.read_csv('../data/new_titanic_features.csv')


# In[45]:


# Create Features and Labels

lst_features = ['Male', 'Family',
        'Pclass2_one', 'Pclass2_two', 'Pclass2_three',
        'Embarked_C', 'Embarked_Q', 'Embarked_S',
        'Age2', 'Fare3_Fare11to50', 'Fare3_Fare51+', 'Fare3_Fare<=10']

X = df[lst_features]
y = df['Survived']


# In[46]:


X.describe()


# In[29]:


from sklearn.model_selection import train_test_split


# In[30]:


# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.2, random_state=RANDOM_STATE)


# In[31]:


from sklearn.linear_model import LogisticRegression


# In[32]:


model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)
pred_train = model.predict(X_train)
pred_test = model.predict(X_test)


# In[47]:


from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
)


# In[34]:


print('Train Accuracy: {:0.3}'.format(accuracy_score(y_train, pred_train)))
print('Test Accuracy: {:0.3}'.format(accuracy_score(y_test, pred_test)))


# In[48]:


confusion_matrix(y_test, pred_test)


# In[49]:


confusion_matrix(pred_test, y_test)


# In[36]:


print(classification_report(y_test, pred_test))


# ## Feature importances (wrong! see exercise 1)

# In[50]:


coeffs = pd.Series(
    model.coef_.ravel(), 
    index=X.columns,
)
coeffs


# In[55]:


coeffs.plot(kind='barh')


# ## Cross Validation

# In[56]:


from sklearn.model_selection import cross_val_score, ShuffleSplit


# In[58]:


cv = ShuffleSplit(n_splits=5, test_size=.4, random_state=RANDOM_STATE)
scores = cross_val_score(model, X, y, cv=cv)
scores


# In[59]:


'Crossval score: %0.3f +/- %0.3f ' % (scores.mean(), scores.std())


# ## Learning curve

# In[60]:


from sklearn.model_selection import learning_curve


# In[61]:


tsz = np.linspace(0.1, 1, 10)
train_sizes, train_scores, test_scores = learning_curve(model, X, y, train_sizes=tsz, cv=3)


# In[62]:


fig = plt.figure()
plt.plot(train_sizes, train_scores.mean(axis=1), 'ro-', label="Train Scores")
plt.plot(train_sizes, test_scores.mean(axis=1), 'go-', label="Test Scores")
plt.title('Learning Curve: Logistic Regression')
plt.ylim((0.5, 1.0))
plt.legend()
plt.draw()
plt.show()


# ### Exercise 1
# 
# Try rescaling the Age feature with [`preprocessing.StandardScaler`](http://scikit-learn.org/stable/modules/preprocessing.html) so that it will have comparable size to the other features.
# 
# - Do the model prediction change?
# - Does the performance of the model change?
# - Do the feature importances change?
# - How can you explain what you've observed?

# In[ ]:





# ### Exercise 2
# 
# Experiment with another classifier for example `DecisionTreeClassifier`, `RandomForestClassifier`,  `SVC`, `MLPClassifier`, `SGDClassifier` or any other classifier of choice you can find here: http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html. 
# 
# - Train the model on both the scaled data and on the unscaled data
# - Compare the score for the scaled and unscaled data
# - how can you get the features importances for tree based models? Check [here](http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html) for some help.
# - Which classifiers are impacted by the age rescale? Why?

# In[ ]:





# ### Exercise 3
# 
# Pick your preferred classifier from Exercise 2 and search for the best hyperparameters. You can read about hyperparameter search [here](http://scikit-learn.org/stable/modules/grid_search.html)
# 
# - Decide the range of hyperparameters you intend to explore
# - Try using [`GridSearchCV`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV) to perform brute force search
# - Try using [`RandomizedSearchCV`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV) for a random search
# - Once you've chosen the best classifier and the best hyperparameter set, redo the learning curve.
# Do you need more data or a better model?

# In[ ]:




