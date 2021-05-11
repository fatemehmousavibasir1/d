#!/usr/bin/env python
# coding: utf-8

# In[27]:


#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import math
h_data=pd.read_csv("HeartDataset.csv")
h_data.dropna()
sex=pd.get_dummies(h_data['sex'], drop_first=True)
h_data=pd.concat([h_data, sex,], axis=1)
h_data.drop("sex", axis=1, inplace=True)

diabetes=pd.get_dummies(h_data['fasting blood sugar'], drop_first=True)
h_data=pd.concat([h_data, diabetes,], axis=1)
h_data.drop("fasting blood sugar", axis=1, inplace=True)
increased=pd.get_dummies(h_data['exercise induced angina'], drop_first=True)
h_data=pd.concat([h_data, increased,], axis=1)
h_data.drop("exercise induced angina", axis=1, inplace=True)
angio=pd.get_dummies(h_data['thal'], drop_first=False)
h_data=pd.concat([h_data, angio,], axis=1)
h_data.drop("thal", axis=1, inplace=True)
h_data.drop("normal", axis=1, inplace=True)

wave=pd.get_dummies(h_data['resting electrocardiographic results'], drop_first=False)
h_data=pd.concat([h_data, wave,], axis=1)
h_data.drop("resting electrocardiographic results", axis=1, inplace=True)
chest=pd.get_dummies(h_data['chest pain type'], drop_first=False)
h_data=pd.concat([h_data, chest,], axis=1)
h_data.drop("chest pain type", axis=1, inplace=True)

slope=pd.get_dummies(h_data['slope of the highest point of ST wave'], drop_first=False)
h_data=pd.concat([h_data, slope,], axis=1)
h_data.drop("slope of the highest point of ST wave", axis=1, inplace=True)

X=h_data[['reversible defect']]

y=h_data['Diagnosis']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression(C=1e8)
logmodel.fit(X_train, y_train)
#print intercept and coefficients
print("intercept:", logmodel.intercept_)
print("coefficent:", logmodel.coef_)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
predictions=logmodel.predict(X_test)
confusion_matrix(y_test, predictions)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)

import statsmodels.api as sm
logit_model=sm.Logit(y, sm.add_constant(X))
result=logit_model.fit()
print(result.summary())
print(result.summary2())


import sklearn.metrics as metrics
# calculate the fpr and tpr for all thresholds of the classification
probs = logmodel.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
# method I: plt
import matplotlib.pyplot as plt
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:




