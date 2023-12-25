#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[36]:


import numpy as np
import pandas as pd
import os,sys
from scipy import stats


# In[3]:


from scipy import stats
import statsmodels.api as sm


# In[4]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import metrics
from sklearn.metrics import classification_report,recall_score,roc_auc_score,roc_curve,accuracy_score,precision_score,precision_recall_curve,confusion_matrix
from sklearn.preprocessing import LabelEncoder


# In[27]:


pd.set_option("display.max_columns",None)
pd.set_option("display.max_colwidth",200)

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


# In[6]:


leads = pd.read_csv("Leads.csv")


# In[8]:


leads.head()


# In[9]:


leads.shape


# In[10]:


leads.columns


# In[11]:


leads.describe()


# In[12]:


leads.info()


# In[13]:


# Checking the number of missing values in each column
leads.isnull().sum().sort_values(ascending=False)


# In[14]:


# Droping all the columns in which greater than 
for c in leads.columns:
    if leads[c].isnull().sum()>3000:
        leads.drop(c, axis=1,inplace=True)


# In[15]:


leads.isnull().sum().sort_values(ascending=False)


# In[16]:


#checking value counts of "City" column
leads['City'].value_counts(dropna=False)


# In[17]:


# dropping the "City" feature
leads.drop(['City'], axis = 1, inplace = True)


# In[18]:


#checking value counts of "Country" column
leads['Country'].value_counts(dropna=False)


# In[19]:


# dropping the "Country" feature
leads.drop(['Country'], axis = 1, inplace = True)


# In[20]:


#Now checking the percentage of missing values in each column

round(100*(leads.isnull().sum()/len(leads.index)), 2)


# In[21]:


# Checking the number of null values again
leads.isnull().sum().sort_values(ascending=False)


# In[31]:


for c in leads:
    print(leads[c].astype('category').value_counts())
    print('___________________________________________________')


# In[32]:


leads['Lead Profile'].astype('category').value_counts()


# In[33]:


leads['How did you hear about X Education'].value_counts()


# In[34]:


leads['Specialization'].value_counts()


# In[40]:


leads.drop(['Lead Profile', 'How did you hear about X Education'], axis = 1, inplace = True)


# In[41]:


from matplotlib import pyplot as plt
import seaborn as sns
sns.pairplot(leads,diag_kind='kde',hue='Converted')
plt.show()


# In[42]:


x_edu = leads[['TotalVisits','Total Time Spent on Website','Page Views Per Visit','Converted']]
sns.pairplot(x_edu,diag_kind='kde',hue='Converted')
plt.show()


# In[43]:


from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer()
transformedx_edu = pd.DataFrame(pt.fit_transform(x_edu))
transformedx_edu.columns = x_edu.columns
transformedx_edu.head()


# In[44]:


sns.pairplot(transformedx_edu,diag_kind='kde',hue='Converted')
plt.show()


# In[45]:


leads.drop(['Do Not Call', 'Search', 'Magazine', 'Newspaper Article', 'X Education Forums', 'Newspaper', 
            'Digital Advertisement', 'Through Recommendations', 'Receive More Updates About Our Courses', 
            'Update me on Supply Chain Content', 'Get updates on DM Content', 
            'I agree to pay the amount through cheque'], axis = 1, inplace = True)


# In[46]:


leads['What matters most to you in choosing a course'].value_counts()


# In[47]:


leads.drop(['What matters most to you in choosing a course'], axis = 1, inplace=True)


# In[48]:


leads.isnull().sum().sort_values(ascending=False)


# In[49]:


# Dropping the null values rows in the column 'What is your current occupation'

leads = leads[~pd.isnull(leads['What is your current occupation'])]


# In[50]:


# Observing Correlation
# figure size
plt.figure(figsize=(10,8))

# heatmap
sns.heatmap(leads.corr(), annot=True,cmap="BrBG", robust=True,linewidth=0.1, vmin=-1 )
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[52]:


# Checking the number of null values again
leads.isnull().sum().sort_values(ascending=False)


# In[53]:


# Dropping the null values rows in the column 'TotalVisits'

leads = leads[~pd.isnull(leads['TotalVisits'])]


# In[54]:


# Checking the number of null values again
leads.isnull().sum().sort_values(ascending=False)


# In[55]:


# Dropping the null values rows in the column 'Lead Source'

leads = leads[~pd.isnull(leads['Lead Source'])]


# In[56]:


# Checking the number of null values again
leads.isnull().sum().sort_values(ascending=False)


# In[57]:


# Drop the null values rows in the column 'Specialization'

leads = leads[~pd.isnull(leads['Specialization'])]


# In[58]:


# Checking the number of null values again
leads.isnull().sum().sort_values(ascending=False)


# In[59]:


print(len(leads.index))
print(len(leads.index)/9240)


# In[60]:


# Let's look at the dataset again

leads.head()


# In[61]:


# Dropping the "Prospect ID" and "Lead Number" 
leads.drop(['Prospect ID', 'Lead Number'], 1, inplace = True)


# In[62]:


leads.head()


# In[63]:


# Checking the columns which are of type 'object'

temp = leads.loc[:, leads.dtypes == 'object']
temp.columns


# In[64]:


# Demo Cell
df = pd.DataFrame({'P': ['p', 'q', 'p']})
df


# In[65]:


pd.get_dummies(df)


# In[66]:


pd.get_dummies(df, prefix=['col1'])


# In[67]:


# Creating dummy variables using the 'get_dummies' command
dummy = pd.get_dummies(leads[['Lead Origin', 'Lead Source', 'Do Not Email', 'Last Activity',
                              'What is your current occupation','A free copy of Mastering The Interview', 
                              'Last Notable Activity']], drop_first=True)


# In[68]:


# Add the results to the master dataframe
leads = pd.concat([leads, dummy], axis=1)


# In[69]:


# Creating dummy variable separately for the variable 'Specialization' since it has the level 'Select' 
# which is useless so we
# drop that level by specifying it explicitly

dummy_spl = pd.get_dummies(leads['Specialization'], prefix = 'Specialization')
dummy_spl = dummy_spl.drop(['Specialization_Select'], 1)
leads = pd.concat([leads, dummy_spl], axis = 1)


# In[70]:


# Dropping the variables for which the dummy variables have been created

leads = leads.drop(['Lead Origin', 'Lead Source', 'Do Not Email', 'Last Activity',
                   'Specialization', 'What is your current occupation',
                   'A free copy of Mastering The Interview', 'Last Notable Activity'], 1)


# In[71]:


leads.head()


# In[72]:


# Importing the `train_test_split` library


# In[73]:


# Put all the feature variables in X

X = leads.drop(['Converted'], 1)
X.head()


# In[74]:


y = leads['Converted']

y.head()


# In[75]:


# Spliting the dataset into 70% train and 30% test

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)


# In[76]:


#lets check the shape
print("X_train Size", X_train.shape)
print("y_train Size", y_train.shape)


# In[78]:


#Scaling


# In[79]:


# Scaling the three numeric features present in the dataset

scaler = MinMaxScaler()

X_train[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']] = scaler.fit_transform(X_train[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']])

X_train.head()


# In[80]:


#Looking at the correlations


# In[81]:


# Looking at the correlation table
plt.figure(figsize = (25,15))
sns.heatmap(leads.corr())
plt.show()


# In[82]:


#Model Building


# In[83]:


# Importing the 'LogisticRegression' and creating a LogisticRegression object
logreg = LogisticRegression()


# In[ ]:





# In[ ]:





# In[ ]:





# In[87]:


# Fit a logistic Regression model on X_train after adding a constant and output the summary

X_train_sm = sm.add_constant(X_train)
logm2 = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# In[88]:


# Importing the 'variance_inflation_factor' library


# In[89]:


# Make a VIF dataframe for all the variables present

vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[90]:


X_train.drop('Lead Source_Reference', axis = 1, inplace = True)


# In[91]:


# Refit the model with the new set of features

logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm1.fit().summary()


# In[92]:


# Make a VIF dataframe for all the variables present

vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[93]:


X_train.drop('Last Notable Activity_Had a Phone Conversation', axis = 1, inplace = True)


# In[94]:


# Refit the model with the new set of features

logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm1.fit().summary()


# In[95]:


X_train.drop('What is your current occupation_Housewife', axis = 1, inplace = True)


# In[96]:


# Refit the model with the new set of features

logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm1.fit().summary()


# In[97]:


X_train.drop('What is your current occupation_Working Professional', axis = 1, inplace = True)


# In[98]:


# Refit the model with the new set of features

logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
res = logm1.fit()
res.summary()


# In[99]:


# Making a VIF dataframe for all the variables present

vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[100]:


# Use 'predict' to predict the probabilities on the train set

y_train_pred = res.predict(sm.add_constant(X_train))
y_train_pred[:10]


# In[101]:


# Reshaping it into an array

y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]


# In[102]:


# Creating a new dataframe containing the actual conversion flag and the probabilities predicted by the model

y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Conversion_Prob':y_train_pred})
y_train_pred_final.head()


# In[103]:


y_train_pred_final['Predicted'] = y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.5 else 0)

# Let's see the head
y_train_pred_final.head()


# In[104]:


confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Predicted )
print(confusion)


# In[105]:


# Let's check the overall accuracy

print(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.Predicted))


# In[106]:


# Let's evaluate the other metrics as well

TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[107]:


# Calculating the 'sensitivity'

TP/(TP+FN)


# In[108]:


# Calculating the 'specificity'

TN/(TN+FP)


# In[109]:


# ROC function

def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[110]:


fpr, tpr, thresholds = metrics.roc_curve(y_train_pred_final.Converted,
                    y_train_pred_final.Conversion_Prob, 
                                         drop_intermediate=False)


# In[111]:


# Importing the 'matplotlib'  to plot the ROC curve`


# In[112]:


# Calling the ROC function

draw_roc(y_train_pred_final.Converted, y_train_pred_final.Conversion_Prob)


# In[113]:


# Let's create columns with different probability cutoffs 

numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()


# In[114]:


# Let's create a dataframe to see the values of accuracy, sensitivity, and specificity at 
# different values of probabiity cutoffs

cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)


# In[115]:


# Let's plot it as well

cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()


# In[116]:


y_train_pred_final['final_predicted'] = y_train_pred_final.Conversion_Prob.map( lambda x: 1 if x > 0.42 else 0)

y_train_pred_final.head()


# In[117]:


# Let's checking the `accuracy` now

metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted)


# In[118]:


# Let's create the confusion matrix once again

confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_predicted )
confusion2


# In[119]:


# Let's evaluate the other metrics as well

TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[120]:


# Calculating the 'Sensitivity'

TP/(TP+FN)


# In[129]:


# Dropping the required columns from X_test as well

X_test.drop(['Lead Source_Reference', 'What is your current occupation_Housewife', 
             'What is your current occupation_Working Professional', 
                     'Last Notable Activity_Had a Phone Conversation'], 1, 
                                inplace = True)


# In[130]:


# Make predictions on the test set and store it in the variable 'y_test_pred'

y_test_pred = res.predict(sm.add_constant(X_test))


# In[131]:


y_test_pred[:10]


# In[132]:


# Converting y_pred to a dataframe

y_pred_1 = pd.DataFrame(y_test_pred)


# In[133]:


# Let's see the head

y_pred_1.head()


# In[134]:


# Converting y_test to dataframe

y_test_df = pd.DataFrame(y_test)


# In[135]:


# Remove index for both dataframes to append them side by side 

y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)


# In[136]:


# Append y_test_df and y_pred_1

y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)


# In[137]:


# Check 'y_pred_final'

y_pred_final.head()


# In[138]:


# Rename the column 

y_pred_final= y_pred_final.rename(columns = {0 : 'Conversion_Prob'})


# In[139]:


# Let's see the head of y_pred_final

y_pred_final.head()


# In[140]:


# Make predictions on the test set using 0.45 as the cutoff

y_pred_final['final_predicted'] = y_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.42 else 0)


# In[141]:


# Check y_pred_final

y_pred_final.head()


# In[142]:


# Let's check the overall accuracy

metrics.accuracy_score(y_pred_final['Converted'], y_pred_final.final_predicted)


# In[143]:


confusion2 = metrics.confusion_matrix(y_pred_final['Converted'], y_pred_final.final_predicted )
confusion2


# In[144]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[145]:


# Calculating the 'sensitivity'
TP / float(TP+FN)


# In[146]:


# Calculating the 'specificity'
TN / float(TN+FP)


# In[147]:


# Making predictions on the test set and store it in the variable 'y_test_pred'

y_test_pred = res.predict(sm.add_constant(X_test))


# In[148]:


y_test_pred[:10]


# In[149]:


# Converting y_pred to a dataframe

y_pred_1 = pd.DataFrame(y_test_pred)


# In[150]:


# Let's see the head

y_pred_1.head()


# In[151]:


# Converting y_test to dataframe

y_test_df = pd.DataFrame(y_test)


# In[152]:


# Removing index for both dataframes to append them side by side 

y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)


# In[153]:


# Append y_test_df and y_pred_1

y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)


# In[154]:


# Checking the 'y_pred_final'

y_pred_final.head()


# In[155]:


# Rename the column 

y_pred_final= y_pred_final.rename(columns = {0 : 'Conversion_Prob'})


# In[156]:


# Let's see the head of y_pred_final

y_pred_final.head()


# In[157]:


# Making predictions on the test set using 0.44 as the cutoff

y_pred_final['final_predicted'] = y_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.44 else 0)


# In[158]:


# Checking y_pred_final

y_pred_final.head()


# In[159]:


# Let's checking the overall accuracy

metrics.accuracy_score(y_pred_final['Converted'], y_pred_final.final_predicted)


# In[160]:


confusion2 = metrics.confusion_matrix(y_pred_final['Converted'], y_pred_final.final_predicted )
confusion2


# In[161]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[162]:


# Calculating the Precision

TP/(TP+FP)


# In[163]:


# Calculating Recall

TP/(TP+FN)


# In[ ]:




