#!/usr/bin/env python
# coding: utf-8

# # Breast Cancer Detection

# ### Importing Libraries

# In[1]:


import pandas as pd


# ### Importing Dataset

# In[2]:


dataset=pd.read_csv("breast_cancer.csv")
X=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values


# ### Splitting the Training Test and Test Set

# In[3]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)


# In[4]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# ### Training the Decision Tree Classifier model on the Training Set

# In[5]:


from sklearn.tree import DecisionTreeClassifier
BC_Detection=DecisionTreeClassifier(criterion="entropy",random_state=0)
BC_Detection.fit(X_train,y_train)


# ### Predicting the Test Set results

# In[6]:


predicted_results=BC_Detection.predict(X_test)
print(predicted_results)


# ### Generating the Confusion Matrix and Checking Accuracy of Model

# In[7]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm=confusion_matrix(y_test,predicted_results)
print("Confusion Matrix :")
print(cm)
accuracy_score(y_test,predicted_results)


# Our Model Has an Accuracy of 95.91%. With 84 out of 3 correct predictions of "Class 2" cancer and 47 out of 50 correct predictions of "Class 4" cancer.

# ### Accuracy of our Model with k-Fold Cross Validation

# In[8]:


from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=BC_Detection,X=X_test,y=y_test,cv=10)
print("Our model has an accuracy of {:.2f}% with a Standard Deviation of {:.2f}%".format(accuracies.mean()*100,accuracies.std()*100))


# ### Breast Cancer Detection

# In[ ]:


ch='y'
while ch=='y':
    ch=str(input("Do you want to Detect ? (Y/N)"))
    ch=ch.lower()
    if ch=='y':
        print("Enter Patient's features to detect Cancer:")
        ct=int(input("Clump Thickness :"))
        ucsize=int(input("Uniformity of Cell Size :"))
        ucshape=int(input("Uniformity of Cell Shape :"))
        ma=int(input("Marginal Adhesion :"))
        secs=int(input("Single Epithelial Cell Size :"))
        bn=int(input("Bare Nuclei :"))
        bc=int(input("Bland Chromatin :"))
        nn=int(input("Normal Nucleoli :"))
        m=int(input("Mitoses :"))

        pred=BC_Detection.predict([[ct,ucsize,ucshape,ma,secs,bn,bc,nn,m]])
        print("Class {}".format(str(pred[0])))


        print("Our model has an accuracy of {:.2f}% with a Standard Deviation of {:.2f}%".format(accuracies.mean()*100,accuracies.std()*100))
    else:
        break


# In[ ]:




