# AI-model-
Spam emails detection model 


Datasets : emails.csv                

Trained model : trained.ipynb 


#**Python code for model**


# python Libraries
import numpy as np

import pandas as pd

import nltk

import nltk.corpus

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import LabelEncoder



# Data Collection and Pre-processing

raw_main_df = pd.read_csv("/content/drive/MyDrive/DataSets/Data/mail_data.csv")

raw_main_df.isnull().sum()

raw_main_df.head()

raw_main_df.shape

print(raw_main_df.value_counts('Category'));

label_encoder = LabelEncoder();

labels = label_encoder.fit_transform(raw_main_df.Category)

raw_main_df['Category'] = labels

raw_main_df.value_counts('Category')


# One more ways to do level encoding : 1. mail_data.loc[mail_data['Category']=='spam','Category',]=0;

X = raw_main_df['Message']

Y = raw_main_df['Category']

print(X)

print(Y)

# Spliting the data into training and test dataset

x_train,x_test,y_train,y_test = train_test_split(X,Y, test_size=0.2,random_state=3)

print(x_train.shape)

print(x_test.shape)

# Features Extraction

feature_extraction = TfidfVectorizer(min_df=1,stop_words='english',lowercase=True) 

print(feature_extraction)

x_train_features = feature_extraction.fit_transform(x_train)

x_test_features = feature_extraction.transform(x_test)

y_train = y_train.astype('int')

y_test = y_test.astype('int')


print(x_train_features)

print(x_test_features)


# Training the Model

# Logistic Regression Model

model = LogisticRegression()

# Evaluating the training model

predicition_on_training_data = model.predict(x_train_features)

accurracy_on_training = accuracy_score(y_train,predicition_on_training_data)

print(accurracy_on_training)

predicition_on_test_data = model.predict(x_test_features)

accuracy_on_test = accuracy_score(y_test,predicition_on_test_data)

print(accuracy_on_test)

# Builiding a predictive Data

input_email = ["As per your request 'Melle Melle (Oru Minnaminunginte Nurungu Vettam)' has been set as your callertune for all Callers. Press *9 to copy your friends Callertune"]

input_data_features = feature_extraction.transform(input_email)

#making prediction

prediction = model.predict(input_data_features)

if prediction==1:

  print("Spam");
	
else :

  print("Ham")








