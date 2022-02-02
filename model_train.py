import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import model
df=pd.read_csv('train_ctrUa4K.csv')

df['Gender']=df['Gender'].map({'Male':0,'Female':1})
df['Married']=df['Married'].map({'No':0,'Yes':1})
df['Loan_Status']=df['Loan_Status'].map({'N':0,'Y':1})

df=df.dropna()

x=df[['Gender','Married','ApplicantIncome','LoanAmount','Credit_History']]
y=df['Loan_Status']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=10)

model=RandomForestClassifier(max_depth=4,random_state=10)
model.fit(x_train,y_train)

pred=model.predict(x_test)

print(accuracy_score(y_test,pred))

pickle_out=open('model.pkl',mode='wb')
pickle.dump(model,pickle_out)
pickle_out.close()
print(df.head())

