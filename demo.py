import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder,StandardScaler
import pickle

df = pd.read_csv('train.csv',index_col=0)
df['Arrival Delay in Minutes']= df['Arrival Delay in Minutes'].mean()

Gender_encoder = LabelEncoder()
Customer_Type_encoder = LabelEncoder()
Type_of_Travel_encoder = LabelEncoder()
Class_encoder = LabelEncoder()

df['Gender'] = Gender_encoder.fit_transform(df['Gender'])
df['Customer Type'] = Customer_Type_encoder.fit_transform(df['Customer Type'])
df['Type of Travel'] = Type_of_Travel_encoder.fit_transform(df['Type of Travel'])
df['Class'] = Class_encoder.fit_transform(df['Class'])

X=df.iloc[:,[1,2,3,4,5]].values
y=df['satisfaction']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.75, test_size = 0.25, random_state = 100)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 100)
classifier.fit(X_train,y_train)
y_prediction = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
Lg_accuracy=accuracy_score(y_test,y_prediction)
print("Accuracy for Logistic Regression :",Lg_accuracy)

#classification report
print('\n,',classification_report(y_test,y_prediction))

inputt=[float(x) for x in "1 0 1 1 0".split(' ')]
final=[np.array(inputt)]

b = classifier.predict(final)
print(b)

pickle.dump(classifier,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))