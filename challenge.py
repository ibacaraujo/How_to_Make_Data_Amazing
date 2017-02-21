import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv('Speed Dating Data.csv')
#print data.head(10)
#print data.shape
#print data.loc[0]
#print len(data[data['match'] == 1])
#print len(data[data['match'] == 0])
#print data.isnull().sum()
data = data.select_dtypes(include=['int64', 'float'])
data = data.fillna(0)
#print data.isnull().sum()
scaler = StandardScaler()
mlp = MLPClassifier()

X = data.drop(['match'], axis=1, inplace=False)
y = data['match']

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

mlp.fit(X_train, y_train)
predictions = mlp.predict(X_test)

print accuracy_score(y_test, predictions)