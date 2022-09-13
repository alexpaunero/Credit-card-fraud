import seaborn
import pandas as pd
import numpy as np
import codecademylib3
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the data
transactions = pd.read_csv('transactions.csv')

# Summary statistics on amount column
transactions['amount'].describe()

# Create isPayment field
transactions['isPayment'] = np.where((transactions['type'] == 'PAYMENT') | (transactions['type'] == 'DEBIT'), 1, 0)

# Create isMovement field
transactions['isMovement'] = np.where((transactions['type'] == 'TRANSFER') | (transactions['type'] == 'DEBIT'), 1, 0)

# Create accountDiff field
transactions['accountDiff'] = transactions['oldbalanceOrg'] - transactions['oldbalanceDest']

# Create features and label variables
features = transactions[['amount', 'isPayment', 'isMovement', 'accountDiff']]
label = transactions['isFraud']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(features, label, train_size = 0.3)

# Normalize the features variables
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Fit the model to the training data
cc_lr = LogisticRegression()
cc_lr.fit(X_train,y_train)

# Score the model on the training data
print(cc_lr.score(X_train,y_train))

# Score the model on the test data
print(cc_lr.score(X_test,y_test))

# Print the model coefficients
print(cc_lr.coef_)

# New transaction data
transaction1 = np.array([123456.78, 0.0, 1.0, 54670.1])
transaction2 = np.array([98765.43, 1.0, 0.0, 8524.75])
transaction3 = np.array([543678.31, 1.0, 0.0, 510025.5])

# Create a new transaction
your_transaction = np.array([246679.31, 0.0, 1.0, 1510025.5])

# Combine new transactions into a single array
sample_transactions = np.stack((transaction1, transaction2, transaction3, your_transaction))

# Normalize the new transactions
sample_transactions = scaler.transform(sample_transactions)

# Predict fraud on the new transactions
print(cc_lr.predict(sample_transactions))

# Show probabilities on the new transactions
print(cc_lr.predict_proba(sample_transactions))
