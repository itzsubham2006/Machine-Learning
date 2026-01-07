# IMPORTING THE MODULES :
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import sklearn.datasets
from sklearn.model_selection import train_test_split

# loading the data_sets
dataset = sklearn.datasets.load_breast_cancer()
print(dataset)

# using the pandas dataframe:
# df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
# df.head()
# x=df
# y=dataset.target
# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=3)

# # Analysing the standard deviation:
# # print(dataset.data.std())

# scalar = StandardScaler()
# scalar.fit(x_train)
# x_train_standardizes=scalar.transform(x_train)
# # print(x_train_standardizes.std())
