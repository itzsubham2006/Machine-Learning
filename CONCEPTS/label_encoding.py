# IT IS THE PROCESS OF CONVERTING THE TEXT VALUES TO THE NUMERICAL VALUES TO TRAIN THE MODEL. 


# IMPORTING THE MODULES :
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv("DATA_SETS/breast_cancer_data.csv")
print(df)
# load the label encoder function:
no_of_serousness = df["diagnosis"].value_counts()

# calling LabelEncoder function
label_encode = LabelEncoder()
labels = label_encode.fit_transform(df.diagnosis)

# Appending the labels into the data_frame
df["target"] = labels
t = df["target"].value_counts
print(df.head())

