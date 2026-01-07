import pandas as pd
import numpy as np

credit_card_data = pd.read_csv("DATA_SETS/credit_data.csv")

numbers = credit_card_data["Class"].value_counts()          # => 0 represents legit transcation and 1 represent fraud transaction.

legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

#   UNDER-SAMPLING => METHOD TO HANDLE IMBALANCE DATA

# BUILDING A SAMPLE DATA-SETS CONTAINING SIMILAR DISTRIBUTION OF LEGIT & FRAUD TRANSCATION :
# no. of fraud transcation is 492
legit_sample = legit.sample(n=492)

#concat two arr : 
new_sample = pd.concat([legit_sample, fraud], axis = 0)
print(new_sample)