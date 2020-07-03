#====================================DATA FORMATTING & PACKAGES=================================

#=========================importing needed packages and source files=========================
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
#%matplotlib inline #needed in Jupyter labs

#======================================About the dataset=====================================
# This dataset is about past loans. The Loan_train.csv data set includes details of 346 customers whose loan are already paid off or defaulted. It includes following fields:

# Loan_status	        Whether a loan is paid off on in collection
# Principal	            Basic principal loan amount at the
# Terms	                Origination terms which can be weekly (7 days), biweekly, and monthly payoff schedule
# Effective_date	    When the loan got originated and took effects
# Due_date	            Since it’s one-time payoff schedule, each loan has one single due date
# Age	                Age of applicant
# Education	            Education of applicant
# Gender	            The gender of applicant

#==========================loading the data into a dataframe==============================
df = pd.read_csv('./loan_train.csv')
datashow = df.head()
datashape = df.shape
print(datashow)
print(datashape)

#===========================Converting all dates to date time object================================
df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
df.head()

#=======================