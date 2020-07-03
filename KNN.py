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
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
#%matplotlib inline #needed in Jupyter labs

#====================================About the dataset=====================================
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

#======================Converting all dates to date time object================================
df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
df.head()


#=================================DATA VISUALIZATION AND PREPROCESSING====================================

#=========================================Class value count======================================
valuecounts = df['loan_status'].value_counts()
print(valuecounts)

#==============================Plot histogram to understand this data better===========================
##########============Using Principal================#########
#setting the space, inital and final value then the samples to generate
bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)
#setting the grids to plot and their titles, colors and how they wrap
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
#setting subsets
g.map(plt.hist, 'Principal', bins=bins, ec="k")
#auto detection of element to be displayed in the axis.
g.axes[-1].legend()
plt.show()

#############================Using Age==================##############
bins = np.linspace(df.age.min(), df.age.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'age', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()


#=============================PRE-PROCESSING: FEATURE SELECTION/EXTRACTION=========================

#====================Lets look at the day of the week people by gender took the loans=====================
df['dayofweek'] = df['effective_date'].dt.dayofweek
bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()

# We see that people who get the loan at the end of the week dont pay it off, so lets use Feature binarization to set a threshold values less then day 4

#====================Setting the loan threshold date limit to day 4 of the week==========================
df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
threshold = df.head()
print(threshold)

#=======================Convert Categorical features to numerical values===================================
#display payment status by gender
gendercount = df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)
print(gendercount)
#convert male to 0 and female to 1
df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
df.head()


#================================ONEHOT ENCODING OF DATA===========================================

#==================================group by education=====================================
#counting and classifying datas by the education and payment status
educount = df.groupby(['education'])['loan_status'].value_counts(normalize=True)
print(educount)
#features before one-hot encoding
featuresb4onehotenc = df[['Principal','terms','age','Gender','education']].head()
print(featuresb4onehotenc)

#======one hot encoding to convert categorical varables to binary variables and append them to feature Data=====
#selecting all features
Feature = df[['Principal','terms','age','Gender','weekend']]
#adding education and encoding it
Feature = pd.concat([Feature, pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
newfeatures = Feature.head()
print(newfeatures)


#=====================================FEATURES SELECTION========================================

#===============================features selection and display==========================
X = Feature
X[0:5]
print(X)

#==================================adding our labels==================================
y = df['loan_status'].values
y[0:5]
print(y)


#======================================NORMALIZING DATA=======================================
X = preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]
print(X)


# Classification
#
# 1. K Nearest Neighbor(KNN)
# 2. Decision Tree
# 3. Support Vector Machine
# 4. Logistic Regression

#======================================CLASSIFICATION========================================

#=================================K NEAREST NEIGHBOR (KNN)==================================

#========================================Train/Test Split=====================================
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

