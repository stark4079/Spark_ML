#!/usr/bin/env python
# coding: utf-8

# # Linear Support Vector Machine
# ## Dataset: House Prices - Advanced Regression Techniques

# ### Import some necessary libraries

# In[1]:


import findspark
findspark.init("D:\spark\spark-3.1.1-bin-hadoop3.2")
import pyspark
from pyspark.sql.functions import col, when
from pyspark.sql.types import IntegerType
from pyspark.sql import SparkSession
from pyspark.ml.classification import LinearSVC
from pyspark.sql.functions import collect_list
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf
from pyspark.ml.feature import VectorAssembler


spark = SparkSession.builder.getOrCreate()


# - Initialize the spark as the folder that installed spark in local disk, then create a SparkSession to run Spark in Jupiter notebook

# ### Load data

# ##### Introduction about dataset

# - Description: [1] "The Ames Housing dataset was compiled by Dean De Cock for use in data science education. It's an incredible alternative for data scientists looking for a modernized and expanded version of the often cited Boston Housing dataset."
# **Reference**
# - Source: Kaggle Website
# - Author: Dean De Cock
# - Title: "House Prices - Advanced Regression TechniquesPredict sales prices and practice feature engineering, RFs, and gradient boosting"
# - Link: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview
# - Date accessed: 25.4.2021

# In[2]:


rawdata = spark.read.load("train.csv", format="csv", header=True, delimiter=",")


# In[3]:


spark.conf.set('spark.sql.repl.eagerEval.enabled', True)
rawdata


# ### Describe the purpose of using linear SVM algorithm in this dataset.
# - We only consider 3 features for analyzing the house price including:
#     - OverallQual: Rates the overall material and finish of the house.
#         - 10 Very Excellent
#         - 9	Excellent
#         - 8	Very Good
#         - 7	Good
#         - 6	Above Average
#         - 5	Average
#         - 4	Below Average
#         - 3	Fair
#         - 2	Poor
#         - 1	Very Poor
#     - OverallCond: Rates the overall condition of the house, which has the same measuring as OverallQual
#     - TotalBsmtSF: the Type 2 finished square feet. 
# - In this task, we analyze the mentioned feature to predict or classify a house into 2 groups:
#     - Group 1: House price is less than 250K: SalePrice < 1500000
#     - Group 2: House price is higher or equal 250K: SalePrice >= 1500000
# - To do this one, we need to preprocess the SalePrice into binary value, 0 if SalePrice less than 150k, 1 in the remaining case.

# #### Step 1: Extract the data with 3 columns: OverallQual, Sale Price, TotalBsmtSF, OverallCond. Then  convert its datatype from string to integer.

# In[4]:


df = rawdata.select(rawdata.SalePrice, rawdata.OverallQual, rawdata.OverallCond,rawdata.TotalBsmtSF)
df = df.withColumn("SalePrice",col("SalePrice").cast(IntegerType()))     .withColumn("OverallQual",col("OverallQual").cast(IntegerType()))    .withColumn("TotalBsmtSF",col("TotalBsmtSF").cast(IntegerType()))    .withColumn("OverallCond",col("OverallCond").cast(IntegerType()))
df


# - Print the data's Schema to validate the converting step.

# In[5]:


df.printSchema()


# - Convert the values of SalePrice column to binary values as decribed above.

# In[6]:


df = df.withColumn("SalePrice", when(df.SalePrice < 150000, 0).otherwise(1))


# - Collect the OverallQual and OverallCond and TotalBsmtSF into an array with 3 elements as the new column Features.

# In[7]:


assembler = VectorAssembler(inputCols=["OverallQual", "OverallCond", "TotalBsmtSF"], outputCol="Features")
df = assembler.transform(df)
df = df.select(df.SalePrice, df.Features)
df


# - To make the attribute Feature for the model, we must convert the column Features to vector dense datatype.

# In[8]:


to_vector = udf(lambda a: Vectors.dense(a), VectorUDT())
data = df.select("SalePrice", to_vector("Features").alias("Feature"))


# In[9]:


# Verify the converting step 
data.printSchema()


# - Spit the data to 70% data for training and 30% data for testing.

# In[10]:


(train, test) = data.randomSplit([0.7, 0.3])


# #### Step 2: Using Linear support vector machine to build the model

# - Create the trainer of Linear Support Vector Machine by function LinearSVC then pass the arguments the number of training epochs, the label column is SalePrice, the Features column is Feature. Then saved into linearSvm class.

# In[11]:


linearSvm = LinearSVC(maxIter=20, regParam=0.1, labelCol="SalePrice", featuresCol="Feature")


# - Fit the training data into the model by using the fit function of linearSvm. 

# In[12]:


linearModel = linearSvm.fit(train)


# #### Step 3: Evaluation

# - Predict on the test sets

# In[13]:


prediction = linearModel.transform(test)
prediction


# - Calculate the True Positive, True Negative, False Positive, False Negative by group by and filter to count those values from the confusion matrix prediction.

# In[14]:


TN = prediction.filter('prediction = 0 AND SalePrice = prediction').count()
TP = prediction.filter('prediction = 1 AND SalePrice = prediction').count()
FN = prediction.filter('prediction = 0 AND SalePrice <> prediction').count()
FP = prediction.filter('prediction = 1 AND SalePrice <> prediction').count()
print(TP, TN)
print(FP, FN)


# - Using the group by function to group by the SalePrice and prediction, then count it into the count column -> Finally, show confusion matrix.

# In[15]:


prediction.groupBy('SalePrice', 'prediction').count().show()


# **Caclulate the Accuracy, Precision, Recall, F1 Score by the confusion matrix**
# - Accuracy: It measures the exact probability that the dataset are correctly predicted by the model.
# $$\begin{equation}
# Accurancy = \dfrac{TP + TN}{P + N}
# \end{equation}$$
# 
# - Precision: mesure the exactness, which percentage of samples are correctly predicted as positive in reality
# $$\begin{equation}
# Precision= \dfrac{TP}{TP + FP}
# \end{equation}$$
# 
# - Recall: mesure the completeness, which percentage of positive samples are predicted.
# $$\begin{equation}
# Recall= \dfrac{TP}{TP + FN} = \dfrac{TP}{P}
# \end{equation}$$
# 
# - F1 Score: is the harmonic mean of precision and recall.
# $$\begin{equation}
# F1 Score = \dfrac{2 × precision × recall}{precision + recall}
# \end{equation}$$

# In[16]:


# calculate metrics by the confusion matrix
accuracy = (TN + TP) / (TN + TP + FN + FP)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
F1_Score =  2 * (precision * recall) / (precision + recall)


# In[17]:


print("Evaluation model by calculate metrics by the confusion matrix")
print("Accuracy: %.3f" % accuracy)
print("Precision: %.3f" % precision)
print("Recall: %.3f" % recall)
print("F1 Score: %.3f" % F1_Score)

