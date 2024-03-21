import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

df = pd.read_csv("FuelConsumption.csv")

# take a look at the dataset
df.head()

#choose features to explore more
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)
print('MODEL:','ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS loaded'
      )

print('~~ SIMPLE LINEAR REGRESSION'
      )
#plot emission values with respect to engine size:
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Co2 Emission")
plt.title("MODEL:engine size => emission values")
plt.show()


#Creating train and test dataset
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

print('TRAIN AND TEST DATA SET DEFINED AS msk <0.8 '
      )

print('TRAIN/TEST: datasets created'
      )

#Train data distribution
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Co2 Emission")
plt.title('TRAINING:enginesize : Co2 Emissions')
plt.show()


print('~~~~~~~MULTIPLE REGRESSION MODEL'
      )
#Multiple Regression Model
from sklearn import linear_model
regr = linear_model.LinearRegression()
print('Linear Model loaded')
x = train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']]
y = train[['CO2EMISSIONS']]

print('Training loaded for Enginesize, cylindes, fuel consumption combination')
regr.fit (x, y)
print('MODEL:=>', 'INDEPENDANTS (X):ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB'
      'DEPENDANT(y): Co2Emissions')

# The coefficients
print ('Coefficients: ', regr.coef_)

#Prediction
y_hat= regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
x = test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']]
y = test[['CO2EMISSIONS']]
print("Mean Squared Error (MSE) : %.2f"
      % np.mean((y_hat - y) ** 2))
# Plotting the actual vs predicted values
plt.scatter(y, y_hat, color='red')
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.title("Actual vs Predicted CO2 Emissions")
plt.show()

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(x, y))

exit()
