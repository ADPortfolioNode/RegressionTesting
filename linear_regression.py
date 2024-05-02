import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("FuelConsumption.csv")
df.head()
df.describe()

print ('file: loaded' )
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)

viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
viz.hist()
plt.show()
plt.savefig('plotFeatures.png')

plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()
plt.savefig('plotfuelconsumptions.png')

plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()
plt.savefig('plotenginesize.png')

plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("Cylenders")
plt.ylabel("Emission")
plt.show()
plt.savefig('plotcylindersemission.png')

print ('MODEL: DEFINING TRAINING AND TESTING DATA 80% TESTING 20% TRAINING')
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()
plt.savefig('plotenginesizeEmission.png')

regr = linear_model.LinearRegression()
# Convert to numeric
train_x = pd.to_numeric(train['FUELCONSUMPTION_COMB'], errors='coerce')
train_y = pd.to_numeric(train['CO2EMISSIONS'], errors='coerce')

print ('MODEL: LOADING')
print ('MODEL :loaded to numberic')
# Drop any rows with NaN values
train_x = train_x.dropna()
train_y = train_y.dropna()

# Reshape the data to fit into the model
train_x = np.array(train_x).reshape(-1, 1)
train_y = np.array(train_y).reshape(-1, 1)

print ('MODEL: TRAINING')
# Now you can fit the data into the model
regr.fit(train_x, train_y)

print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")

from sklearn.metrics import r2_score

print ('RESULTS: TESTING')
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)

print ('RESULTS: DEFINED')
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y , test_y_))

print (f'exercise: MAE: %.2f^ % np.mean(np.absolute(test_y_ - test_y))')
#exercise
train_x = np.asanyarray(train[['FUELCONSUMPTION_COMB']])
regr.fit(train_x, train_y)
predictions = regr.predict(test_x)

print("Mean Absolute Error: %.2f" % np.mean(np.absolute(predictions - test_y)))

print("~~~~~~~~~~End of Lined~~~~~~~~~~")