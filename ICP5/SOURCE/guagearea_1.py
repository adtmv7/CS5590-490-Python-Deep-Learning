import pandas as pd
import matplotlib.pyplot as plt

#Read data from dataset
train = pd.read_csv('./train.csv')

#Display the scatter plot of GarageArea and SalePrice
plt.scatter(train.GarageArea, train.SalePrice, color='red')
plt.xlabel('GarageArea')
plt.ylabel('SalePrice')
plt.show()

#Delete the outlier value of GarageArea
outlier_drop = train[(train.GarageArea <1000) & (train.GarageArea >200)]

##Display the scatter plot of GarageArea and SalePrice after filtering
plt.scatter(outlier_drop.GarageArea, outlier_drop.SalePrice, color='blue')
plt.xlabel('GarageArea')
plt.ylabel('SalePrice')
plt.show()

print(train.SalePrice.describe())