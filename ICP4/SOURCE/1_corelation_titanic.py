import pandas as pd

#Reading the dataset and dropping null values
train_df= pd.read_csv('train.csv')
print(train_df.columns.values)
print(train_df['Survived'].value_counts(dropna='False'))

#Remove Survived from the train set
X_train= train_df.drop("Survived",axis=1)
Y_train= train_df["Survived"]

#Training on sex column and cateogarizing sex
train_df['Sex'] = train_df['Sex'].map({'female': 1, 'male': 0}).astype(int)

#calculating correlation between survived number and sex of passengers
print(train_df['Survived'].corr(train_df['Sex']))

#corelation between sex and survived
relation = train_df[["Survived","Sex"]].groupby(['Sex'], as_index = False).mean()
print (relation)

