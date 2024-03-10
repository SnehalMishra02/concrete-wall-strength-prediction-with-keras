#importing important modules (pandas and numpy)
import pandas as pd
import numpy as np

#creating a dataframe of the given data using pd, and observing the data
concrete_data = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv')
concrete_data.head()

#Noting size of dataset, performing sanity check and cleanliness of data check
concrete_data.shape
concrete_data.describe()
concrete_data.isnull().sum()

#splitting cols into predictors and target
predictors = concrete_data[concrete_data.columns[concrete_data.columns != 'Strength']] # all columns except Strength
target = concrete_data['Strength'] # Strength column

#sanity check
predictors.head()
target.head()

#number of ip cols
ip = predictors.shape[1]


#splitting data into train and test sets using sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
def split():
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test= train_test_split(predictors,target,random_state=104,test_size=0.30,shuffle=True)



#creating keras model
from keras.models import Sequential
from keras.layers import Dense
def regr_model():
    # create model
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=(ip,)))
    model.add(Dense(1))
    
    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

#fitting model
def train():
    global model
    model = regr_model()
    # fit the model
    model.fit(X_train, y_train, validation_split=0.3, epochs=50, verbose=2)

#prediction
def test():
    y_hat=model.predict(X_test)
    return mean_squared_error(y_test,y_hat)

l_mse=[]

#performing process 50 times
for i in range(50):
    print(10*"-"+str(i+1)+10*'-')
    split()
    regr_model()
    train()
    mse = test()
    l_mse.append(mse)

# Mean and standard deviation of 50 mse's
mean = np.mean(l_mse)
sd = np.std(l_mse)

print("Mean: {}".format(mean,".2f"))
print("Standard Deviation: {}".format(sd,".2f"))
