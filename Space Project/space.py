import numpy as np # linear algebra
import pandas as pd 

sample_submission = pd.read_csv("sample_submission.csv")
test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")

all_id = test.iloc[:,0:1]

x = train.iloc[:,1:12]
y = train.iloc[:,-1]





homeplanet_dummies =pd.get_dummies(x[["HomePlanet"]],prefix='HomePlanet')
x = pd.concat([x,homeplanet_dummies],axis=1)
cryosleep_dummies = pd.get_dummies(x[["CryoSleep"]],prefix='CryoSleep')
x =pd.concat([x,cryosleep_dummies],axis=1)
desination_dummies = pd.get_dummies(x[["Destination"]],prefix='Destination')
x =pd.concat([x,desination_dummies],axis=1)
vip_dummies = pd.get_dummies(x[["VIP"]],prefix='VIP')
x =pd.concat([x,vip_dummies],axis=1)

x = x.iloc[:,4:]
x = x.drop(columns="VIP")



x["Age"] = x["Age"].replace(np.nan,x["Age"].mean())
x["RoomService"] = x["RoomService"].replace(np.nan,x["RoomService"].value_counts().index[0])
x["FoodCourt"] = x["FoodCourt"].replace(np.nan,x["FoodCourt"].value_counts().index[0])
x["ShoppingMall"] = x["ShoppingMall"].replace(np.nan,x["ShoppingMall"].value_counts().index[0])
x["Spa"] = x["Spa"].replace(np.nan,x["Spa"].value_counts().index[0])
x["VRDeck"] = x["VRDeck"].replace(np.nan,x["VRDeck"].value_counts().index[0])


X = x.values


x_test = test.iloc[:,1:12]

homeplanet_dummies =pd.get_dummies(x_test[["HomePlanet"]],prefix='HomePlanet')
x_test = pd.concat([x_test,homeplanet_dummies],axis=1)
cryosleep_dummies = pd.get_dummies(x_test[["CryoSleep"]],prefix='CryoSleep')
x_test =pd.concat([x_test,cryosleep_dummies],axis=1)
desination_dummies = pd.get_dummies(x_test[["Destination"]],prefix='Destination')
x_test =pd.concat([x_test,desination_dummies],axis=1)
vip_dummies = pd.get_dummies(x_test[["VIP"]],prefix='VIP')
x_test =pd.concat([x_test,vip_dummies],axis=1)

x_test = x_test.iloc[:,4:]
x_test = x_test.drop(columns=["VIP"])

x_test["Age"] = x_test["Age"].replace(np.nan,x_test["Age"].mean())
x_test["RoomService"] = x_test["RoomService"].replace(np.nan,x_test["RoomService"].value_counts().index[0])
x_test["FoodCourt"] = x_test["FoodCourt"].replace(np.nan,x_test["FoodCourt"].value_counts().index[0])
x_test["ShoppingMall"] = x_test["ShoppingMall"].replace(np.nan,x_test["ShoppingMall"].value_counts().index[0])
x_test["Spa"] = x_test["Spa"].replace(np.nan,x_test["Spa"].value_counts().index[0])
x_test["VRDeck"] = x_test["VRDeck"].replace(np.nan,x_test["VRDeck"].value_counts().index[0])

X_test = x_test.values



from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = 'entropy')

dtc.fit(X,y)
y_pred = dtc.predict(X_test)
print(y_pred)





df = pd.DataFrame(y_pred)

df = pd.concat([all_id,df],axis=1)

df = df.rename(columns={0:"Transported"})

df.set_index("PassengerId",inplace=True)
df.to_csv("Submission_DTC.csv")


