# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 22:40:33 2019

@author: mkrishna
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

np.random.seed(123)

### Import the training data
data = pd.read_csv('train.csv')
re_data = data

### Data cleaning: Drop the incomplete features
data = data.drop(["Id","GarageYrBlt","Alley","PoolQC","Fence","MiscFeature","MasVnrType","MasVnrArea","FireplaceQu",
                  "BsmtQual","BsmtCond","BsmtFinType1","BsmtExposure","BsmtFinType2","GarageType", "GarageFinish",
                  "GarageQual","GarageCond","LotFrontage","Neighborhood"],axis=1)

cleanup_nums1 = {"MSZoning": {"RL": 5, "RM": 4, "FV": 3, "RH": 2, "C (all)": 1},
                "Street": {"Pave": 1, "Grvl": 0},
                "LotShape": {"Reg": 4, "IR1": 3, "IR2": 2, "IR3": 1},
                "LandContour": {"Lvl": 4, "Bnk": 3, "HLS": 2, "Low": 1},
                "Utilities": {"AllPub": 2, "NoSeWa": 1},
                "LotConfig": {"Inside": 5, "Corner": 4, "CulDSac": 3, "FR2": 2, "FR3":1},
                "LandSlope": {"Gtl": 3, "Mod": 2, "Sev": 1},
                "Condition1": {"Norm": 9, "Feedr": 8, "Artery": 7, "RRAn": 6, "PosN": 5, "RRAe": 4, "PosA": 3, "RRNn": 2, 
                               "RRNe": 1},
                "Condition2": {"Norm": 9, "Feedr": 8, "Artery": 7, "RRAn": 6, "PosN": 5, "RRAe": 4, "PosA": 3, "RRNn": 2,
                               "RRNe": 1},
                "BldgType": {"1Fam": 5, "TwnhsE": 4, "Duplex": 3, "Twnhs": 2, "2fmCon": 1},
                "HouseStyle": {"1Story": 8, "2Story": 7, "1.5Fin": 6, "SLvl": 5, "SFoyer": 4, "1.5Unf": 3, "2.5Unf": 2, 
                               "2.5Fin": 1},
                "RoofStyle": {"Gable": 6, "Hip": 5, "Flat": 4, "Gambrel": 3, "Mansard": 2, "Shed": 1},
                "RoofMatl": {"CompShg": 8, "Tar&Grv": 7, "WdShngl": 6, "WdShake": 5, "Metal": 4, "Roll": 3, "Membran": 2,
                             "ClyTile": 1}} 
data.replace(cleanup_nums1, inplace=True)

cleanup_nums2 = {"Exterior1st": {"VinylSd": 17, "HdBoard": 16, "MetalSd": 15, "Wd Sdng": 14, "Plywood": 13, "CemntBd": 12,
                                 "BrkFace": 11, "WdShing": 10, "Stucco": 9, "AsbShng": 8, "Stone": 7, "BrkComm": 6, "ImStucc": 5,
                                 "AsphShn": 4, "CBlock": 3, "PreCast": 2, "Other": 1}, 
                 "Exterior2nd": {"VinylSd": 17, "HdBoard": 16, "MetalSd": 15, "Wd Sdng": 14, "Plywood": 13, "CmentBd": 12,
                                 "BrkFace": 11, "Wd Shng": 10, "Stucco": 9, "AsbShng": 8, "Stone": 7, "Brk Cmn": 6, "ImStucc": 5,
                                 "AsphShn": 4, "CBlock": 3, "PreCast": 2, "Other": 1},
                 "ExterQual": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0},
                 "ExterCond": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0},
                 "Foundation": {"PConc": 6, "CBlock": 5, "BrkTil": 4, "Slab": 3, "Stone": 2, "Wood": 1},
                 "PoolQC": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0},
                 "HeatingQC": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0},
                 "Heating": {"GasA": 6,"GasW": 5,"Grav": 4, "Wall": 3, "OthW": 2, "Floor": 1},
                 "CentralAir": {"Y": 1, "N": 0},
                 "Electrical": {"SBrkr": 5, "FuseA": 4, "FuseF": 3, "FuseP": 2, "Mix": 1},
                 "KitchenQual": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0},
                 "Functional": {"Typ": 7, "Min2": 6, "Min1": 5, "Mod": 4,"Maj1": 3, "Maj2": 2, "Sev": 1},
                 "PavedDrive": {"Y": 3, "N": 2, "P": 1},
                 "SaleType": {"WD": 9, "New": 8,"COD": 7, "ConLD": 6, "ConLI": 5, "ConLw": 4, "CWD": 3, "Oth": 2, "Con": 1},
                 "SaleCondition": {"Normal": 6, "Partial": 5, "Abnorml": 4, "Family": 3,"Alloca": 2, "AdjLand": 1}}

data.replace(cleanup_nums2, inplace=True)
data = data.fillna({"Electrical" : 5})

### From the correlation function, drop out the dependent variables which do not affect the model
corr = data.corr()
fig,ax=plt.subplots(figsize=(30,30))
sns.heatmap(corr,ax=ax,annot=True,linewidths=0.05,fmt='.2f',cmap="magma")
plt.show()

columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= 0.95:
            if columns[j]:
                columns[j] = False

selected_columns = data.columns[columns]
selected_columns = selected_columns.drop("SalePrice")
data = data[selected_columns]
result = pd.DataFrame()
result['SalePrice'] = re_data.iloc[:,-1]

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectPercentile

rfr=RandomForestRegressor(n_estimators=100,random_state=42)

X_train,X_test,y_train,y_test=train_test_split(data.values,result.values,test_size=0.20,random_state=42)

### Using Selectpercentile, now we drop more features which are also irrelavant
select = SelectPercentile(percentile=100)
select.fit(X_train, y_train)
X_train_selected = select.transform(X_train)

rfr.fit(X_train_selected,y_train)
X_test_selected = select.transform(X_test)
y_pred = rfr.predict(X_test_selected)
from sklearn.metrics import r2_score
print("r_square score:",r2_score(y_test,y_pred))

data_test = pd.read_csv('test.csv')
data_test_ = data_test

data_test = data_test.drop(["Id","GarageYrBlt","Alley","PoolQC","Fence","MiscFeature","MasVnrType","MasVnrArea","FireplaceQu",
                  "BsmtQual","BsmtCond","BsmtFinType1","BsmtExposure","BsmtFinType2","GarageType", "GarageFinish",
                  "GarageQual","GarageCond","LotFrontage","Neighborhood"],axis=1)

cleanup_nums3 = {"MSZoning": {"RL": 5, "RM": 4, "FV": 3, "RH": 2, "C (all)": 1},
                "Street": {"Pave": 1, "Grvl": 0},
                "LotShape": {"Reg": 4, "IR1": 3, "IR2": 2, "IR3": 1},
                "LandContour": {"Lvl": 4, "Bnk": 3, "HLS": 2, "Low": 1},
                "Utilities": {"AllPub": 2, "NoSeWa": 1},
                "LotConfig": {"Inside": 5, "Corner": 4, "CulDSac": 3, "FR2": 2, "FR3":1},
                "LandSlope": {"Gtl": 3, "Mod": 2, "Sev": 1},
                "Condition1": {"Norm": 9, "Feedr": 8, "Artery": 7, "RRAn": 6, "PosN": 5, "RRAe": 4, "PosA": 3, "RRNn": 2, 
                               "RRNe": 1},
                "Condition2": {"Norm": 9, "Feedr": 8, "Artery": 7, "RRAn": 6, "PosN": 5, "RRAe": 4, "PosA": 3, "RRNn": 2,
                               "RRNe": 1},
                "BldgType": {"1Fam": 5, "TwnhsE": 4, "Duplex": 3, "Twnhs": 2, "2fmCon": 1},
                "HouseStyle": {"1Story": 8, "2Story": 7, "1.5Fin": 6, "SLvl": 5, "SFoyer": 4, "1.5Unf": 3, "2.5Unf": 2, 
                               "2.5Fin": 1},
                "RoofStyle": {"Gable": 6, "Hip": 5, "Flat": 4, "Gambrel": 3, "Mansard": 2, "Shed": 1},
                "RoofMatl": {"CompShg": 8, "Tar&Grv": 7, "WdShngl": 6, "WdShake": 5, "Metal": 4, "Roll": 3, "Membran": 2,
                             "ClyTile": 1}} 
data_test.replace(cleanup_nums3, inplace=True)

cleanup_nums4 = {"Exterior1st": {"VinylSd": 17, "HdBoard": 16, "MetalSd": 15, "Wd Sdng": 14, "Plywood": 13, "CemntBd": 12,
                                 "BrkFace": 11, "WdShing": 10, "Stucco": 9, "AsbShng": 8, "Stone": 7, "BrkComm": 6, "ImStucc": 5,
                                 "AsphShn": 4, "CBlock": 3, "PreCast": 2, "Other": 1}, 
                 "Exterior2nd": {"VinylSd": 17, "HdBoard": 16, "MetalSd": 15, "Wd Sdng": 14, "Plywood": 13, "CmentBd": 12,
                                 "BrkFace": 11, "Wd Shng": 10, "Stucco": 9, "AsbShng": 8, "Stone": 7, "Brk Cmn": 6, "ImStucc": 5,
                                 "AsphShn": 4, "CBlock": 3, "PreCast": 2, "Other": 1},
                 "ExterQual": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0},
                 "ExterCond": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0},
                 "Foundation": {"PConc": 6, "CBlock": 5, "BrkTil": 4, "Slab": 3, "Stone": 2, "Wood": 1},
                 "PoolQC": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0},
                 "HeatingQC": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0},
                 "Heating": {"GasA": 6,"GasW": 5,"Grav": 4, "Wall": 3, "OthW": 2, "Floor": 1},
                 "CentralAir": {"Y": 1, "N": 0},
                 "Electrical": {"SBrkr": 5, "FuseA": 4, "FuseF": 3, "FuseP": 2, "Mix": 1},
                 "KitchenQual": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0},
                 "Functional": {"Typ": 7, "Min2": 6, "Min1": 5, "Mod": 4,"Maj1": 3, "Maj2": 2, "Sev": 1},
                 "PavedDrive": {"Y": 3, "N": 2, "P": 1},
                 "SaleType": {"WD": 9, "New": 8,"COD": 7, "ConLD": 6, "ConLI": 5, "ConLw": 4, "CWD": 3, "Oth": 2, "Con": 1},
                 "SaleCondition": {"Normal": 6, "Partial": 5, "Abnorml": 4, "Family": 3,"Alloca": 2, "AdjLand": 1}}

data_test.replace(cleanup_nums2, inplace=True)

data_test = data_test.fillna({"MSZoning" : 5})
data_test = data_test.fillna({"Utilities" : 2})
data_test = data_test.fillna({"BsmtFullBath" : 0})
data_test = data_test.fillna({"BsmtHalfBath" : 0})
data_test = data_test.fillna({"KitchenQual" : 3})
data_test = data_test.fillna({"Functional" : 7})
data_test = data_test.fillna({"GarageArea" : 0})
data_test = data_test.fillna({"SaleType" : 9})
data_test = data_test.fillna({"TotalBsmtSF" : 0})
data_test = data_test.fillna({"BsmtUnfSF" : 0})
data_test = data_test.fillna({"BsmtFinSF2" : 0})
data_test = data_test.fillna({"BsmtFinSF1" : 0})
data_test = data_test.fillna({"Exterior2nd" : 17})
data_test = data_test.fillna({"Exterior1st" : 17})
data_test = data_test.fillna({"GarageCars" : 2})

data_test = data_test[selected_columns]
X_test_ = select.transform(data_test.values)
y_pred_ = rfr.predict(X_test_)

submission = pd.DataFrame({"Id": data_test_["Id"],"SalePrice": y_pred_})
submission.to_csv('submission.csv', index=False)
