# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 02:22:19 2022

@author: User
"""

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.offline import plot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


df = pd.read_csv(r"C:/Users/User/OneDrive/Desktop/datasetler/kc_house_data.csv")
df.columns
df.dtypes
df.info()
df.describe()


del df['id']
del df['date']
del df['sqft_above']
del df['sqft_basement']
del df['lat']
del df['long']
del df['sqft_lot15']


df["price"] = df["price"].astype("int64")

df["bathrooms"] = df["bathrooms"].astype("int64")


data = df['price'].max()
data1= df['price'].min()

"condition ve grade sutunlari uzre evlerin ortalama qiymeti
pt1 = pd.pivot_table(df,index = ['condition',],
                     columns='grade',
                     values='price',aggfunc='mean')

"zipcode sutunu uzre evlerin qiymeti"
crs1 = pd.crosstab(index = [df['zipcode']], columns= df['price'])

"sqft_livingi 2000-den kicik olan,view 3, waterfrontu 1 olan evlerin sayi"
c1 = df[(df['sqft_living']<=2000) & (df['view'] ==3) & (df['waterfront'] ==1)]

"2005 den sonra tikilmis 2 mertebeli grade-i 10,view 4 olan evlerin qiymeti" 

c3 = df[(df['yr_built']>=2005) & (df['floors'] ==2) & (df['grade'] ==10) & (df['view'] ==4)]

"sqft_living ve condition sutunlari uzre evlerin ortalama qiymeti"
pt3 = pd.pivot_table(df,index = ['view',],
                     columns='waterfront',
                     values='sqft_living',aggfunc='mean')

"bathroomu 2-den kicik,bedroomu 1den kicik grade 3-den kicik, view 1den kicik olan evler"
c4 = df[(df['bathrooms']<=2) & (df['bedrooms'] <=1) & (df['grade'] <=3) & (df['view'] <=1)]

"conditionu 5, grade 12, view 4 olan evler"
c5 = df[(df['condition']==5) & (df['grade']==12) & (df['view'] ==4)]

'zipcode ve grade sutunlari sahili baxan ve baxmayan evler'
crs2 = pd.crosstab(index = [df['zipcode'], df['grade']],
                   columns= df['waterfront'])



x = df.drop('price', axis = 1)

y = df[['price']]



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)

sc = StandardScaler()
x_train = pd.DataFrame(sc.fit_transform(x_train))
x_test = pd.DataFrame(sc.transform(x_test))

lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

result = pd.DataFrame()
result['y_test'] = y_test['price']
result['y_pred'] = y_pred

mean_squared_error = ((y_test-y_pred)**2).mean()

lr_coef = pd.DataFrame()
lr_coef['Feature'] = x.columns
lr_c = np.transpose(lr.coef_)
lr_coef['Coef_Val'] = lr_c


fig = px.scatter_mapbox(df,lat="grade",lon="price",hover_name="condition",hover_data=["waterfront"],
                        color_discrete_sequence=["fuchsia"],zoom=3,height=300)
plot(fig)

                       
                









