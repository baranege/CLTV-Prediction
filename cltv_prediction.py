#BGNBD(Beta Geometric Negative Binomial Distribution) &
# GG(Gamma Gamma) CLTV Estimation

##Libraries and Functions

#!pip install lifetimes
#pip install sqlalchemy
pip install mysql-connector-python

from sqlalchemy import create_engine
import datetime as dt
import pandas as pd
import mysql.connector
#import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
#from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler



pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


#read dataset
df_ = pd.read_excel("datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()
df.head()
df.describe().T

###SQL CONNECTION(SONRA SIL!!!)
creds = {'user': 'synan_dsmlbc_group_8_admin',
         'passwd': 'iamthedatascientist*****!',
         'host': 'db.github.rocks',
         'port': 3306,
         'db': 'synan_dsmlbc_group_8'}

connstr = 'mysql+mysqlconnector://{user}:{passwd}@{host}:{port}/{db}'
conn = create_engine(connstr.format(**creds))

#data processing
df.isnull().sum()
df[df["Country"] == "United Kingdom"].isnull().sum()
df[~df["Invoice"].str.contains("C", na = False)].isnull().sum()
df = df[df["Country"] == "United Kingdom"]
df = df[~df["Invoice"].str.contains("C", na = False)]
df.dropna(inplace = True)
df = df[df["Quantity"] > 0]

# Task1: 6 months CLTV prediction for customers in UK
replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")
df.describe().T

df["TotalPrice"] = df["Quantity"] * df["Price"]
today_date = dt.datetime(2011, 12, 11)


cltv_df = df.groupby('Customer ID').agg({'InvoiceDate': [lambda date: (date.max() - date.min()).days,
                                                         lambda date: (today_date - date.min()).days],
                                         'Invoice': lambda num: num.nunique(),
                                         'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

cltv_df.columns = cltv_df.columns.droplevel(0)

#rename variables
cltv_df.columns = ["recency", "T", "frequency", "monetary"]

#defining monetary as average return per sale
cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]

#selecting the positive sales
cltv_df = cltv_df[cltv_df["monetary"] > 0]
cltv_df.head()

#converting recency and T(tenure) to weekly values for BGNBD
cltv_df["recency"] = cltv_df["recency"] / 7
cltv_df["T"] = cltv_df["T"] / 7

#frequency must be greater than 1
cltv_df = cltv_df[(cltv_df["frequency"] > 1)]

## Modelling BG-NBD Framework

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])

## Modelling Gamma-Gamma Framework

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary'])


## CLTV Prediction with BG-NBD and GG models for 6 months

cltv6 = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=1,
                                   freq="W",
                                   discount_rate=0.01)

cltv6.head()

cltv6.shape
cltv6 = cltv6.reset_index()
cltv6.sort_values(by="clv", ascending=False).head(50)
cltv6_final = cltv_df.merge(cltv, on="Customer ID", how="left")
cltv6_final.sort_values(by="clv", ascending=False).head(10)


# Task2: 1 month and 12 months CLTV prediction for the years 2010-2011

# Prediction for one month

cltv1 = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=1,
                                   freq="W",
                                   discount_rate=0.01)

cltv1.head()

cltv1.shape
cltv1 = cltv1.reset_index()
cltv1.sort_values(by="clv", ascending=False).head(50)
cltv1_final = cltv_df.merge(cltv1, on="Customer ID", how="left")
# Top 10
cltv1_final.sort_values(by="clv", ascending=False).head(10)

# Prediction for 12 months

cltv12 = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=12,
                                   freq="W",
                                   discount_rate=0.01)

cltv12.head()

cltv12.shape
cltv12 = cltv12.reset_index()
cltv12.sort_values(by="clv", ascending=False).head(50)
cltv12_final = cltv_df.merge(cltv12, on="Customer ID", how="left")
# Top 10
cltv12_final.sort_values(by="clv", ascending=False).head(10)

# Task 3: Segmentation and Recommendation

# Segmentation of all customers according to 6 months CLTV prediction

cltv6 = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=6,
                                   freq="W",
                                   discount_rate=0.01)

cltv6_final = cltv_df.merge(cltv6, on = "Customer ID", how = "left")
cltv6_final.head()

# Standardization

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(cltv6_final[["clv"]])
cltv6_final["scaled_clv"] = scaler.transform(cltv6_final[["clv"]])

# Segmentation
cltv6_final["cltv_segment"] = pd.qcut(cltv6_final["clv"], 4, labels=["D", "C", "B", "A"])
cltv6_final.head()

# Task 4: Sending work to SQL database
cltv6_final.to_sql(name='baran_ege', con=conn, if_exists='replace', index=False)
