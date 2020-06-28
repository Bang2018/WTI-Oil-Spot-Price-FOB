# -*- coding: utf-8 -*-
"""
Created on Sun May 24 23:14:42 2020
Crude oil is a raw or unrefined petroleum product composed 
of hydrocarbon and other organic materials. Brent Crude is produced near sea 
and West Texas Intermediate (WTI) is produced in landlocked areas 
like Texas, Louisiana, and North Dakota. 
Price of Brent crude is higher than WTI because of logistic cost. 
Most oil is priced using Brent Crude as the benchmark. 
Price of Brent Crude also varies with WTI. 
Many traders, thus, firmly consider WTI as an important pricing benchmark.
Spot Price: It is the price of crude oil in current market.
FOB (Free On Board): FOB price encompasses the price of commodity and cost of loading. It excludes the cost of delivery.
In this tutorial, I have used Seasonal ARIMA to predict 
average monthly price of WTI Crude Oil for June, July, 
August, September, November, and December.
Please check the following output:
    1.log.txt
    2.results.txt
@author: Krishnendu Mukherjee
"""
import numpy as np
import pandas as pd
import os
import glob
import datetime as dt
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

#import statsmodels.api as sm


data_path ="D:/Deep Lrning Code/Oil_Price_Prediction/data/"
file = "Oil Price Monthly.csv"
logfile = "log.txt"
outputfile = "results.txt"

def preproc(dirloc,filename,datetimecol,datetime_colname,nrows):
    fpath = os.path.join(dirloc,filename)
    search=False
    if os.path.exists(fpath):
       str1 = "Loading file "+ str(fpath) + " at " + str(dt.datetime.now())
       print(str1)
       with open(os.path.join(dirloc[:-5],logfile),"a") as f:
           f.write(str1)
           f.write("\n")
       f.close()    
       df=pd.read_csv(fpath)
       print(df.head(nrows))
       print("Number of rows :", df.shape[0]," Number of Columns :", df.shape[1],"\n")
       if(datetimecol):
         if (df.shape[1] == 2):
             print("Univariate Time Series...\n")
         else:
           print("Multivariate Time Series ....\n")
    else:
        print("No such file or directory\n")
        search=True
    if (search):
        print("You have mentioned ", filename,"\n" )
        print("Available files at ",dirloc[:-5])
        for dirs,_,file in os.walk(dirloc[:-5]):
            print(file)
    
    ''' Datetimecol index should be zero'''
    print("Searhing Missing Values..\n")
    count=0
    #col_name = [i for i in df.columns if i != "Year"]
    for i in df.columns:
        if (i != datetime_colname):
            for j in range(0,len(df)-1):
                if (df.loc[j,i]== "NA" or df.loc[j,i] == " "):
                   count=count+1
        print("Column :",i," has :",round(count*100/len(df),2), " % missing values \n") 
        if(count > 0):
            df.dropna()
        else:
            pass
    
    print("Exploratory Data Analysis \n")
    
    for i in df.columns:
        if (df.dtypes[i] == np.float64 or df.dtypes[i] == np.int64):
            print("Column Name:",i)
            print("Mean:",round(np.mean(df[i]),2))
            print("Median:", round(np.median(df[i]),2))
            print("Standard Deviation:",round(np.std(df[i]),2))
            print("Maximum:",np.max(df[i]))
            print("Minimum:",np.min(df[i]))
            plt.figure()
            df[i].hist(figsize=(5,5))
            plt.title("Histogram:"+ str(i))
            plt.show()
            print("Outliers Detection...")
            plt.figure()
            plt.boxplot(df[i])
            plt.title("Outliers:"+str(i))
            plt.show()
            
    return df

def plot_seasonal_decomp(df,colname):
    df[colname]=pd.to_datetime(df[colname])
    plt.figure(figsize=(10,6))
    plt.plot(df[colname],df["WTI Spot Price FOB "],alpha=0.75)
    plt.xlabel("Year")
    plt.ylabel("Barrels per Dollar")
    plt.title("WTI Crude Oil FOB")
    plt.show()
    
    print("Loading required librarires for ARIMA ....\n")
    from statsmodels.tsa.seasonal import seasonal_decompose
    df= df.set_index(colname)
    #df=df["WTI Spot Price FOB "].resamples("MS").mean()
    seas = seasonal_decompose(df,model="additive")
    plt.figure()
    seas.trend.plot(title="Trend|Method:Additive",figsize=(10,4))
    plt.figure()
    seas.seasonal.plot(title="Seasonal|Method:Additive",figsize=(10,4))  
    plt.figure()
    seas.resid.plot(title="residual|Method:Additive",figsize=(10,4))
    
    seasmt = seasonal_decompose(df,model="multiplicative")
    plt.figure()
    seasmt.trend.plot(title="Trend|Method:Multiplicative",figsize=(10,4))
    plt.figure()
    seasmt.seasonal.plot(title="Seasonal|Method:Multiplicative",figsize=(10,4))  
    plt.figure()
    seasmt.resid.plot(title="residual|Method:Multiplicative",figsize=(10,4))
               
    return df

def Auto_Arima(df,dirloc,filename):
    import itertools
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    
    p=d=q=range(0,3)
    pdq = list(itertools.product(p,d,q))
    seas_decomp=[]
    for x in pdq:
        x1=(x[0],x[1],x[2],12)
        seas_decomp.append(x1)
    print("Computating AIC of Different Sesonal ARIMA.....\n")
    arima_order=[]
    seas_order=[]
    aic_val=[]
    
    for params in pdq:
        for seas_par in seas_decomp:
            mod = SARIMAX(df,order=params,seasonal_order=seas_par,enforce_stationarity=False, enforce_invertibility=False,freq="MS").fit()
            arima_order.append(params)
            seas_order.append(seas_par)
            aic_val.append(round(mod.aic,2))
            print("SARIMA: {} X {} | AIC = {}".format(params,seas_par,round(mod.aic,2)))
            
    results = pd.DataFrame({"ARIMA Order":arima_order,"Seasonal Order":seas_order,"AIC Value":aic_val}) 
    results_sorted = results.sort_values(by="AIC Value",ascending=True)
    results_sorted=results_sorted.reset_index(drop=True)
    print("Selected SARIMA Order:",results_sorted.head(2))
    
    final_model = SARIMAX(df,order=results_sorted["ARIMA Order"][0],seasona_order=results_sorted["Seasonal Order"][0],enforce_stationarity=False, enforce_invertibility=False,freq="MS").fit()
    print("Final Model Result Summary {}".format(final_model.summary()))
    print(results_sorted["ARIMA Order"][0])
    print(results_sorted["Seasonal Order"][0])
    predictions = final_model.predict(start=dt.datetime.strptime("2020-06-01","%Y-%m-%d"),end=dt.datetime.strptime("2020-12-01","%Y-%m-%d"))
    print("Average Monthly WTI Crude Oil Spot Price from June to Dec 2020:")
    print(predictions)
    with open(os.path.join(dirloc[:-5],outputfile),"a") as f:
         f.write("Simulation Result of SARIMA....\n")
         f.write(str(results_sorted))
         f.write("\n")
         f.write(str(predictions))
    f.close()
    return results_sorted

def correlogram(df):
    import statsmodels.api as sm
    fig = plt.figure(figsize=(15,6))
    ax1 = fig.add_subplot(211)
    acf = sm.graphics.tsa.plot_acf(df["WTI Spot Price FOB "],lags=100,ax=ax1,alpha=.05,use_vlines=True,fft=True)
    ax2=fig.add_subplot(212)
    pacf=sm.graphics.tsa.plot_pacf(df["WTI Spot Price FOB "],lags=100,ax=ax2,alpha=0.05,use_vlines=True)
    
    
proc_dat = preproc(data_path,file,True,"Date",nrows=10) 
decomp_dat=plot_seasonal_decomp(proc_dat,"Date") 
correlogram(decomp_dat)  
results=Auto_Arima(decomp_dat,data_path,outputfile)    

