__author__ = 'KranthiDhanala'
from math import log,sqrt,pow

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import numpy as np


def main():
    # Make the graphs a bit prettier, and bigger
    pd.set_option('display.mpl_style', 'default')

    # Always display all the columns
    pd.set_option('display.width', 5000)
    pd.set_option('display.max_columns', 60)


    location = "Data"
    #missing values
    na_values = ["M",'-']
    #read weather data
    weather_data = pd.read_csv(location+"\weather.csv", sep=",", parse_dates="date", na_values=na_values)
    #read key_data
    key_data = pd.read_csv(location+"\key.csv", sep=",")
    #read train data
    train_data = pd.read_csv(location+"/train.csv", sep=",", parse_dates="date")

    #train data merged with key

    train_key_merge = pd.merge(train_data, key_data, on="store_nbr", how="inner")
    #test data
    test_data = pd.read_csv(location+"/test.csv",sep =",",parse_dates="date")
    #test data merged with key
    test_key_merge = pd.merge(test_data, key_data, on="store_nbr", how="inner")

    #fill missing snow and preciptotal as 0
    weather_data.ix[weather_data["snowfall"].isnull(),"snowfall"] = 0
    weather_data.ix[weather_data["preciptotal"].isnull(),"preciptotal"] = 0


    #set trace to 0.1 for snowfall and 0.01 for rainfall
    weather_data.ix[weather_data["preciptotal"]== '  T',"preciptotal"] = 0.01
    weather_data.ix[weather_data["snowfall"]== '  T',"snowfall"] = 0.1


    """filling missing values in weather data"""
    for i in range(1,21):
        dewpoint_null = weather_data["dewpoint"].isnull() & weather_data["station_nbr"] == i
        wetbulb_null = weather_data["wetbulb"].isnull() & weather_data["station_nbr"] == i
        tmax_null = weather_data["tmax"].isnull() & weather_data["station_nbr"] == i
        tmin_null = weather_data["tmin"].isnull() & weather_data["station_nbr"] == i
        stnpressure_null = weather_data["stnpressure"].isnull() & weather_data["station_nbr"] == i
        sealevel_null = weather_data["sealevel"].isnull() & weather_data["station_nbr"] == i
        resultspeed_null = weather_data["resultspeed"].isnull() & weather_data["station_nbr"] == i
        resultdir_null = weather_data["resultdir"].isnull() & weather_data["station_nbr"] == i
        avgspeed_null = weather_data["avgspeed"].isnull() & weather_data["station_nbr"] == i

        #calculate column means for each station and fill na with missing values
        dewpoint_mean = weather_data[weather_data["dewpoint"].notnull() & weather_data["station_nbr"] == i]["dewpoint"].mean()
        weather_data.ix[dewpoint_null,"dewpoint"] = dewpoint_mean

        wetbulb_mean = weather_data[weather_data["wetbulb"].notnull() & weather_data["station_nbr"] == i]["wetbulb"].mean()
        weather_data.ix[wetbulb_null,"wetbulb"] = wetbulb_mean

        tmax_mean = weather_data[weather_data["tmax"].notnull() & weather_data["station_nbr"] == i]["tmax"].mean()
        weather_data.ix[tmax_null,"tmax"] = tmax_mean

        tmin_mean = weather_data[weather_data["tmin"].notnull() & weather_data["station_nbr"] == i]["tmin"].mean()
        weather_data.ix[tmin_null,"tmin"] = tmin_mean

        stnpressure_mean = weather_data[weather_data["stnpressure"].notnull() & weather_data["station_nbr"] == i]["stnpressure"].mean()
        weather_data.ix[stnpressure_null,"stnpressure"] = stnpressure_mean

        sealevel_mean = weather_data[weather_data["sealevel"].notnull() & weather_data["station_nbr"] == i]["sealevel"].mean()
        weather_data.ix[sealevel_null,"sealevel"] = sealevel_mean

        resultspeed_mean = weather_data[weather_data["resultspeed"].notnull() & weather_data["station_nbr"] == i]["resultspeed"].mean()
        weather_data.ix[resultspeed_null,"resultspeed"] = resultspeed_mean

        resultdir_mean = weather_data[weather_data["resultdir"].notnull() & weather_data["station_nbr"] == i]["resultdir"].mean()
        weather_data.ix[resultdir_null,"resultdir"] = resultdir_mean

        avgspeed_mean = weather_data[weather_data["avgspeed"].notnull() & weather_data["station_nbr"] == i]["avgspeed"].mean()
        weather_data.ix[avgspeed_null,"avgspeed"] = avgspeed_mean



    #fill missing values for tavg where tmax and tmin are given
    fill_cond = weather_data["tmax"].notnull() & weather_data["tmin"].notnull() & weather_data["tavg"].isnull()
    temp = (weather_data[fill_cond]["tmax"]+weather_data[fill_cond]["tmin"])/2
    temp = temp.round()
    weather_data.ix[fill_cond,"tavg"] = temp.values


    #fill missing values for heat and cool where tavg is present
    def fun_heat(x):
        if x["tavg"] < 65:
            return round(abs(x["tavg"]- 65))
        else:
            return 0

    def fun_cool(x):
        if x["tavg"] < 65:
            return 0
        else:
            return round(abs(x["tavg"]- 65))


    fill_cond = weather_data["tavg"].notnull() & weather_data["heat"].isnull() & weather_data["cool"].isnull()
    temp_heat = weather_data[fill_cond][["tavg","heat"]].apply(fun_heat,axis=1)
    weather_data.ix[fill_cond,"heat"] = temp_heat.values

    temp_cool = weather_data[fill_cond][["tavg","heat"]].apply(fun_cool,axis=1)
    weather_data.ix[fill_cond,"cool"] = temp_cool.values

    #identifying storm
    weather_data["snowfall"] = weather_data["snowfall"].astype(float)
    weather_data["preciptotal"] = weather_data["preciptotal"].astype(float)
    snow_str = "SN|SG"
    weather_data["is_snow_storm"] = (weather_data.codesum.str.contains(snow_str)) & (weather_data["snowfall"].notnull()) & (weather_data["snowfall"] >= 2.0)
    weather_data["is_rain_storm"] = (weather_data.codesum.str.contains("RA")) & (~weather_data.codesum.str.contains("SN")) & (weather_data["preciptotal"].notnull()) & (weather_data["preciptotal"] >= 1.0)
    weather_data["is_storm"] = (weather_data["is_snow_storm"]== True) | (weather_data["is_rain_storm"] == True)
    weather_data["act_storm"] = weather_data["is_storm"]

    weather_data["train"] = weather_data["is_storm"] == False
    weather_data["test"] = False

    # to identify days around storm
    def event_identifier(dat):
        event_list = []
        event_list.append(dat)
        event_list.append(dat+np.timedelta64(1,'D'))
        event_list.append(dat+np.timedelta64(2,'D'))
        event_list.append(dat+np.timedelta64(3,'D'))

        event_list.append(dat-np.timedelta64(1,'D'))
        event_list.append(dat-np.timedelta64(2,'D'))
        event_list.append(dat-np.timedelta64(3,'D'))
        return event_list

    #only the storm dates are marked true, now mark all week days of the storm as storm data
    weather_data["date"] = pd.to_datetime(weather_data["date"],coerce = True)
    for i in range(1,21):
        temp = weather_data[(weather_data["station_nbr"]== i)&(weather_data["is_storm"]==True)]["date"]
        storm_dates = temp.values
        len_dt = len(storm_dates)
        count = 0
        for dat in storm_dates:

            rain_storm = weather_data[(weather_data["date"]==dat) & (weather_data["station_nbr"]==i)]["is_rain_storm"].values
            snow_storm = weather_data[(weather_data["date"]==dat) & (weather_data["station_nbr"]==i)]["is_snow_storm"].values
            #add +3 and -3 dates around storm
            event_dates = event_identifier(dat)

            #get unique dates
            event_dates = list(set(event_dates))
            check = (weather_data["date"].isin(event_dates)) & (weather_data["station_nbr"]==i)
            weather_data.ix[check,"is_storm"] = True
            weather_data.ix[check,"is_rain_storm"] = rain_storm
            weather_data.ix[check,"is_snow_storm"] = snow_storm

            if count < round(len_dt/2.0):
                weather_data.ix[check,"train"] = True
            else:
                weather_data.ix[check,"test"] = True
            count = count + 1




    #convert all features as float to make compatible with scikit learn
    weather_data["snowfall"]= weather_data["snowfall"].astype(float)
    weather_data["preciptotal"]= weather_data["preciptotal"].astype(float)
    weather_data["is_storm"] = weather_data["is_storm"].astype(float)
    weather_data["is_rain_storm"] = weather_data["is_rain_storm"].astype(float)
    weather_data["is_snow_storm"] = weather_data["is_snow_storm"].astype(float)




    #merge train,key data with weather data to form complete data set
    train_key_merge["date"] = pd.to_datetime(train_key_merge["date"],coerce = True)
    total_data = pd.merge(train_key_merge, weather_data, on=["station_nbr", "date"], how="left")


    test_key_merge["date"] = pd.to_datetime(test_key_merge["date"],coerce = True)
    total_test_data =  pd.merge(test_key_merge, weather_data, on=["station_nbr", "date"], how="left")

    print "Modelling Features from data"
    #derive some new features from given data

    #data is available from this date
    init_date = pd.to_datetime(pd.Series("2012-01-01"),format='%Y-%m-%d')
    start_date = init_date[0]

    def calculate_date_diff(x):
        ret = x - start_date
        return (ret.days)/1.0 + 1

    #day
    total_data["day"] = pd.DatetimeIndex(total_data["date"]).day
    #month
    total_data["month"] = pd.DatetimeIndex(total_data["date"]).month
    #year
    total_data["year"] = pd.DatetimeIndex(total_data["date"]).year
    #weekday
    total_data["dayofweek"] = pd.DatetimeIndex(total_data["date"]).dayofweek # 0 -->monday 6--> sunday
    #is weekend
    total_data["weekend"] = 0
    weekend_check = total_data["dayofweek"].isin([5,6])
    total_data.ix[weekend_check,"weekend"] = 1
    #day of year
    total_data["dayofyear"] = pd.DatetimeIndex(total_data["date"]).dayofyear
    #week number
    total_data["week"] = pd.DatetimeIndex(total_data["date"]).week
    #date from baseline
    total_data["diff_data"] = total_data["date"].apply(calculate_date_diff)


    #derive features from test data
    total_test_data["diff_data"] = total_test_data["date"].apply(calculate_date_diff)
    total_test_data["day"] = pd.DatetimeIndex(total_test_data["date"]).day
    total_test_data["month"] = pd.DatetimeIndex(total_test_data["date"]).month
    total_test_data["year"] = pd.DatetimeIndex(total_test_data["date"]).year
    total_test_data["dayofweek"] = pd.DatetimeIndex(total_test_data["date"]).dayofweek # 0 -->monday 6--> sunday
    total_test_data["weekend"] = 0
    weekend_check = total_test_data["dayofweek"].isin([5,6])
    total_test_data.ix[weekend_check,"weekend"] = 1
    total_test_data["dayofyear"] = pd.DatetimeIndex(total_test_data["date"]).dayofyear
    total_test_data["week"] = pd.DatetimeIndex(total_test_data["date"]).week

    total_data["units"] = total_data["units"].astype(float)
    #initially mark all units in test data to zeros
    total_test_data["units"] = 0


    #validation

    def train_test_split(data):
        train_data = data[data["train"] == True]
        test_data = data[data["test"] == True]
        return train_data,test_data

    def rmsle(predicted, actual):
        error = 0.0
        for i in range(len(actual)):
            error += pow(log(actual[i]+1)-log(predicted[i]+1), 2)
        return sqrt(error/len(actual))

    #change it to true for parameter tuning and cross validation
    validation = False

    if validation:

        train_data,test_data = train_test_split(total_data)
        test_data["predicted"] = 0
        for i in range(1,46):

            for j in range(1,112):
                temp = train_data[(train_data["store_nbr"] == i) & (train_data["item_nbr"] == j)].set_index("date") #& (total_data["units"]>0)
                if len(temp[temp["units"] > 0]) > 0:
                    print i,j
                    features = list(total_data.columns.values)
                    features = [x for x in features if x not in ["codesum","month","is_rain_storm","week","is_snow_storm","year","item_nbr","date","depart","sunrise","sunset","units","store_nbr","station_nbr","train","test","predicted","act_storm"]]
                    temp = temp[np.abs(temp.units-temp.units.mean())<=(3*temp.units.std())]
                    X = temp[features].values
                    Y = temp["units"].values

                    scalerX = StandardScaler().fit(X)
                    X = scalerX.transform(X)


                    #predict log of the actual units + 1
                    Y = Y + 1
                    Y = np.log(Y)



                    #rng = np.random.RandomState(1)
                    rf_reg_model =  RandomForestRegressor(n_estimators = 120,random_state = 0, min_samples_split = 9, oob_score = True, n_jobs = -1)


                    rf_reg_model.fit(X,Y)
                    #print zip(rf_reg_model.feature_importances_,features)
                    condition = (test_data["store_nbr"] == i) & (test_data["item_nbr"] == j)
                    temp_test = test_data[condition]

                    predict_input = temp_test[features].values
                    predict_input = scalerX.transform(predict_input)

                    predict_out = rf_reg_model.predict(predict_input)
                    predict_out = np.exp(predict_out)
                    predict_out = predict_out - 1
                    test_data.ix[condition,"predicted"] = predict_out
        print "calculating rmsle.."
        err = rmsle(test_data["predicted"].values,test_data["units"].values)
        print err
        return






    print "Learning from data.."


    for i in range(1,46):

        for j in range(1,112):
            temp = total_data[(total_data["store_nbr"] == i) & (total_data["item_nbr"] == j)].set_index("date") #& (total_data["units"]>0)
            if len(temp[temp["units"] > 0]) > 0:
                print i,j
                #plt.plot(temp.index, temp['units'],'-')
                #plt.show()
                #train the model with days and predict on missing dates
                features = list(total_data.columns.values)

                features = [x for x in features if x not in ["codesum","month","year","item_nbr","date","depart","sunrise","sunset","units","store_nbr","station_nbr","train","test","predicted","act_storm"]]
                temp = temp[np.abs(temp.units-temp.units.mean())<=(3*temp.units.std())]
                #print features
                X = temp[features].values
                scalerX = StandardScaler().fit(X)
                X = scalerX.transform(X)
                #X = np.reshape(X, (-1, 1))
                Y = temp["units"].values
                #predict log of the actual units + 1
                Y = Y + 1
                Y = np.log(Y)
                rf_reg_model = RandomForestRegressor(n_estimators = 100, random_state = 0, min_samples_split = 7, oob_score = False, n_jobs = -1)

                rf_reg_model.fit(X,Y)
                #print zip(rf_reg_model.feature_importances_,features) to check feature importances
                condition = (total_test_data["store_nbr"] == i) & (total_test_data["item_nbr"] == j)
                temp_test = total_test_data[condition]

                predict_input = temp_test[features].values
                predict_input = scalerX.transform(predict_input)

                #predict_input = np.reshape(predict_input, (-1, 1))
                predict_out = rf_reg_model.predict(predict_input)
                predict_out = np.exp(predict_out)
                predict_out = predict_out - 1
                #print predict_out
                total_test_data.ix[condition,"units"] = predict_out


    total_test_data["units"] = total_test_data["units"].apply(lambda x: round(x))
    def create_id(x):
        res = str(x["store_nbr"])+"_"+str(x["item_nbr"])+"_"+str(x["date"])
        return res[:-9] #to remove the time from date time
    total_test_data["id"] = total_test_data.apply(create_id,axis=1)
    total_test_data["units"] = total_test_data["units"].apply(lambda x: 0 if x < 0 else x)
    #total_test_data = total_test_data.set_index("id")
    print "writing to csv"
    csv_data = total_test_data[["id","units"]]
    csv_data = csv_data.set_index("id")
    csv_data.to_csv("final_submission.csv", sep=',')



    return




if __name__ == "__main__":
    main()