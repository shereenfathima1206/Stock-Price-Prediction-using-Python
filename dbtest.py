import pandas as pd
from pandas_datareader import data, wb
import datetime
import re
import sys
import time
import datetime
import requests

import numpy as np
import pandas as pd
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import unicodedata
sid = SentimentIntensityAnalyzer()

from treeinterpreter import treeinterpreter as ti
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report,confusion_matrix

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from datetime import datetime, timedelta

from sklearn.neural_network import MLPClassifier
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from treeinterpreter import treeinterpreter as ti
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

class Database:
    def readhead(self,id):
        print("CSV File Name");
        print(id)
        df1=pd.read_csv(id)
        df1=df1.head(10)
        return df1

    def readtail(self,id):
        print("CSV File Name");
        print(id)
        df1=pd.read_csv(id)
        df1=df1.tail(10)
        return df1

    def readdesc(self,id):
        print("CSV File Name");
        print(id)
        df1=pd.read_csv(id)
        df1=df1.describe()
        return df1

    def readmulti(self,id,id1,id2):
        print("CSV File Name");
        print(id,id1,id2)
        f1 = pd.read_csv(id)
        f2 = pd.read_csv(id1)
        f3 = pd.read_csv(id2)
        f1=f1.head()
        f2=f2.head()
        f3=f3.head()
        stocks = pd.DataFrame({'AAPL': f1["Adj Close"],
                      'AMZN': f2["Adj Close"],
                      'GOOG': f3["Adj Close"]})
 
        stocks.head()        
        return stocks
      
    def split_crumb_store(v):
        return v.split(':')[2].strip('"')
    
    def find_crumb_store(lines):
        for l in lines:
            if re.findall(r'CrumbStore', l):
                return l
        print("Did not find CrumbStore")
    
    def get_cookie_value(r):
        return {'B': r.cookies['B']}
    
    def get_page_data(self):
        url = "https://finance.yahoo.com/quote/%s/?p=%s" % (self, id)
        r = requests.get(url)
        cookie = Database.get_cookie_value(r)

        lines = r.content.decode('unicode-escape').strip(). replace('}', '\n')
        return cookie, lines.split('\n')
    
    def get_cookie_crumb(self, symbol):
        cookie, lines = Database.get_page_data(symbol)
        crumb = Database.split_crumb_store(Database.find_crumb_store(lines))
        return cookie, crumb
    
    def get_data(symbol, start_date, end_date, cookie, crumb):
        filename = '%s.csv' % (symbol)
        url = "https://query1.finance.yahoo.com/v7/finance/download/%s?period1=%s&period2=%s&interval=1d&events=history&crumb=%s" % (symbol, start_date, end_date, crumb)
        response = requests.get(url, cookies=cookie)
        with open (filename, 'wb') as handle:
            for block in response.iter_content(1024):
                handle.write(block)
                
    def get_now_epoch(self):
        # @see https://www.linuxquestions.org/questions/programming-9/python-datetime-to-epoch-4175520007/#post5244109
        return int(time.time())
    
    def download_quotes(self, id):
        start_date = 0 
        print("mmcc100")
        end_date = Database.get_now_epoch(self)
        print("mmcc200")
        cookie, crumb = Database.get_cookie_crumb(self, id)
        Database.get_data(id, start_date, end_date, cookie, crumb)
    




    def PullData(self):
        df_stocks = pd.read_pickle('pickled_ten_year_filtered_data.pkl')
        print(df_stocks.head())
        return df_stocks.head()
    #PullData()

    def AdjCloseData(self):
        df_stocks = pd.read_pickle('pickled_ten_year_filtered_data.pkl')
        df_stocks['prices'] = df_stocks['adj close'].apply(np.int64)
        df_stocks = df_stocks[['prices', 'articles']]
        return df_stocks.tail()
    #AdjCloseData()

    #Prediction
    def NewColData(self):
        df_stocks = pd.read_pickle('pickled_ten_year_filtered_data.pkl')
        df_stocks['prices'] = df_stocks['adj close'].apply(np.int64)
        df_stocks = df_stocks[['prices', 'articles']]
        df = df_stocks[['prices', 'articles']].copy()
        # Adding new columns to the data frame
        df["compound"] = ''
        df["neg"] = ''
        df["neu"] = ''
        df["pos"] = ''

        for date, row in df_stocks.T.iteritems():
            try:
                #sentence = unicodedata.normalize('NFKD', df_stocks.loc[date, 'articles']).encode('ascii','ignore')
                sentence = unicodedata.normalize('NFKD', df_stocks.loc[date, 'articles'])
                ss = sid.polarity_scores(sentence)
                df.set_value(date, 'compound', ss['compound'])
                df.set_value(date, 'neg', ss['neg'])
                df.set_value(date, 'neu', ss['neu'])
                df.set_value(date, 'pos', ss['pos'])
            except TypeError:
                print (df_stocks.loc[date, 'articles'])
                print (date)
        return df.head()
    #NewColData()

    #Random Forest Predictor

    def RandomForestPrediction(self):
        df_stocks = pd.read_pickle('pickled_ten_year_filtered_data.pkl')
        # selecting the prices and articles
        df_stocks['prices'] = df_stocks['adj close'].apply(np.int64)
        df_stocks = df_stocks[['prices', 'articles']]

        df_stocks['articles'] = df_stocks['articles'].map(lambda x: x.lstrip('.-'))

        df = df_stocks[['prices']].copy()

        df["compound"] = ''
        df["neg"] = ''
        df["neu"] = ''
        df["pos"] = ''

        for date, row in df_stocks.T.iteritems():
            try:
                #sentence = unicodedata.normalize('NFKD', df_stocks.loc[date, 'articles']).encode('ascii','ignore')
                sentence = unicodedata.normalize('NFKD', df_stocks.loc[date, 'articles'])
                ss = sid.polarity_scores(sentence)
                df.set_value(date, 'compound', ss['compound'])
                df.set_value(date, 'neg', ss['neg'])
                df.set_value(date, 'neu', ss['neu'])
                df.set_value(date, 'pos', ss['pos'])
            except TypeError:
                print (df_stocks.loc[date, 'articles'])
                print (date)

        train_start_date = '2007-01-01'
        train_end_date = '2014-12-31'
        test_start_date = '2015-01-01'
        test_end_date = '2016-12-31'
        train = df.loc[train_start_date : train_end_date]
        test = df.loc[test_start_date:test_end_date]

        sentiment_score_list = []
        for date, row in train.T.iteritems():
            #sentiment_score = np.asarray([df.loc[date, 'compound'],df.loc[date, 'neg'],df.loc[date, 'neu'],df.loc[date, 'pos']])
            sentiment_score = np.asarray([df.loc[date, 'neg'],df.loc[date, 'pos']])
            sentiment_score_list.append(sentiment_score)
        numpy_df_train = np.asarray(sentiment_score_list)

        sentiment_score_list = []
        for date, row in test.T.iteritems():
            #sentiment_score = np.asarray([df.loc[date, 'compound'],df.loc[date, 'neg'],df.loc[date, 'neu'],df.loc[date, 'pos']])
            sentiment_score = np.asarray([df.loc[date, 'neg'],df.loc[date, 'pos']])
            sentiment_score_list.append(sentiment_score)
        numpy_df_test = np.asarray(sentiment_score_list)
        y_train = pd.DataFrame(train['prices'])
        y_test = pd.DataFrame(test['prices'])

        rf = RandomForestRegressor()
        rf.fit(numpy_df_train, y_train)

        print (rf.feature_importances_)
        prediction, bias, contributions = ti.predict(rf, numpy_df_test)
        idx = pd.date_range(test_start_date, test_end_date)
        predictions_df = pd.DataFrame(data=prediction[0:], index = idx, columns=['prices'])
        predictions_plot = predictions_df.plot()
        fig = y_test.plot(ax = predictions_plot).get_figure()
        fig.savefig("/static/images/RFGraph100.png")

        return predictions_df.tail(10)    #RandomForestPrediction()



    def RandomForestGraph(self):
        df_stocks = pd.read_pickle('pickled_ten_year_filtered_data.pkl')
        # selecting the prices and articles
        df_stocks['prices'] = df_stocks['adj close'].apply(np.int64)
        df_stocks = df_stocks[['prices', 'articles']]

        df_stocks['articles'] = df_stocks['articles'].map(lambda x: x.lstrip('.-'))

        df = df_stocks[['prices']].copy()

        df["compound"] = ''
        df["neg"] = ''
        df["neu"] = ''
        df["pos"] = ''

        for date, row in df_stocks.T.iteritems():
            try:
                #sentence = unicodedata.normalize('NFKD', df_stocks.loc[date, 'articles']).encode('ascii','ignore')
                sentence = unicodedata.normalize('NFKD', df_stocks.loc[date, 'articles'])
                ss = sid.polarity_scores(sentence)
                df.set_value(date, 'compound', ss['compound'])
                df.set_value(date, 'neg', ss['neg'])
                df.set_value(date, 'neu', ss['neu'])
                df.set_value(date, 'pos', ss['pos'])
            except TypeError:
                print (df_stocks.loc[date, 'articles'])
                print (date)

        train_start_date = '2007-01-01'
        train_end_date = '2014-12-31'
        test_start_date = '2015-01-01'
        test_end_date = '2016-12-31'
        train = df.loc[train_start_date : train_end_date]
        test = df.loc[test_start_date:test_end_date]

        sentiment_score_list = []
        for date, row in train.T.iteritems():
            #sentiment_score = np.asarray([df.loc[date, 'compound'],df.loc[date, 'neg'],df.loc[date, 'neu'],df.loc[date, 'pos']])
            sentiment_score = np.asarray([df.loc[date, 'neg'],df.loc[date, 'pos']])
            sentiment_score_list.append(sentiment_score)
        numpy_df_train = np.asarray(sentiment_score_list)

        sentiment_score_list = []
        for date, row in test.T.iteritems():
            #sentiment_score = np.asarray([df.loc[date, 'compound'],df.loc[date, 'neg'],df.loc[date, 'neu'],df.loc[date, 'pos']])
            sentiment_score = np.asarray([df.loc[date, 'neg'],df.loc[date, 'pos']])
            sentiment_score_list.append(sentiment_score)
        numpy_df_test = np.asarray(sentiment_score_list)
        y_train = pd.DataFrame(train['prices'])
        y_test = pd.DataFrame(test['prices'])

        rf = RandomForestRegressor()
        rf.fit(numpy_df_train, y_train)

        #print (rf.feature_importances_)
        prediction, bias, contributions = ti.predict(rf, numpy_df_test)
        #print(prediction)
        idx = pd.date_range(test_start_date, test_end_date)
        predictions_df = pd.DataFrame(data=prediction[0:], index = idx, columns=['prices'])
        #print(predictions_df)
        predictions_plot = predictions_df.plot()
        fig = y_test.plot(ax = predictions_plot).get_figure()
        fig.savefig("randomforest.png")
        


    #RandomForestGraph()


    #LogisticRegression
    def offset_value(test_start_date, test, predictions_df):
        temp_date = test_start_date
        average_last_5_days_test = 0
        average_upcoming_5_days_predicted = 0
        total_days = 10
        for i in range(total_days):
            average_last_5_days_test += test.loc[temp_date, 'prices']
            temp_date = datetime.strptime(temp_date, "%Y-%m-%d").date()
            difference = temp_date + timedelta(days=1)
            temp_date = difference.strftime('%Y-%m-%d')
        average_last_5_days_test = average_last_5_days_test / total_days

        temp_date = test_start_date
        for i in range(total_days):
            average_upcoming_5_days_predicted += predictions_df.loc[temp_date, 'prices']
            temp_date = datetime.strptime(temp_date, "%Y-%m-%d").date()
            difference = temp_date + timedelta(days=1)
            temp_date = difference.strftime('%Y-%m-%d')
        average_upcoming_5_days_predicted = average_upcoming_5_days_predicted / total_days
        difference_test_predicted_prices = average_last_5_days_test - average_upcoming_5_days_predicted
        return difference_test_predicted_prices

    def LogisticRegressionGraph(self):
        df_stocks = pd.read_pickle('pickled_ten_year_filtered_data.pkl')
        # selecting the prices and articles
        df_stocks['prices'] = df_stocks['adj close'].apply(np.int64)
        df_stocks = df_stocks[['prices', 'articles']]

        df_stocks['articles'] = df_stocks['articles'].map(lambda x: x.lstrip('.-'))

        df = df_stocks[['prices']].copy()

        df["compound"] = ''
        df["neg"] = ''
        df["neu"] = ''
        df["pos"] = ''

        for date, row in df_stocks.T.iteritems():
            try:
                #sentence = unicodedata.normalize('NFKD', df_stocks.loc[date, 'articles']).encode('ascii','ignore')
                sentence = unicodedata.normalize('NFKD', df_stocks.loc[date, 'articles'])
                ss = sid.polarity_scores(sentence)
                df.set_value(date, 'compound', ss['compound'])
                df.set_value(date, 'neg', ss['neg'])
                df.set_value(date, 'neu', ss['neu'])
                df.set_value(date, 'pos', ss['pos'])
            except TypeError:
                print (df_stocks.loc[date, 'articles'])
                print (date)


        years = [2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016]
        prediction_list = []
        for year in years:
            # Splitting the training and testing data
            train_start_date = str(year) + '-01-01'
            train_end_date = str(year) + '-10-31'
            test_start_date = str(year) + '-11-01'
            test_end_date = str(year) + '-12-31'
            train = df.loc[train_start_date : train_end_date]
            test = df.loc[test_start_date:test_end_date]

            # Calculating the sentiment score
            sentiment_score_list = []
            for date, row in train.T.iteritems():
                sentiment_score = np.asarray([df.loc[date, 'compound'],df.loc[date, 'neg'],df.loc[date, 'neu'],df.loc[date, 'pos']])
                #sentiment_score = np.asarray([df.loc[date, 'neg'],df.loc[date, 'pos']])
                sentiment_score_list.append(sentiment_score)
            numpy_df_train = np.asarray(sentiment_score_list)
            sentiment_score_list = []
            for date, row in test.T.iteritems():
                sentiment_score = np.asarray([df.loc[date, 'compound'],df.loc[date, 'neg'],df.loc[date, 'neu'],df.loc[date, 'pos']])
                #sentiment_score = np.asarray([df.loc[date, 'neg'],df.loc[date, 'pos']])
                sentiment_score_list.append(sentiment_score)
            numpy_df_test = np.asarray(sentiment_score_list)

            # Generating models
            lr = LogisticRegression()
            lr.fit(numpy_df_train, train['prices'])
            prediction = lr.predict(numpy_df_test)
            prediction_list.append(prediction)
            #print train_start_date + ' ' + train_end_date + ' ' + test_start_date + ' ' + test_end_date
            idx = pd.date_range(test_start_date, test_end_date)
            #print year
            predictions_df_list = pd.DataFrame(data=prediction[0:], index = idx, columns=['prices'])
            difference_test_predicted_prices = offset_value(test_start_date, test, predictions_df_list)
            # Adding offset to all the advpredictions_df price values
            predictions_df_list['prices'] = predictions_df_list['prices'] + difference_test_predicted_prices
            predictions_df_list
            # Smoothing the plot
            predictions_df_list['ewma'] = pd.ewma(predictions_df_list["prices"], span=10, freq="D")
            predictions_df_list['actual_value'] = test['prices']
            predictions_df_list['actual_value_ewma'] = pd.ewma(predictions_df_list["actual_value"], span=10, freq="D")
            # Changing column names
            predictions_df_list.columns = ['predicted_price', 'average_predicted_price', 'actual_price', 'average_actual_price']
            predictions_df_list.plot()
            predictions_df_list_average = predictions_df_list[['average_predicted_price', 'average_actual_price']]
            predictions_df_list_average.plot()
            plt.savefig("LRGraph.png")
            fig.savefig("LRGraph.png")
     
    #LogisticRegressionGraph()
    #MLP Classifier
    def MLPClassifierGraph():

        df_stocks = pd.read_pickle('pickled_ten_year_filtered_data.pkl')
        # selecting the prices and articles
        df_stocks['prices'] = df_stocks['adj close'].apply(np.int64)
        df_stocks = df_stocks[['prices', 'articles']]

        df_stocks['articles'] = df_stocks['articles'].map(lambda x: x.lstrip('.-'))

        df = df_stocks[['prices']].copy()

        df["compound"] = ''
        df["neg"] = ''
        df["neu"] = ''
        df["pos"] = ''


        for date, row in df_stocks.T.iteritems():
            try:
                #sentence = unicodedata.normalize('NFKD', df_stocks.loc[date, 'articles']).encode('ascii','ignore')
                sentence = unicodedata.normalize('NFKD', df_stocks.loc[date, 'articles'])
                ss = sid.polarity_scores(sentence)
                df.set_value(date, 'compound', ss['compound'])
                df.set_value(date, 'neg', ss['neg'])
                df.set_value(date, 'neu', ss['neu'])
                df.set_value(date, 'pos', ss['pos'])
            except TypeError:
                print (df_stocks.loc[date, 'articles'])
                print (date)


        years = [2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016]
        prediction_list = []
        for year in years:
            # Splitting the training and testing data
            train_start_date = str(year) + '-01-01'
            train_end_date = str(year) + '-10-31'
            test_start_date = str(year) + '-11-01'
            test_end_date = str(year) + '-12-31'
            train = df.loc[train_start_date : train_end_date]
            test = df.loc[test_start_date:test_end_date]

            # Calculating the sentiment score
            sentiment_score_list = []
            for date, row in train.T.iteritems():
                sentiment_score = np.asarray([df.loc[date, 'compound'],df.loc[date, 'neg'],df.loc[date, 'neu'],df.loc[date, 'pos']])
                #sentiment_score = np.asarray([df.loc[date, 'neg'],df.loc[date, 'pos']])
                sentiment_score_list.append(sentiment_score)
            numpy_df_train = np.asarray(sentiment_score_list)
            sentiment_score_list = []
            for date, row in test.T.iteritems():
                sentiment_score = np.asarray([df.loc[date, 'compound'],df.loc[date, 'neg'],df.loc[date, 'neu'],df.loc[date, 'pos']])
                #sentiment_score = np.asarray([df.loc[date, 'neg'],df.loc[date, 'pos']])
                sentiment_score_list.append(sentiment_score)
            numpy_df_test = np.asarray(sentiment_score_list)

            # Generating models
            mlpc = MLPClassifier(hidden_layer_sizes=(100, 200, 100), activation='relu', 
                                 solver='lbfgs', alpha=0.005, learning_rate_init = 0.001, shuffle=False) # span = 20 # best 1
            mlpc.fit(numpy_df_train, train['prices'])   
            prediction = mlpc.predict(numpy_df_test)

            prediction_list.append(prediction)
            #print train_start_date + ' ' + train_end_date + ' ' + test_start_date + ' ' + test_end_date
            idx = pd.date_range(test_start_date, test_end_date)
            #print year
            predictions_df_list = pd.DataFrame(data=prediction[0:], index = idx, columns=['prices'])
            difference_test_predicted_prices = offset_value(test_start_date, test, predictions_df_list)
            # Adding offset to all the advpredictions_df price values
            predictions_df_list['prices'] = predictions_df_list['prices'] + difference_test_predicted_prices
            predictions_df_list
            # Smoothing the plot
            predictions_df_list['ewma'] = pd.ewma(predictions_df_list["prices"], span=20, freq="D")
            predictions_df_list['actual_value'] = test['prices']
            predictions_df_list['actual_value_ewma'] = pd.ewma(predictions_df_list["actual_value"], span=20, freq="D")
            # Changing column names
            predictions_df_list.columns = ['predicted_price', 'average_predicted_price', 'actual_price', 'average_actual_price']
            predictions_df_list.plot()
            predictions_df_list_average = predictions_df_list[['average_predicted_price', 'average_actual_price']]
            predictions_df_list_average.plot()
            plt.savefig("MLPGraph.png")


db = Database()
db.RandomForestPrediction()
