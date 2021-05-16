import warnings
import matplotlib.pyplot as plt
import np as np
from sklearn.metrics import mean_squared_error
from pandas.tseries.offsets import DateOffset
from statsmodels.tsa.seasonal import seasonal_decompose
from tqdm.contrib import itertools
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
from pandas.plotting import autocorrelation_plot
import statsmodels.api as sm
import matplotlib
from pylab import rcParams
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA


# Settings for matplotlib.
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'
rcParams['figure.figsize'] = (18, 8)




def main(file):
    names = ['BAYERN', 'BADEN-WÃRTTEMBERG', 'BRANDENBURG', 'HAMBURG', 'BREMEN', 'MECKLENBURG-VORPOMMERN', 'HESSEN',
             'NIEDERSACHSEN', 'RHEINLAND-PFALZ', 'NORDRHEIN-WESTFALEN', 'SACHSEN', 'SAARLAND', 'SCHLESWIG-HOLSTEIN',
             'SACHSEN-ANHALT', 'THÃRINGEN'] # Creating an array with all city names of text file
    df = pd.read_csv(file, header='infer', parse_dates=['date']) # Reading the data.txt using pandas read_csv method
    df["confirmedInfections"] = df.groupby('ID')['confirmedInfections'].diff(periods=1).fillna(df.confirmedInfections).astype(int) # Subtract confirmed infections to find daily cases
    names_parser = 0 # Counter to parse correct city name for the methods
    for i in df.ID.unique():
        df_temp = df.groupby('ID').get_group(i)  # Group the dataframe of each ID
        timeseries(df_temp, names[names_parser]) # Calling timeseries function
        names_parser += 1 # Counter to place the index to the correct city name


def timeseries(dataframe, city_name):
    cols = ['ID', 'name']  # Creating columns to be dropped
    dataframe.drop(cols, axis=1, inplace=True)  # Dropping columns that I don't need
    dataframe.columns = ["date", "Covid cases"] # Renaming the columns for better handling
    dataframe.describe()
    dataframe.set_index('date', inplace=True) # Setting and choosing the date column as index
    dataframe.plot(figsize=(15, 6))  # Setting figure size

    # Plotting typical timeseries graph
    plt.title(city_name)  # Giving a title to graph based on city
    plt.ylabel("Covid cases")  # Giving a name to y-axis
    plt.xlabel("Date")  # Giving a name to x-axis
    plt.show()  # Plotting the graph


    #Autocorrelation plot
    autocorrelation_plot(dataframe['Covid cases'])  # Creating an autocorrelation plot of covid cases for each city
    plt.show() # Showing the plot

    # Plotting trend, seasonality and resid
    # Period needs to be in a specific number based on the number of data we have in each frame
    decomposition = sm.tsa.seasonal_decompose(dataframe, model='additive', period=int(len(dataframe) / 2))
    decomposition.plot() # Applying decomposition formula to axes
    plt.show() # Showing the plot


    forecast(dataframe) # Calling the forecast function


def forecast(dataframe):
    # Define the p, d and q parameters to take any value between 0 and 2
    p = d = q = range(0, 2)

    # Generate all different combinations of p, q and q triplets
    pdq = list(itertools.product(p, d, q))

    # Generate all different combinations of seasonal p, q and q triplets
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

    # Summarizing and printing the possible combinations for ARIMA and seasonal_roder
    print('Examples of parameter combinations for Seasonal ARIMA...')
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

    # Specify to ignore warning messages
    warnings.filterwarnings("ignore")
    # Creating a list in order to save all the combinations of what the below for loop calculates
    AIC_list = pd.DataFrame({}, columns=['param', 'param_seasonal', 'AIC'])

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                # Searching for all order and seasonal order combinations
                mod = sm.tsa.statespace.SARIMAX(dataframe['Covid cases'],
                                                order=param, 
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)

                results = mod.fit()
                # Printing each combination that for loop finds
                print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
                # Creating a temp an applying to that (x,y,z)|(a,b,c,d)|AIC:(float number) combination
                temp = pd.DataFrame([[param, param_seasonal, results.aic]], columns=['param', 'param_seasonal', 'AIC'])
                AIC_list = AIC_list.append(temp, ignore_index=True)
                del temp

            except:
                continue

    # Find minimum value in AIC
    m = np.amin(AIC_list['AIC'].values)
    # Find index number for lowest AIC
    l = AIC_list['AIC'].tolist().index(m)
    # Presenting the order and seasonal order parameters based on minimum AIC
    Min_AIC_list = AIC_list.iloc[l, :]

    # Applying the combination of the minimum AIC we found to order and seasonal order
    mod = sm.tsa.statespace.SARIMAX(dataframe['Covid cases'],
                                    order=Min_AIC_list['param'],
                                    seasonal_order=Min_AIC_list['param_seasonal'],
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)

    results = mod.fit()
    print(results.summary().tables[1])
    print("### Min_AIC_list ### \n{}".format(Min_AIC_list))
    # Plotting diagnostics to check if our parameters will be precise
    results.plot_diagnostics(figsize=(15, 12))
    plt.show()

    # Setting the time we decide to start forecasting
    pred = results.get_prediction(start=pd.to_datetime('2020-11-24'), dynamic=False)
    pred_ci = pred.conf_int()
    ax = dataframe['2020':].plot(label='observed')
    # Apply the orange line to see how applies before forecast
    pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)

    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.2)

    ax.set_xlabel('Date')
    ax.set_ylabel('Covid cases')
    plt.legend()
    plt.show()

    # Applying truth and forecasted values to variables
    y_forecasted = pred.predicted_mean
    y_truth = dataframe['2020-11-24':]

    # Calling mse_and_rmse function
    mse_and_rmse(y_truth, y_forecasted)



    pred_dynamic = results.get_prediction(start=pd.to_datetime('2020-11-24'), dynamic=True, full_results=True)
    pred_dynamic_ci = pred_dynamic.conf_int()



    # Extract the predicted and true values of our time series
    y_forecasted = pred_dynamic.predicted_mean
    y_truth = dataframe['2020-11-24':]


    # Get forecast 30 steps (one month) ahead in future
    pred_uc = results.get_forecast(steps=30)

    # Get confidence intervals of forecasts
    pred_ci = pred_uc.conf_int()

    # Final settings before the forecast
    ax = dataframe.plot(label='observed', figsize=(20, 15))
    pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.25)
    ax.set_xlabel('Date')
    ax.set_ylabel('Covid cases')

    plt.legend()
    plt.show()


def mse_and_rmse(y_actual,y_predicted):
    # Calculating mse and rmse with sklearn.metrics library
    mse = mean_squared_error (y_actual, y_predicted, squared=True)
    print("MSE:", mse)
    rmse = mean_squared_error(y_actual, y_predicted, squared=False)
    print("RMSE:", rmse)




# Loading data.txt
main('data.txt')
