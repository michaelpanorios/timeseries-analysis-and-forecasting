# Time-series analysis and forecasting
The purpose of this study is to analyze the COVID-19 time series for different geographical areas of a country. In particular, a data set is given in which the number of COVID-19 cases is recorded on a daily basis for different regions of Germany for a period of one year. There are basically many time series - one for each area. The number of cases is cumulative, ie the number given in the file for a date refers to the total number of cases up to that date. Also time series analysis techniques for the study of the whole. Specifically, the construction of a predictive model is required, the evaluation of accuracy.

## As a beginner
For the elaboration of the work, I had to spend the first hours reading the relevant slides as well as various other sources of the internet, in order to understand the study of time series and the creation of predictive models. After completing the above I proceeded on building the code. First, I tried to parse the given txt file, and then I tried to subtract all the days with yesterday so as to identify the daily cases and not their cumulative. Next, I created the methods named `time-series`, `mse_and_rmse`, `and forecast`.

## Problems and solutions
I thought that because the number of cities is large, it was a good practice to look up the file in the main based on the id of each city in order to automate the process for each city. The detailed explanation follows the technical report. Finally, the problems I encountered were in the construction of the predictive model as I used various techniques and libraries which did not bring proper results. However, the technique I eventually used solved my problem.

## Tools
For the work I used Python which is the best language for data analysis, as it has a very user-friendly interface and a multitude of packages and libraries that speed up the process of analysis in large files, which ultimately simplifies the methodology and saves significant time. compared to other programming languages. Finally, the libraries I used in the project are the following:

• Matplotlib.pyplot: for the construction of graphs. 

• Sklearn.metrics: for calculating MSE and RMSE. 

• Pandas: to identify the text file.

## Reading the report
Unfortunately, the report is in Greek language. The file that contains the analysis of code and the methodologies is uploaded with the name of ***Τεχνική Αναφορά.pdf***. I have also uploaded all the time-series graphs of each city based on ***data.txt*** file.
