# Machine-Learning-Tasks
Let's say we have decided that we want to trade in J.P. Morgan. Next, we should decide whether to go long in J.P. Morgan at a given point in time. Thus, the problem statement will be:  Whether to buy J.P. Morgan's stock at a given time or not?

## 1. Target and Features
A machine learning algorithm requires a set of features to predict the target variable.

### Import Libraries
First, we will import the necessary libraries.

### Read the Data
The 15-minute OHLCV data of J.P. Morgan stock price is stored in a CSV file JPM.csv. The data ranges from January 2017 to December 2019. To read the CSV file, we use the read_csv method of pandas. 

### Target Variable
Target variable is what the machine learning model tries to predict in order to solve the problem statement. It is referred to as y.

We will create a column, signal. The signal column will have two labels, 1 and 0. Whenever the label is 1, the model indicates a buy signal. And whenever the label is 0, the model indicates do not buy. We will assign 1 to the signal column whenever the future returns will be greater than 0.

The future_returns can be calculated using the pct_change method of pandas. The pct_change method will calculate the percentage change for the current time period.

Since we want the future returns, we will shift the percentage change for the current period to the previous time period. This can be done using the shift method of pandas.

### Features
In order to predict the signal, we will create the input variables for the ML model. These input variables are called features. The features are referred to as X. We can create the features in such a way that each feature in your dataset has some predictive power.

We will start by creating the 15-minute, 30-minute and 75-minute prior percentage change columns.

Next, we will calculate the technical indicators, RSI and ADX. These can be done by using the RSI and ADX method of the talib library.

Since there are 6.5 trading hours in a day, and ours is a 15-minutes data, the time period will be 6.5*4.

We will now create the simple moving average and rolling correlation of the close price. This can be done by using the mean and the corr method of the pandas library.

We will calculate the daily moving average and correlation.

Let us now calculate the volatility of the stock. This can be done by calculating the rolling standard deviation of the pct_change column.

### Create X and y
Before creating the features (X) and target(y), we will drop the rows with any missing values.

Since we have created features using the original columns of the dataset, we will not consider the original columns (high, low, open, volume, close) in features.

Store the signal column in y and features in X. The columns in the variable X will be the input for the ML model and the signal column in y will be the output that the ML model will predict.

#### Stationarity Check
Since most ML algorithm requires stationary features, we will drop the non-stationary columns from X.

We can use the adfuller method from the statsmodels library to perform this test in Python, and compare the p-value.
- If the p-value is less than or equal to 0.05, you reject H0.
- If the p-value is greater than 0.05, you fail to reject H0.

We can see that all the columns except sma are stationary. Hence, the sma column is dropped from the dataset.

#### Correlation Check
Let us now check for correlation between the features.

We can see that the correlation between volatility and volatility2 is above threshold of 0.7. Hence, we should drop any one of the above columns. We will drop the volatility2 column.

### Display the Final Features

#### Save the Files
The dataframes X and y has the features and target variables stored. For further use, we can export this into a CSV file using the to_csv method of pandas.

### Conclusion
As most of the ML models requires stationary features, the final features in the dataframe are stationary. We have also dropped the highly correlated features. 


## 2. Train - Test Split
The train-test split means splitting the data into two parts:
1. Training data (train_data)
2. Testing data (test_data)

The machine learning algorithm is trained on the train_data, and then it is applied to the test_data. The machine learning output is compared with the actual output for the test_data to evaluate how the model performs on 'unseen' data (data not known at the time of training).

### Import Libraries
First, we will import the necessary libraries.

### Read the Data
We will read the target (y) and features (X) for J. P. Morgan using the read_csv method of the pandas library.

### Train-Test Split
In the train-test split you divide the data into two parts.
Let's use the train_test_split to split the data in an 80% train and 20% test proportion.

A few observations after the train-test split:

1. The dimensions of the original dataset show that there were 7 features and 19318 observations in the feature dataset (X). The target variable (y) has one column and the same number of observations as X.

2. The dimensions of the train_data show that X_train has 7 features and 15454 observations. That is 80% of 19318, rounded down to the nearest integer. The target variable for the train data (y_train) has one column and the same number of observations as X_train.

3. The dimensions of the test_data show that X_test has 7 features and 3864 observations. That is the balance 20% of 19318. The target variable for the train data (y_test) has one column and the same number of observations as X_test.

### Visualise the Data
Let's plot one of the columns of the features to see how the data is split.

We can see that the train_data (blue points) and the test_data (orange points) are randomly shuffled.

Do we want randomly shuffled data for our train and test datasets?

The answer depends on what type of data we are handling. If we are handling discrete observations, then we can shuffle the indices for the train-test split. But we are dealing with financial time-series data. For time-series data, the order of indices matters and we can not do random shuffling. This is because the indices in time series data are timestamps that occur one after the other (in sequence). The data would make no sense if the timestamps are shuffled. The reason for that is simple. We can not use the data from 2021 to train your model, and then use the model to predict the prices in 2017. It is not possible in real life as we do not have access to future data.

### Correct Way of Splitting Time-series Data
To split the time-series data we must not shuffle the datasets. We can specify the shuffle parameter to False. It is set to True by default, so not specifying it in the method call results in a shuffled output.

#### Save the Files
As seen on the plot, the train and test data points are not shuffled. The model is trained on train_data (blue part) and then the performance is evaluated for the test_data (orange part). The previous issue where we possibly were using future data to predict the past will not occur now. In this illustration, the model will be trained on data up to May 2019 (blue part) and then the model will be used to make predictions for the future.


## 3. ML Classification Model Training and Forecasting
We will now use the X_train and y_train to train a machine learning model. The model training is also referred to as "fitting" the model.

After the model is fit, the X_test will be used with the trained machine learning model to get the predicted values (y_pred).

### Import Libraries
First, we will import the necessary libraries.

### Read the Data
The target (y) and features (X) for the train and test dataset is read from the CSV files. 

### Select a Classification Model
Now we will select a classification model, in this case we will use a RandomForestClassifier.

The RandomForestClassifier model from the sklearn package is used to create the classification tree model. 

Parameters:
1. n_estimators: The number of trees in the forest.
2. max_features: The number of features to consider when looking for the best split.
3. max_depth: The maximum depth of a tree.
4. random_state: Seed value for the randomised bootstrapping and feature selection. This is set to replicate results for subsequent runs.

Returns:
A RandomForestClassifier type object that can be fit on the test data and then used for making forecasts.

### Train the Model
Now it is time for the model to learn from the X_train and y_train. We call the fit function of the model and pass the X_train and y_train datasets.

Parameters:
1. model: The model (RandomForestClassifier) object.
2. X_train: The features from the training dataset.
3. y_train: The target from the training dataset.

Returns:
The fit function trains the model using the data passed to it. The trained model is stored in the model object where the fit function was applied.

### Forecast Data
The model is now ready to make forecasts. We can now pass the unseen data (X_test) to the model and obtain the model predicted values (y_pred). To make the forecast, the predict function is called and the unseen data is passed as a parameter.

Parameters:
1. model: The model (RandomForestClassifier) object.
2. X_test: The features from the testing dataset.

Returns:
A numpy array of the predicted outputs is obtained.

Let's make one prediction using the model. For illustration, we are using the first data point in the X_test.

The data is for the 28th May 2019. Let us pass this to the model and get the prediction.

The predicted model output is 1. This means that the model is signaling to take a long position on 28th May 2019. Let's apply the model to all of the testing dataset.

The model predictions are stored in y_pred. 0 means no position and 1 means a long position. With the y_pred we can now place trades using an ML model.

#### Save the Files

### Conclusion
As we can see, the model correctly predicts the first three values of the test_data. But how do we know the accuracy of the model prediction for the entire dataset? We need to learn some metric for measuring the model performance.

## 4. Metrics to Evaluate a Classifier
Let's figure out whether the forecasts are good or bad. To do that, we will use the predicted output (y_pred) and the expected output (y_test).

### Import Libraries
First, we will import the necessary libraries.

### Read the Data
To evaluate the performance, we will read the model predicted values (y_pred) and the expected target values (y_test) from the test_data.

### Accuracy
Accuracy is the total correct predictions divided by the total predictions. We plot the data to see how the correct and incorrect predictions are distributed. The green points are where the prediction was correct and the red points are where the predictions were incorrect.

The accuracy is 51.55%.

#### Confusion Matrix
The confusion matrix is a table that can be used to interpret the model performance. The labels of the confusion matrix are the actions predicted by the model on the x-axis and the expected actions on the y-axis.

Parameters:
1. y_test: The observed target from the training dataset.
2. y_pred: The predicted target from the model.

Returns: A numpy array of the confusion matrix.

The confusion matrix gives us the following information:
1. True Positive: 1007 correct predictions for taking a long position.
2. False Positive: 951 incorrect predictions for taking a long position when the expected action was no position.
3. True Negative: 985 correct predictions for taking no position.
4. False Negative: 921 incorrect predictions for taking no position when the expected action was to take a long position.

#### Classification Report
The scikit-learn library has a function called classification_report which provides measures like precision, recall, f1-score and support for each class. Precision and recall indicate the quality of our predictions. The f1-score gives the harmonic mean of precision and recall. The support values are used as weights to compute the average values of precision, recall and f1-score.

An f1-score above 0.5 is usually considered a good number.

Parameters:
1. y_test: The observed target from the training dataset.
2. y_pred: The predicted target from the model.

Returns:
Classification Report containing precision, recall, f1-score and support.

In the left-most column, you can see the values 0.0 and 1.0. These represent the position as follows:
1. 0 means no position
2. 1 means a long position

So from the table, we can say that the ML Model has an overall accuracy score of 0.52. The accuracy we calculated was 51.55% which is approximately 0.52. Apart from accuracy, we can identify the precision, recall, and f1-score for the signals as well.

Support is the number of actual occurrences of the class in the specified dataset. Thus, in the total signal, there were 1936 occurrences of 0, and 1928 occurrences of the 1 signal.

The accuracy score tells you how the ML model performed in total.

What are macro and weighted average?

Sometimes, the signal values might not be balanced. There could be instances where the number of occurrences for 0 is barely 50 while the number of occurrences for 1.0 is 500. In this scenario, the weighted average will give more weightage to the signal 1. In contrast, the macro average takes a simple average of all the occurrences.

## 5. Strategy Backtesting
So far we have seen different model evaluation metrics like accuracy, precision, recall and f1-score. After we are satisfied with the model performance, we can take the signals generated by them to trade and analyse the returns. Not only returns, but we should also analyse the risk associated with generating the returns.

### Import Libraries
First, we will import the necessary libraries.

### Read the Data
We will read the signals whether to buy J.P. Morgan's stock or not stored in the CSV JPM_predicted_2019.csv using he read_csv method of pandas. Then, store it in a dataframe strategy_data.

Also, read the close price of J.P. Morgan. This is stored in a column close in the CSV file JPM_2017_2019.csv. Store it in close column in strategy_data. While reading the close price data, slice the period to match the signal data.

### Calculate Strategy Returns

### Plot the Equity Curve
We can use an equity curve to visualise how the portfolio value has changed over a period of time. Plot the cumulative_returns columns of the strategy_data to check the same.

We can see that the strategy generated a cumulative returns of 28.10% in seven months.

### Performance Metrics
We can analyse the returns generated by the strategy and the risk associated with them using different performance metrics.

### Annualised returns
It is the average annual return of a strategy.

There are 252 trading days in a year, and 6.5 trading hours in a day. Since we are working with 15-minute data, the number of trading frequencies in a year is  252∗6.5∗4 . And the numerator in the exponent term is  252∗6.5∗4 .

We can see that the average annual return of the strategy is 52.20%.

### Annualised volatility
Annualised volatility is a measure of change in the price over a year. 

The annualised volatility is 14.90%.
Annualised volatility of 14.90% means that for approximately 68% time in a year, the current time's price would differ by less than 14.90% from the previous time.

### Maximum drawdown
Maximum drawdown is the maximum value a portfolio lost from its peak. It is the maximum loss the strategy can make. Higher the value of the drawdown, higher would be the losses.

We can see that the maximum drawdown is 7.94%. This means that the maximum value that the portfolio lost from its peak was 7.94%.

### Sharpe Ratio
Sharpe ratio measures the performance of a portfolio when compared to a risk-free asset. It is the ratio of the returns earned in excess of the risk-free rate to the volatility of the returns.

A portfolio with a higher Sharpe ratio will be preferred over a portfolio with a lower Sharpe ratio.

The Sharpe ratio is 2.89.
The Sharpe ratio of 2.89 indicates that the returns are pretty good when compared to the risk associated.

Note: To keep the notebook simple, the transaction cost and slippage were not considered while analysing the performance of the strategy.
