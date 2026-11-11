# Electricity consumption forecasting

## Overview

Predict hourly household electricity consumption using historical consumption data and machine learning.

Most of the project work and workflow are in the Jupyter notebook. Based on this, some Python modules have been created to reproduce the project and make use of the data processing for this dataset.

## Dataset

The dataset used in this project is:

Hebrail, G. & Berard, A. (2006). Individual Household Electric Power Consumption [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C58K54.

It is a time-series dataset with nine features. It uses a one-minute sampling rate over almost four years. The dataset contains all timestamps but has some missing values, around 1.25% of the rows, where measurement values are missing. 

Variable information according to the source: (citation)
1. date: Date in format dd/mm/yyyy
2. time: time in format hh:mm:ss
3. global_active_power: household global minute-averaged active power (in kilowatt)
4. global_reactive_power: household global minute-averaged reactive power (in kilowatt)
5. voltage: minute-averaged voltage (in volts)
6. global_intensity: household global minute-averaged current intensity (in amperes)
7. sub_metering_1: energy sub-metering No. 1 (in watt-hour of active energy). It corresponds to the kitchen, containing mainly a dishwasher, an oven and a microwave (hot plates are not electric but gas-powered).
8. sub_metering_2: energy sub-metering No. 2 (in watt-hour of active energy). It corresponds to the laundry room, containing a washing machine, a tumble dryer, a refrigerator and a light.
9. sub_metering_3: energy sub-metering No. 3 (in watt-hour of active energy). It corresponds to an electric water heater and an air-conditioner.

## Problem

Given information available up to time t, predict electricity consumption at t+1.

## Methodology

### Data processing

- Change missing measures from ?-symbols to NaN values.
- Change the type of the measurement values to float.
- Make the date and time features into datetime format and set it as the index of the Pandas dataframe.
- Aggregate the data to hourly data instead of minutely. (To suit the project's purpose)
- Impute missing values by taking the average of past data points. For the training data, future data was also included in the average.

See *preprocessing.py* for more details on the data processing, especially the clean_data function.

### Features

With the goal of improving the prediction accuracy, three features were removed from the dataset, and three new ones were added.

Based on how the RMSE and MAE were affected by the removal of a feature, *Global_reactive_power*, *Voltage* and *Global_intensity* were deemed less important and thus removed from the dataset.

A feature that measures the electrical consumption of the previous hour, as well as features for the current hour and day of the week, was added to improve performance.

### Models

- Previous-hour baseline
- Previous-day baseline
- Previous-week baseline
- Linear Regression
- Random Forest
- Histogram-based Gradient Boosting
- Single Layer Artificial Neural Network
- Small Multilayer Perceptron

### Evaluation

MAE and RMSE were used to evaluate the performance on both the validation set and the test set. 

When choosing a final model, time-series cross-validation was used to find a model that receives a better score consistently. 

## Results

The results for the validation set are as follows: 

| Model | MAE | RMSE |
|---|---:|---:|
| Previous-hour baseline | 0.406 | 0.616 |
| Previous-day baseline | 0.576 | 0.823 |
| Previous-week baseline | 0.573 | 0.819 |
| Linear Regression | 0.381| 0.528 |
| Random Forest | 0.342 | 0.490 |
| HistGradientBoosting | 0.331 | 0.477 |
| Single Layer Artificial Neural Network | - | 0.519 |
| Small Multilayer Perceptron | - | 0.502 |

Note that the MAE was not collected for the neural networks.

The results for the test set were:
| Model | MAE | RMSE |
|---|---:|---:|
| Previous-hour baseline | 0.377 | 0.575 |
| Fine-tuned HistGradientBoosting | 0.316 | 0.448 |

A 95% confidence interval for the RMSE is [0.429, 0.465]. So we estimate that, with 95% confidence, the model's true population-level RMSE on unseen data will be between 0.429 and 0.465. This is a Student's t confidence interval for MSE, and then taking the square root. This calculation assumes independent observations and may underestimate uncertainty if temporal dependence is present in the prediction errors, which is likely based on previous analysis.

The mean value for the electrical consumption is 1.092 kW, while the median is 0.602 kW.

## Error Analysis

Error analysis on the final model shows that the 10 worst errors do not differ much between the test set and the validation set. Just looking at the worst errors when using the test set, it shows that they're mostly in the evening and at night. The errors are also evenly distributed among the days of the week, although Mondays and Thursdays perform worse than other days. 

## Findings

From the validation set results, it can be seen that all the models had lower MAE and RMSE than the baselines. The histogram-based gradient boosting model outperformed all the other models in terms of both MAE and RMSE.

The final model's MAE and RMSE can be considered quite high. The following table shows the MAE and RMSE relative to the median or mean value of the electrical consumption:
| Metric | Mean | Median |
|---|---:|---:|
| MAE  | 28.9% | 52.5% |
| RMSE | 40.9% | 74.3% |

For example, looking at the RMSE relative to the median, the RMSE is 74.3% of the median. This means the errors are relatively large compared to the typical magnitude of the electrical consumption. The MAE being 52.5% of the median also suggests this. Therefore, the model's predictions may not be very accurate.


## Limitations

Since the dataset is only from one household in Paris, the results may not hold for other households, especially outside Paris and France. Thus, the model may perform especially worse for data from elsewhere. 

The model also only predicts the next hour, as it relies on data from the previous hour.

Features, such as weather, electricity prices and the number of occupants, could be added to improve accuracy.

The most recent data point in the dataset is from 2010, and may thus not represent current electrical consumption patterns, especially since technology has changed during the 16 years since.

## How to run

A simple program to predict the next hour's electrical consumption can be started by importing the module *predict.py* and calling the function *prediction_interface()*. The module also contains other functions to use and forecast with the final model.

The files *preprocessing.py* and *features.py* contain functions to modify the dataset to how it was used in this project.