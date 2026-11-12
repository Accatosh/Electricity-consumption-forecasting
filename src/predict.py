import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from pathlib import Path

hgb_model: Pipeline | None = None
MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "electricity_consumption_hgb_model.pkl"
)

def load_model() -> Pipeline:
    """
    Loads the trained HGB model into the global variable for the model. Also, returns the model.
    """
    global hgb_model

    if hgb_model is None:
        hgb_model = joblib.load(MODEL_PATH)

    return hgb_model

def predict_with_model(X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """
    Predicts electrical consumption with the trained HGB model.
    """
    global hgb_model
    if hgb_model is None:
        load_model()
    return hgb_model.predict(X)


def prediction_interface():
    """
    Starts a program that lets you predict the electrical consumption for the next hour.
    """
    print("Welcome to Electricity Consumption Forecasting.")
    print("By using a Histogram-based Gradient Boosting model, this program tries to forecast")
    print("the electrical consumption of a household in Paris.")
    print("Press Q and then enter to quit whenever.")
    

    features = ['Prev_sub_metering_1', 'Prev_sub_metering_2', 'Prev_sub_metering_3', 'lag_1h', 'hour', 'day_of_week']
    questions = ["What was the energy sub-metering No. 1 for the previous hour? ",
                 "What was the energy sub-metering No. 2 for the previous hour? ",
                 "What was the energy sub-metering No. 3 for the previous hour? ",
                 "What was the electrical consumption the previous hour? ",
                 "For which hour do you want to predict? (0-23) ",
                 "For which day of the week are you predicting? (Monday=0, ..., Sunday=6) "]
    received_input = {}
    in_text = ""

    i = 0
    while in_text.upper() != "Q":
        if (i % len(questions) == 0):
            if (i != 0):
                received_input = pd.DataFrame.from_dict(received_input) 
                result = predict_with_model(received_input)[0]
                received_input = {}
                print("The predicted electrical consumption is: ", result, " kW.")
                in_text = input("Press enter to continue. ")
                i = 0
                if in_text.upper() == "Q":
                    break

            print("---")
            print("Please insert some values to start forecasting.")
            print("Write 'NA' to insert a missing value.")
            in_text = input("Press enter to continue. \n")
            if in_text.upper() == "Q":
                break

        in_text = input(questions[i])

        try:
            received_input[features[i]] = [float(in_text)]
        except ValueError:
            if in_text == "NA":
                received_input[features[i]] = [np.nan]
            else:
                continue
        i += 1

    print("Goodbye.")