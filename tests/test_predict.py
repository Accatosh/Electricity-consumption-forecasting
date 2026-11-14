import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from src.predict import get_model, reset_model, load_model, predict_with_model

# Tests using pytest

pred_input = pd.DataFrame.from_dict([{
        'Prev_sub_metering_1': 0, 
        'Prev_sub_metering_2': 0, 
        'Prev_sub_metering_3': 0, 
        'lag_1h': 0, 
        'hour': 0, 
        'day_of_week': 0}]) 

def test_load_model_stores_loaded_model():
    reset_model()
    assert get_model() is None

    result = load_model()

    assert get_model() is not None, "load_model does not save the loaded model."
    assert type(result) == Pipeline

def test_predict_with_model_loads_model_if_needed():
    global pred_input
    reset_model()
    assert get_model() is None

    pred = predict_with_model(pred_input)

    assert get_model() is not None, "predict_with_model does not load the model when a model has not been loaded."
    assert type(pred) == np.ndarray
    assert type(pred[0]) == np.float64

def test_predict_with_model_uses_loaded_model():
    global pred_input

    reset_model()
    assert get_model() is None

    loaded_model = load_model()

    pred = predict_with_model(pred_input)

    assert get_model() == loaded_model, "The pre-loaded model is not used."
    assert type(pred) == np.ndarray
    assert type(pred[0]) == np.float64
