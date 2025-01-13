import joblib
import numpy as np

def predict_spcies(features):
    model=joblib.load('model.joblib')
    prediction  = model.predict([features])
    return prediction[0]
