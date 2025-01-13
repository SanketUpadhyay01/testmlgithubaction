from src.train import train_model
from src.predict import predict_spcies
import numpy as np

def test_model_accuracy():
    accuracy = train_model()
    assert accuracy > 0.8, "Model accuracy should be above 80%"

def test_predict_species():
    setosa_features = [5.1, 3.5, 1.4, 0.2]
    prediction = predict_spcies(setosa_features)
    assert prediction in [0, 1, 2], "Prediction should be a valid class"
