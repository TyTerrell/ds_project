import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# create the class to then call later for model deployment on local device

class RandomForestPredictor:
    def __init__(self):
        self.model = None

    def train(self, features, target):
        """
        this is used to train the model, using the test predictors and test response as inputs.
        """
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(features, target)

    def save_model(self, filename='rf_model.pkl'):
        """
        using joblib, we can save the model locally so it can be loaded in later for use.
        """
        joblib.dump(self.model, filename)
        print(f"Model saved to {filename}")

    def load_model(self, filename='rf_model.pkl'):
        """
        using joblib's load function, we can call to load the model saved on the local device.
        """
        self.model = joblib.load(filename)
        print(f"Model loaded from {filename}")

    def predict(self, input_data):
        """
        here we can use predict function for the RandomForestRegressor model inside the class, so we can just refer to a
        variable and easily access its functionality.
        """
        if self.model is None:
            raise ValueError("Model is not trained yet. Please train or load a model before prediction.")
        return self.model.predict(input_data)
    
# going through this process gives me the impression that using specific functionality for modeling that isn't in published sources can be done this way.
# for example, a custom neural network with custom layers can be designed here in the class, and then very easily called and accessed in the main project, whether it be in the training stages
# or the deployment for use - although that would be likely uploaded to the cloud etc.