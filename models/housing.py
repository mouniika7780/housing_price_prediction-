import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import pickle
import os

class HousingModel:
    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.feature_names = ["square_footage", "bedrooms", "bathrooms","year_built", "lot_size","distance_to_city_center", "school_rating"]
        self.metrics = {}
        self.is_trained = False

    def train(self, data_path: str):
        df = pd.read_csv(data_path)
        X = df[self.feature_names]
        y = df["price"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        self.model.fit(X_train_scaled, y_train)
        y_pred = self.model.predict(X_test_scaled)
        self.metrics = {
            "r2_score": round(r2_score(y_test, y_pred), 4),
            "mae": round(mean_absolute_error(y_test, y_pred), 2),
            "rmse": round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
            "training_samples": len(X_train),
            "test_samples": len(X_test)
        }
        self.is_trained = True
        return self.metrics

    def predict(self, features: list) -> list:
        df = pd.DataFrame(features, columns=self.feature_names)
        scaled = self.scaler.transform(df)
        predictions = self.model.predict(scaled)
        return [round(float(p), 2) for p in predictions]


    def get_coefficients(self) -> dict:
        return dict(zip(self.feature_names, [round(c, 4) for c in self.model.coef_]))
    


    def save(self, path: str = "saved_models/housing_price_model.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)



    @staticmethod
    def load(path: str = "saved_models/housing_price_model.pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)