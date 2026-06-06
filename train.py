import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models import HousingModel

def train_and_save():
    model = HousingModel()
    model.train("data/housing.csv")
    model.save("saved_models/housing_price_model.pkl")

if __name__ == "__main__":
    train_and_save()