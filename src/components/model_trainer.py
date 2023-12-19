import os
import sys

from src.logging import logging
from src.exception import CustomException
from src.utils import save_object,evaluate_model
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("splitting training and test input data")

            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Linear Regression":LinearRegression(),
                "Random Forest Regression":RandomForestRegressor(),
                "Ada Boost Regreesion":AdaBoostRegressor(),
                "Gradient Boost Regression":GradientBoostingRegressor(),
                "Decision Tree Regressor":DecisionTreeRegressor(),
                "KNeighbor Regression":KNeighborsRegressor(),
                "XGBoost Regression":XGBRegressor()
            }

            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)
            # print("model_report :{}",format(model_report))    
            best_model_score=max(sorted(model_report.values()))
            # print("Best model score",format(best_model_score))
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and Test dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test,predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e,sys)

