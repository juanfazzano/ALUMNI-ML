import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report


#! global variablesclear
random_state = 42

#! Open data and base information
raw_data = pd.read_csv("reservas_hoteles/data/reservas_hoteles.csv")
raw_data = pd.read_csv("../data/reservas_hoteles.csv")
raw_data.info()
desc_data = raw_data.describe()
desc_data

# difference target
target_data = raw_data["booking_status"].value_counts()
print(f"min/max = {(target_data.min() / target_data.max()):.2f}%")



#! EDA
# delete columns
EDA_data = raw_data.drop("Booking_ID", axis=1)


# * mapping
def map_column(df, column_name, mapping_dict):
    df[column_name] = df[column_name].map(mapping_dict)
    return df


set(EDA_data["type_of_meal_plan"])
# {'Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected'}
mapping_type_of_meal_plan = {
    "Not Selected": 0,
    "Meal Plan 1": 1,
    "Meal Plan 2": 2,
    "Meal Plan 3": 3,
}

set(EDA_data["room_type_reserved"])
# {'Room_Type 1', 'Room_Type 2', 'Room_Type 3', 'Room_Type 4', 'Room_Type 5', 'Room_Type 6', 'Room_Type 7'}
mapping_room_type_reserved = {
    "Room_Type 1": 1,
    "Room_Type 2": 2,
    "Room_Type 3": 3,
    "Room_Type 4": 4,
    "Room_Type 5": 5,
    "Room_Type 6": 6,
    "Room_Type 7": 7,
}


set(EDA_data["booking_status"])
# {'Canceled', 'Not_Canceled'}
mapping_booking_status = {"Canceled": 1, "Not_Canceled": 0}

EDA_data = map_column(EDA_data, "type_of_meal_plan", mapping_type_of_meal_plan)
EDA_data = map_column(EDA_data, "room_type_reserved", mapping_room_type_reserved)
EDA_data = map_column(EDA_data, "booking_status", mapping_booking_status)


# * one-hot encoding
def one_hot_encode(df, column_name, categories):
    for category in categories:
        df[f"{column_name}_{category}"] = (df[column_name] == category).astype(int)
    df.drop(column_name, axis=1, inplace=True)
    return df


list_market_segment_type = list(set(EDA_data["market_segment_type"]))
# {'Aviation', 'Complementary', 'Corporate', 'Offline', 'Online'}
EDA_data = one_hot_encode(EDA_data, "market_segment_type", list_market_segment_type)
"""
# * outliers detection
# quartiles and IQR
Q1 = EDA_data.quantile(0.25)
Q3 = EDA_data.quantile(0.75)
IQR = Q3 - Q1

# outliers
outliers = (EDA_data < (Q1 - 1.5 * IQR)) | (EDA_data > (Q3 + 1.5 * IQR))
EDA_data_cleaned = EDA_data[~outliers.any(axis=1)]
"""
# * normalization
min_max_scaler = MinMaxScaler()

# list of no boolean {0, 1}
normalize_column = [
    column for column in EDA_data.columns if set(EDA_data[column]) != {0, 1}
]
EDA_data[normalize_column] = min_max_scaler.fit_transform(EDA_data[normalize_column])

#! Graphs

for column in EDA_data.columns:
    plt.figure(figsize=(10,6))
    sns.char(data=raw_data, x="booking_status", y=column)
    
for column in EDA_data.columns:
    plt.figure(figsize=(10,6))
    sns.
#! model

# * data split
X = EDA_data.drop("booking_status", axis=1)
y = EDA_data["booking_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_state
)

rf_model = RandomForestClassifier(random_state=42)

# * gridsearch
"""
# Definir los hiperparámetros a ajustar
param_grid = {
    "n_estimators": [ 500, 1000],
    "max_depth": [10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
}

grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    cv=5,  # crossvalidation
    n_jobs=-1, # para darle POWER
    verbose=2, # print information
    scoring="accuracy",
)
grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
#Best parameters: {'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 500}

"""
rf_model = RandomForestClassifier(
    random_state=random_state,
    n_jobs=-1,
    max_depth=None,
    max_features="sqrt",
    min_samples_leaf=1,
    min_samples_split=2,
    n_estimators=500,
)

rf_model.fit(X_train, y_train)

# * predict
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# * shap
importances = rf_model.feature_importances_
print("Feature Importances:")
print(importances)

# sort shap
feature_names = X_train.columns
sorted_indices = importances.argsort()[::-1]

print("Shap:")
for index in sorted_indices:
    print(f"{feature_names[index]}: {(importances[index]):.2f}%")
# Shap:
# lead_time: 0.32%
# avg_price_per_room: 0.16%
# no_of_special_requests: 0.11%
# arrival_date: 0.09%
# arrival_month: 0.08%
# no_of_week_nights: 0.05%
# no_of_weekend_nights: 0.04%
# market_segment_type_Online: 0.03%
# no_of_adults: 0.02%
# arrival_year: 0.02%
# type_of_meal_plan: 0.02%
# market_segment_type_Offline: 0.02%
# room_type_reserved: 0.02%
# no_of_children: 0.01%
# required_car_parking_space: 0.01%
# market_segment_type_Corporate: 0.01%
# repeated_guest: 0.00%
# no_of_previous_bookings_not_canceled: 0.00%
# market_segment_type_Complementary: 0.00%
# market_segment_type_Aviation: 0.00%
# no_of_previous_cancellations: 0.00%
