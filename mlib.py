"""MLOps Library"""

# Handling data
import numpy as np
import pandas as pd

# Models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Model export
import joblib

# logging
import logging

import json

logging.basicConfig(level=logging.INFO)

MODEL_PATH = "model/model_binary_class.dat.gz"
MODEL_OUTPUT_PATH = "model/model_outputs.json"


def load_model(model=MODEL_PATH):
    """Load model from disk"""

    clf = joblib.load(model)
    return clf


def load_outputs():
    """Load model from disk"""

    with open(MODEL_OUTPUT_PATH, encoding='utf8') as json_file:
        data_loaded = json.load(json_file)
    return (data_loaded["f1_value"], data_loaded["threshold"])


def data():
    """Read data"""
    df = pd.read_csv("raw_data/Train Data.csv")
    return df


def change_data_types(df):
    """change data types for better manage"""

    # cm
    df["height"] = df["height"].str.extract("(\\d+)").astype(int)
    # kg
    df["weight"] = df["weight"].str.extract("(\\d+)").astype(int)
    # euros
    df["Value"] = pd.to_numeric(df["Value"].str.replace("€|\\.", "", regex=True))
    df["Wage"] = pd.to_numeric(df["Wage"].str.replace("€|\\.", "", regex=True))

    return df


def get_position_zone(df):
    """Transform 'preferred_positions' to 'position_zone'"""
    position_zone = []
    for x in df["preferred_positions"]:

        listb = {"DEFENDING": 0, "MIDFIELD": 0, "ATTACKING": 0, "GOALKEEPER": 0}
        for y in x:
            if y in ["GK"]:
                listb["GOALKEEPER"] = 1
            else:
                if y in ["LF", "RF", "CF", "ST"]:
                    listb["ATTACKING"] = listb["ATTACKING"] + 1
                if y in ["CAM", "RM", "RW", "CDM", "CM", "LM", "LW"]:
                    listb["MIDFIELD"] = listb["MIDFIELD"] + 1
                if y in ["LB", "LWB", "RB", "RWB", "CB"]:
                    listb["DEFENDING"] = listb["DEFENDING"] + 1

        # Keep the position with the highest value
        position_zone.append(max(listb, key=listb.get))

    df.loc[:, "position_zone"] = position_zone
    df = df.drop(columns=["preferred_positions"])

    return df


def one_hot_coding(df):
    """Conver columns type 'object' to one hot coding, then
    delete leaving only the One hot coding"""
    dummies_object = None
    columns_type_object = []

    for i, column_type in enumerate([str(d) for d in df.dtypes]):
        if column_type == "object":
            column_name = df.columns[i]
            columns_type_object.append(column_name)

            dummies = pd.get_dummies(df[column_name], prefix=column_name)

            if dummies_object is None:
                dummies_object = dummies
            else:
                dummies_object = pd.concat([dummies_object, dummies], axis=1)

    df = df.drop(columns_type_object, axis="columns")
    df = pd.concat([df, dummies_object], axis=1)
    return df


def clean_data():
    df = data()
    df = df.drop(columns=["ID"])
    df["nivelEstudio"] = df.nivelEstudio.replace("Maestr√≠a", "Maestria")
    df["edad"] = df["edad"].fillna(df.edad.mean())
    # edad
    df["edad"] = df["edad"].fillna(df.edad.mean())
    # emailScore
    df["emailScore"] = df["emailScore"].fillna(0)
    # browser
    df["browser"] = df["browser"].fillna("UNKNOWN")
    #  NUMTDC_AV
    cond_list = [df["NUMTDC_AV"] <= 3, df["NUMTDC_AV"] > 3]
    choice_list = ["3 o menos TC", "Mas de 3 TC"]

    df["NUMTDC_AV"] = np.select(cond_list, choice_list, default="UNKNOWN")
    # 'Proporción de gasto'
    df["Proporción de gasto"] = df["gastosMensuales"] / df["ingresosMensuales"]

    return df


def retrain(tsize=0.3, model_name="model_binary_class.dat.gz"):
    """Retrains the model
    See this notebook: Baseball_Predictions_Export_Model.ipynb
    """
    df = clean_data()
    y = df["label"].values  # Target
    # Transform to log
    X = df.drop(["label"], axis=1)  # Feature(s)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=tsize, random_state=100, stratify=y
    )

    model = LogisticRegression()

    numeric_pipeline = Pipeline(steps=[("scale", MinMaxScaler())])

    categorical_pipeline = Pipeline(
        steps=[("one-hot", OneHotEncoder(handle_unknown="ignore", sparse=False))]
    )

    numerical_features = X_train.select_dtypes(include=["number"]).columns
    categorical_features = X_train.select_dtypes(exclude=["number"]).columns

    full_processor = ColumnTransformer(
        transformers=[
            ("number", numeric_pipeline, numerical_features),
            ("category", categorical_pipeline, categorical_features),
        ]
    )

    pipeline = Pipeline(steps=[("preprocess", full_processor), ("model", model)])

    pipeline.fit(X_train, y_train)
    thresholds = np.linspace(0, 1, 100)

    f1_values = [
        get_f1_by_threshold(threshold, pipeline, X_valid, y_valid)
        for threshold in thresholds
    ]

    df = pd.DataFrame({"f1_values": f1_values, "threshold": np.linspace(0, 1, 100)})

    best_values = df.sort_values("f1_values", ascending=False).iloc[0]

    f1_value, threshold = best_values

    logging.debug(f"best f1_value: {f1_value}")
    logging.debug(f"threshold: {threshold}")

    # We can write the new model
    # joblib.dump(model, model_name)
    return round(f1_value, 4), model_name


def get_f1_by_threshold(threshold, model_pipline, X_valid, y_valid):
    y_pred_pos = model_pipline.predict_proba(X_valid)[
        :, 1
    ]  # probabilidades de pertenecer a la clase 1
    y_pred_class = y_pred_pos > threshold

    return f1_score(y_valid, y_pred_class)


def scale_input(pX, df):
    """Scales values for input"""
    df = df.drop(columns=["Value"])
    input_scaler = StandardScaler().fit(df)
    scaled_input = input_scaler.transform(pX)
    return scaled_input


def scale_target(target, df):
    """Scales Target 'y' Value"""

    y = df["Value"].values  # Target
    y = np.log(y)
    y = y.reshape(-1, 1)  # Reshape
    scaler = StandardScaler()
    y_scaler = scaler.fit(y)
    scaled_target = y_scaler.inverse_transform(target)
    return scaled_target


def human_readable_payload(pred_proba, threshold):
    # """Takes numpy array and returns back human readable dictionary"""

    pred_class = pred_proba > threshold

    if pred_class == 1:
        pred_class = "Moroso"
    else:
        pred_class = "No Moroso"

    result = {
        "Type_user": pred_class,
        "probability to be Moroso": round(pred_proba, 2),
        "threshold": round(threshold, 2),
    }
    return result


def predict(pX):
    # """Takes weight and predicts height"""

    clf = load_model()  # loadmodel
    threshold = load_outputs()[1]
    #  NUMTDC_AV
    cond_list = [pX["NUMTDC_AV"] <= 3, pX["NUMTDC_AV"] > 3]
    choice_list = ["3 o menos TC", "Mas de 3 TC"]

    pX["NUMTDC_AV"] = np.select(cond_list, choice_list, default="UNKNOWN")
    # 'Proporción de gasto'
    pX["Proporción de gasto"] = pX["gastosMensuales"] / pX["ingresosMensuales"]

    pred_proba = float(clf.predict_proba(pX)[:, 1])  #
    result = human_readable_payload(pred_proba, threshold)
    return result
