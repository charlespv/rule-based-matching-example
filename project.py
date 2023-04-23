import pandas as pd
from giskard import GiskardClient

test_data = pd.DataFrame()
test_data["sentences"] = ["apple", "apples", "orange", "kiwi", "Apple"]
test_data["sentences"] = test_data["sentences"].astype("str")
test_data["target"] = ["positif", "positif", "negative", "negative", "positif"]
test_data["target"] = test_data["target"] .astype("str")


def wrapped_prediction_function(X):
    predict_proba = pd.DataFrame()
    predict_proba["negative"] = X["sentences"].apply(
        lambda x: 1.0 if x != "apple" else 0.0)
    predict_proba["positif"] = X["sentences"].apply(
        lambda x: 1.0 if x == "apple" else 0.0)
    predict_proba = predict_proba.to_numpy()
    print(predict_proba)
    return predict_proba


url = "http://localhost:19000"
# you can generate your API token in the Admin tab of the Giskard application (for installation, see: https://docs.giskard.ai/start/guides/installation)
token = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJhZG1pbiIsInRva2VuX3R5cGUiOiJBUEkiLCJhdXRoIjoiUk9MRV9BRE1JTiIsImV4cCI6MTY4NjA3MDk2N30.AluiVBmsd8Dxmpscwx8DkM8JM3hzC5Oq462Q1ktccRY"
client = GiskardClient(url, token)
# project = client.create_project("topic_detection",
#                                 "TOPIC DETECTION",
#                                 "DOWNSTREAM INFERENCE REVIEW")
topic_detection = client.get_project("topic_detection")
topic_detection.upload_model_and_df(
    prediction_function=wrapped_prediction_function,
    model_type='classification',
    df=test_data,
    column_types={
        'sentences': 'text',
        'target': 'category'
    },
    target='target',
    feature_names=['sentences'],
    classification_labels=['negative', 'positif']
)
