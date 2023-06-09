import pandas as pd
from giskard import GiskardClient

import spacy
import fr_dep_news_trf
from spacy.matcher import Matcher

NLP = fr_dep_news_trf.load()

def single_prediction(raw_text):
        # print("start single prediction")
        # print(raw_text)
        matcher = Matcher(NLP.vocab)
        pattern = [{"TEXT": {"FUZZY": "écologie"}}]
        matcher.add("greenlite", [pattern])
        proba = 0
        doc = NLP(raw_text)
        matches = matcher(doc)
        # print(matches)
        match_id_amount = 0
        for match_id, start, end in matches:
            string_id = NLP.vocab.strings[match_id]  # Get string representation
            span = doc[start:end]  # The matched span
            match_id_amount += 1
            # print(match_id, string_id, start, end, span.text)
        if match_id_amount == 0:
            proba = 0
        else:
            proba = 1
        return proba

def wrapped_prediction_function(X):
    predict_proba = pd.DataFrame()
    # print("start batch prediction")
    predict_proba["negative"] = X["sentences"].apply(
        lambda x: 1.0 if single_prediction(x) == 0 else 0)
    predict_proba["positif"] = X["sentences"].apply(
        lambda x: 1.0 if single_prediction(x) == 1 else 0)
    predict_proba = predict_proba.to_numpy()
    return predict_proba


test_data = pd.DataFrame()
sentences_dataset = [
    "L'écologie, ou écologie scientifique, est une science qui étudie les interactions des êtres vivants entre eux et avec leur milieu.",
    "L'ensemble des êtres vivants, de leur milieu de vie et des relations qu'ils entretiennent forme un écosystème.",
    "L'écologie fait partie intégrante de la discipline plus vaste qu'est la science de l'environnement (ou science environnementale).",
    "Une conception plus restreinte définit l'écologie comme l'étude des flux de matière et d'énergie (réseaux trophiques) dans un écosystème.",
    "Chaque niveau d'organisation apporte des propriétés émergentes, liées aux interactions entre ces composantes.",
]
test_data["sentences"] = sentences_dataset
test_data["target"] = ["positive", "negative", "positive", "positive", "negative"]


#predict_proba = wrapped_prediction_function(test_data)
#print(predict_proba)


url = "http://localhost:19000"
# you can generate your API token in the Admin tab of the Giskard application (for installation, see: https://docs.giskard.ai/start/guides/installation)
token = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJhZG1pbiIsInRva2VuX3R5cGUiOiJBUEkiLCJhdXRoIjoiUk9MRV9BRE1JTiIsImV4cCI6MTY4NjA3MDk2N30.AluiVBmsd8Dxmpscwx8DkM8JM3hzC5Oq462Q1ktccRY"
client = GiskardClient(url, token)
project = client.create_project("topic_detection",
                                 "TOPIC DETECTION",
                                 "DOWNSTREAM INFERENCE REVIEW")
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
    classification_labels=['negative', 'positive']
)
