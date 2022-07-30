from fastapi import FastAPI
from pydantic import BaseModel
from loguru import logger
import joblib
import datetime
import pathlib
import json
import os

from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

# set the current working directory based on this file
CURDIR = str(os.path.abspath(os.getcwd()))

GLOBAL_CONFIG = {
    "model": {
        "featurizer": {
            "sentence_transformer_model": "all-mpnet-base-v2",
            "sentence_transformer_embedding_dim": 768
        },
        "classifier": {
            "serialized_model_path": "../data/news_classifier.joblib"
        }
    },
    "service": {
        "log_destination": "../data/logs.out"
    }
}


class PredictRequest(BaseModel):
    source: str
    url: str
    title: str
    description: str


class PredictResponse(BaseModel):
    scores: dict
    label: str


class TransformerFeaturizer(BaseEstimator, TransformerMixin):
    def __init__(self, dim, sentence_transformer_model):
        self.dim = dim
        self.sentence_transformer_model = sentence_transformer_model

    #estimator. Since we don't have to learn anything in the featurizer, this is a no-op
    def fit(self, X, y=None):
        return self

    #transformation: return the encoding of the document as returned by the transformer model
    def transform(self, X, y=None):
        X_t = []
        for doc in X:
            X_t.append(self.sentence_transformer_model.encode(doc))
        return X_t


class NewsCategoryClassifier:
    def __init__(self, config: dict) -> None:
        self.config = config
        """
        [TO BE IMPLEMENTED]
        1. Load the sentence transformer model and initialize the `featurizer` of type `TransformerFeaturizer` (Hint: revisit Week 1 Step 4)
        2. Load the serialized model as defined in GLOBAL_CONFIG['model'] into memory and initialize `model`
        """

        # initialize the transformer model
        sentence_transformer_model = SentenceTransformer('sentence-transformers/{model}'.format(model=self.config['model']['featurizer']['sentence_transformer_model']))
        # initialize the dimension of the embedding
        dim = self.config['model']['featurizer']['sentence_transformer_embedding_dim']
        
        # initializer the featurizer
        featurizer = TransformerFeaturizer(dim= dim,
                                           sentence_transformer_model=sentence_transformer_model)
        # serialized model path
        serialized_model_path = CURDIR + '/' +  self.config['model']['classifier']['serialized_model_path']
        
        # initialize the model
        model = joblib.load(serialized_model_path)
        
        # create pipeline
        self.pipeline = Pipeline([
            ('transformer_featurizer', featurizer),
            ('classifier', model)
        ])

    def predict_proba(self, model_input: dict) -> dict:
        """
        [TO BE IMPLEMENTED]
        Using the `self.pipeline` constructed during initialization, 
        run model inference on a given model input, and return the 
        model prediction probability scores across all labels

        Output format: 
        {
            "label_1": model_score_label_1,
            "label_2": model_score_label_2 
            ...
        }
        """
        # extract description from input and create an array of out it

        # create the predictions for all classes                                           
        prediction = self.pipeline.predict_proba(model_input)[0]

        # create a dictionary in the results (order of classes is in self.pipeline.classes_)
        output = { label : class_proba for label, class_proba in zip(self.pipeline.classes_, prediction) }

        # return the output
        return output

    def predict_label(self, model_input: dict) -> str:
        """
        [TO BE IMPLEMENTED]
        Using the `self.pipeline` constructed during initialization,
        run model inference on a given model input, and return the
        model prediction label

        Output format: predicted label for the model input
        """
        # predict the label and return it
        return self.pipeline.predict(model_input)[0]


app = FastAPI()

@app.on_event("startup")
def startup_event():
    """
        [TO BE IMPLEMENTED]
        2. Initialize the `NewsCategoryClassifier` instance to make predictions online. You should pass any relevant config parameters from `GLOBAL_CONFIG` that are needed by NewsCategoryClassifier 
        3. Open an output file to write logs, at the destimation specififed by GLOBAL_CONFIG['service']['log_destination']
        
        Access to the model instance and log file will be needed in /predict endpoint, make sure you
        store them as global variables
    """
    global news_category_classifier
    global log

    # initialize the news_category_classifier
    news_category_classifier = NewsCategoryClassifier(GLOBAL_CONFIG)

    # create a handle to a log file
    log = open(CURDIR + '/' + news_category_classifier.config['service']['log_destination'], mode="a")
    
    logger.info("Setup completed")


@app.on_event("shutdown")
def shutdown_event():
    # clean up
    """
        [TO BE IMPLEMENTED]
        1. Make sure to flush the log file and close any file pointers to avoid corruption
        2. Any other cleanups
    """
    # close the log file                                           
    log.close()
                                           
    logger.info("Shutting down application")


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    # get model prediction for the input request
    # construct the data to be logged
    # construct response
    """
        [TO BE IMPLEMENTED]
        1. run model inference and get model predictions for model inputs specified in `request`
        2. Log the following data to the log file (the data should be logged to the file that was opened in `startup_event`, and writes to the path defined in GLOBAL_CONFIG['service']['log_destination'])
        {
            'timestamp': <YYYY:MM:DD HH:MM:SS> format, when the request was received,
            'request': dictionary representation of the input request,
            'prediction': dictionary representation of the response,
            'latency': time it took to serve the request, in millisec
        }
        3. Construct an instance of `PredictResponse` and return
    """
    # start time: capture the start of request, and save it
    start = datetime.datetime.now()

    # extract the model_input
    model_input = request.description

    # check for valid model_input data: must be string and not empty
    if model_input: #and type(model_input) is str:
       # call the classifer to get the scores
       scores     = news_category_classifier.predict_proba([model_input])
       # call the classifier to get the label
       prediction = news_category_classifier.predict_label([model_input])
    else: # input is invalid, not a string
       scores = {}
       prediction = 'NA'
                                           
    # end time: capture time after all execution
    end = datetime.datetime.now()

    # compute the latency as elapsed time in microsends
    latency = (end - start).total_seconds () * 1000 

    # create the response object                                           
    response = PredictResponse(scores=scores, label=prediction)
                                           
    # create data to be logged                                           
    data = {
        'timestamp'  : start.strftime("<%Y:%m:%d %H:%M:%S>"),
        'request'    : dict(request),
        'prediction' : response.dict(),
        'latency'    : latency
    }
    # for debugging
    # print(json.dumps(data))

    # log the data
    log.write(json.dumps(data))
    log.write('\n')
    # we flush the file. It's not necessary per se because python flushes
    # the stream when the file is closed
    log.flush()

    return response

@app.get("/")
def read_root():
    return {"Hello": "World"}
