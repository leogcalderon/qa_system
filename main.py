import torch
from fastapi import FastAPI, Form
from transformers import BertTokenizer
from src import (
    utils,
    model
)

MODEL_PATH = 'model/qa.pkl'

description = """
Busca en Google Search el mejor articulo para la pregunta, para luego
extraer la respuesta de dicho articulo.
"""

app = FastAPI(title='QA System', description=description)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.QAModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
tokenizer = BertTokenizer.from_pretrained('distilbert-base-cased')

@app.get('/predict')
def predict(question: str):
    answers = utils.get_answer(
        question, model, tokenizer, device, n_answers=1
    )
    return {'question': question, 'answers': answers}
