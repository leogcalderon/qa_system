import torch
import justext
import requests
import logging
import itertools
from googlesearch import search
from rank_bm25 import BM25Okapi

logging.basicConfig(level='INFO')

def get_context(question, n=5):
    """
    Busca en Google Search la pregunta, extrae los textos de los
    n primeros resultados para luego devolver el contexto que mas
    puntaje bm25 tiene.
    """
    logging.info(f'[GOOGLE SEARCH] Searching articles for Q: {question}')
    google_search = search(question)
    top_n_searchs = itertools.islice(google_search, n)
    contexts = []
    logging.info('[GOOGLE SEARCH] Parsing articles')
    for result in top_n_searchs:
        response = requests.get(result)
        paragraphs = justext.justext(response.content, justext.get_stoplist("English"))
        context = ''
        for paragraph in paragraphs:
            if not paragraph.is_boilerplate:
              context += paragraph.text + ' '

        contexts.append(context)

    tokenized_contexts = [context.split(" ") for context in contexts]
    bm25 = BM25Okapi(tokenized_contexts)
    tokenized_question = question.split(" ")
    return bm25.get_top_n(tokenized_question, contexts, n=1)[0]

def return_inputs(question, context, tokenizer, device, chunk_size=384):
    """
    A partir de una pregunta y contexto, devuelve un diccionario
    con tensores para el modelo.
    """
    input_ids = (question + context + [tokenizer.sep_token_id])
    start_context = input_ids.index(tokenizer.sep_token_id) + 1

    token_type_ids = (
        [0] * start_context +
        [1] * (len(input_ids) - start_context) +
        [0] * (chunk_size - len(input_ids))
    )

    attention_mask = (
        [1] * len(input_ids) +
        [0] * (chunk_size - len(input_ids))
    )

    input_ids = (
        input_ids +
        [tokenizer.pad_token_id] * (chunk_size - len(input_ids))
    )

    return {
        'input_ids': torch.Tensor([input_ids]).long().to(device),
        'attention_mask': torch.Tensor([attention_mask]).long().to(device),
        'token_type_ids': torch.Tensor([token_type_ids]).long().to(device)
    }

def get_inputs(question, context, tokenizer, device, chunk_size=384):
    """
    Devuelve una lista con los chunks de chunk_size
    listos para alimentar el modelo.
    """
    question = tokenizer(question)['input_ids']
    context = tokenizer(context, add_special_tokens=False)['input_ids']
    chunks = []

    # Si la [CLS] pregunta [SEP] contexto [SEP] son menores a 384
    # generamos un ejemplo
    if len(question + context + [tokenizer.sep_token_id]) <= chunk_size:
        chunks.append(return_inputs(question, context, tokenizer, device))

    # Si superan 384, devolvemos una lista con los chunks preprocesados de 384
    else:
        for i in range(0, len(question + context + [tokenizer.sep_token_id]), chunk_size - len(question)):
            context_chunk = context[i:i + chunk_size - len(question) - 1]
            chunks.append(return_inputs(question, context_chunk, tokenizer, device))

    return chunks

def get_answer(question, model, tokenizer, device, n_answers=1):
    """
    Busca el mejor contexto de la busqueda en Google search
    tokeniza y prepara tensores para el modelo y devuelve
    la/s respuesta/s predicha
    """
    best_context = get_context(question, n=5)
    chunks = get_inputs(question, best_context, tokenizer, device)
    answers = []
    logging.info('[MODEL] Making predictions')
    model.eval()
    model.to(device)
    for chunk in chunks:
        results = model(**chunk)
        start_idx = results[0].argmax(1).squeeze(0)
        end_idx = results[1].argmax(1).squeeze(0)
        answer = tokenizer.decode(
            chunk['input_ids'][0, start_idx:end_idx]
        )
        answers.append(answer)
    answers = [ans for ans in answers if len(ans) > 0][:n_answers]
    logging.info(f'[MODEL] Predictions: {answers}')
    return answers
