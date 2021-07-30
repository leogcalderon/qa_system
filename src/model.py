from torch import nn
from transformers import DistilBertModel, DistilBertConfig

class QAModel(nn.Module):
    """
    Baseline con Bert-base-cased pre entrenado y dos capas lineares
    para calcular el comienzo y final de la respuesta. Ademas
    devuelve el token cls para el entrenamiento adversario.
    """
    def __init__(self, dropout=0.3):
        super().__init__()
        self.configuration = DistilBertConfig(vocab_size=28996)
        self.bert = DistilBertModel(self.configuration)
        self.emb_dim = self.bert.embeddings.word_embeddings.embedding_dim
        self.max_len = self.bert.embeddings.position_embeddings.num_embeddings

        self.start_mlp = nn.Linear(self.emb_dim, 1)
        self.end_mlp = nn.Linear(self.emb_dim, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, token_type_ids, attention_mask):

        # input_ids, token_type_ids, attention_mask = [batch_size, chunk_size]
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        last_hidden_state = bert_output.last_hidden_state
        # last_hidden_state = [batch_size, chunk_size, emb_dim]

        # Tirar a 0 todos los valores que son pregunta y padding.
        context_hidden_state = last_hidden_state * token_type_ids.unsqueeze(-1).repeat(1, 1, self.emb_dim)
        # context_hidden_state = [batch_size, chunk_size, emb_dim]

        # MLP para softmax del token de entrada y de salida
        start_logits = self.start_mlp(self.dropout(context_hidden_state))
        end_logits = self.end_mlp(self.dropout(context_hidden_state))
        # start_logits, end_logits = [batch_size, chunk_size, 1]

        return start_logits, end_logits
