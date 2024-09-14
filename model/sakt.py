import logging
import os
import numpy as np
import torch
from torch.nn import Module, Parameter, Embedding, Sequential, Linear, ReLU, \
    MultiheadAttention, LayerNorm, Dropout
from torch.nn.init import kaiming_normal_
from torch.nn.functional import binary_cross_entropy
from sklearn import metrics


class SAKT(Module):
    """
        This implementation has a reference from: \
        https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoderLayer

        Args:
            number_questions: the total number of the questions(KCs) in the given dataset
            n: the length of the sequence of the questions or responses
            d: the dimension of the hidden vectors in this model
            number_attention_heads: the number of the attention heads in the \
                multi-head attention module in this model
            dropout: the dropout rate of this model
    """

    def __init__(self, number_questions, n, d, number_attention_heads, dropout):
        super().__init__()
        self.number_questions = number_questions
        self.n = n
        self.d = d
        self.number_attention_heads = number_attention_heads
        self.dropout = dropout

        self.M = Embedding(self.number_questions * 2, self.d)
        self.E = Embedding(self.number_questions, d)
        self.P = Parameter(torch.Tensor(self.n, self.d))

        kaiming_normal_(self.P)

        self.attn = MultiheadAttention(
            self.d, self.number_attention_heads, dropout=self.dropout
        )
        self.attn_dropout = Dropout(self.dropout)
        self.attn_layer_norm = LayerNorm(self.d)

        self.FFN = Sequential(
            Linear(self.d, self.d),
            ReLU(),
            Dropout(self.dropout),
            Linear(self.d, self.d),
            Dropout(self.dropout),
        )
        self.FFN_layer_norm = LayerNorm(self.d)

        self.pred = Linear(self.d, 1)


    def forward(self, q, r, qry):

        """
            Args:
                q: the question(KC) sequence with the size of [batch_size, n]
                r: the response sequence with the size of [batch_size, n]
                qry: the query sequence with the size of [batch_size, m], \
                    where the query is the question(KC) what the user wants \
                    to check the knowledge level of

            Returns:
                p: the knowledge level about the query
                attn_weights: the attention weights from the multi-head \
                    attention module
        """
        x = q + self.number_questions * r
        M = self.M(x).permute(1, 0, 2)
        E = self.E(qry).permute(1, 0, 2)
        P = self.P.unsqueeze(1)

        causal_mask = torch.triu(
            torch.ones([E.shape[0], M.shape[0]]), diagonal=1
        ).bool()

        M = M + P

        S, attn_weights = self.attn(E, M, M, attn_mask=causal_mask)
        S = self.attn_dropout(S)
        S = S.permute(1, 0, 2)
        M = M.permute(1, 0, 2)
        E = E.permute(1, 0, 2)
        S = self.attn_layer_norm(S + M + E)
        F = self.FFN(S)
        F = self.FFN_layer_norm(F + S)
        p = torch.sigmoid(self.pred(F)).squeeze()

        return p, attn_weights

    def train_model(
            self, train_loader, test_loader, number_epochs, optimizer, check_point_path
    ):
        """
            Args:
                train_loader: the PyTorch DataLoader instance for training
                test_loader: the PyTorch DataLoader instance for test
                number_epochs: the number of epochs
                optimizer: the optimizerimization to train this model
                check_point_path: the path to save this model"s parameters
        """
        accuracies = []
        loss_means = []

        max_accuracy = 0

        for epoch in range(1, number_epochs + 1):
            loss_mean = []

            for data in train_loader:
                question, response, question_shift, response_shift, masked = data

                self.train()

                predict, _ = self(question.long(), response.long(), question_shift.long())
                predict = torch.masked_select(predict, masked)
                true_score = torch.masked_select(response_shift, masked)

                optimizer.zero_grad()
                loss = binary_cross_entropy(predict, true_score)
                loss.backward()
                optimizer.step()

                loss_mean.append(loss.detach().cpu().numpy())

            with torch.no_grad():
                for data in test_loader:
                    question, response, question_shift, response_shift, masked = data

                    self.eval()

                    predict, _ = self(question.long(), response.long(), question_shift.long())
                    predict = torch.masked_select(predict, masked).detach().cpu()
                    true_score = torch.masked_select(response_shift, masked).detach().cpu()

                    accuracy = metrics.roc_auc_score(
                        y_true=true_score.numpy(), y_score=predict.numpy()
                    )

                    loss_mean = np.mean(loss_mean)

                    logging.debug(
                        "Epoch: {},   AUC: {},   Loss Mean: {}"
                            .format(epoch, accuracy, loss_mean)
                    )

                    if accuracy > max_accuracy:
                        torch.save(
                            self.state_dict(),
                            os.path.join(
                                check_point_path, "model.ckpt"
                            )
                        )
                        max_accuracy = accuracy

                    accuracies.append(accuracy)
                    loss_means.append(loss_mean)

        return accuracies, loss_means
