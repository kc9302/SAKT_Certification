from typing import Type
import torch
from torch.utils.data import DataLoader
import time
import datetime
from sklearn import metrics
import logging
import numpy as np
np.set_printoptions(threshold=np.inf, linewidth=np.inf)


# 초기화 파일에서 선언한 함수 불러오기.
from __init__ import config, device, find_datasets_path

# ccommon library.
from common.utils import collate_fn, logging_sakt_config

# Dataloader library.
from data_loaders.ai_hub_eval import AI_HUB

# KT model library.
from model.sakt import SAKT


def eval_model(model, loader, epoch):

    # 일반화 성능 검증
    with torch.no_grad():
        for data in loader:
            logging.debug(
                "\n" + "\n" + " ###################### " + \
                "\n" + " ##### start test ##### " + \
                "\n" + " ###################### " + "\n"
            )

            question, response, question_shift, response_shift, masked = data

            model.eval()
            logging.debug("\n" + "\n" + " ############################# " + \
                          "\n" + " ######## input data ######### " + \
                          "\n" + " ############################# " + "\n" + "\n"
                          )
            # 텐서를 NumPy 배열로 변환
            numpy_array = question.numpy()

            # NumPy 배열을 리스트로 변환
            list_from_numpy = numpy_array.tolist()

            logging.debug("\n" + "\n" + " ############################# " + \
                          "\n" + " ###### question data ######## " + \
                          "\n" + " ############################# " + \
                          "\n" + str(list_from_numpy) + "\n"
                          )
            # logging.debug("\n"+"\n"+str(np.array(question.type(torch.LongTensor)))+"\n")

            logging.debug("\n" + "\n" + " ############################# " + \
                          "\n" + " ###### response data ######## " + \
                          "\n" + " ############################# " + \
                          "\n" + str(np.array(response)) + "\n"
                          )

            # logging.debug("\n"+str(question)+"\n")
            predict, _ = model(question.long(), response.long(), question_shift.long())

            logging.debug("\n" + "\n" + " #############################  " + \
                          "\n" + " ###### predict data ######### " + \
                          "\n" + " #############################  " + \
                          "\n" + str(np.array(predict)) + "\n"
                          )

            predict = torch.masked_select(predict, masked).detach().cpu()
            true_score = torch.masked_select(response_shift, masked).detach().cpu()

            logging.debug("\n" + "\n" + " ################################ " + \
                          "\n" + " ###### true_score data ######### " + \
                          "\n" + " ################################ " + \
                          "\n" + str(np.array(true_score)) + "\n"
                          )

            accuracy = metrics.roc_auc_score(
                y_true=true_score.numpy(), y_score=predict.numpy()
            )

            logging.debug(
                "\n" + "\n" + " ################## " + \
                "\n" + " ##### Result ##### " + \
                "\n" + " ################## " + \
                "\n" + " Epoch: {},   AUC: {}".format(epoch, accuracy) + "\n"
            )


class RunModel:

    # Operation flow sequence 3-1.
    def __init__(
            self,
            model_name=Type[str],
            dataset_name=Type[str],
            date_info=None
    ) -> None:
        """
        Initialization function to initialize parameters.

        Args:
            model_name: The model name.
            dataset_name: The dataset name.
            date_info: datetime now.
        """
        self.dataset_name = dataset_name

        if self.dataset_name == "AI_HUB":
            self.dataset = AI_HUB()

        # Operation flow sequence 3-1-1.
        self.model = None
        self.date_info = date_info
        self.model_name = str(model_name)

        # Operation flow sequence 3-1-2.
        if self.dataset_name == "AI_HUB":
            config.set("train_config", "batch_size", str(self.dataset.length))
            config.set("train_config", "number_epochs", "1")
            config.set("train_config", "train_ratio", "0.8")
            config.set("train_config", "learning_rate", "0.001")
            config.set("train_config", "optimizer", "adam")
            config.set("train_config", "sequence_length", "50")
            config.set("sakt", "n", "50")
            config.set("sakt", "d", "5")
            config.set("sakt", "number_attention_heads", "5")
            config.set("sakt", "dropout", "0.5")

        self.model_config = dict(config.items(self.model_name))
        self.train_config = dict(config.items("train_config"))
        self.batch_size = int(self.train_config["batch_size"])
        self.number_epochs = int(self.train_config["number_epochs"])
        self.train_ratio = float(self.train_config["train_ratio"])
        self.learning_rate = float(self.train_config["learning_rate"])
        self.optimizer = self.train_config["optimizer"]
        self.sequence_length = int(self.train_config["sequence_length"])
        self.n = int(self.model_config["n"])
        self.d = int(self.model_config["d"])
        self.number_attention_heads = int(self.model_config["number_attention_heads"])
        self.dropout = float(self.model_config["dropout"])

        _, self.ckeckpoint_path = find_datasets_path(file_name="model.ckpt")

    # Operation flow sequence 3-2.
    def run_model(self):
        """
        Function to train the model.

        Args:
            Initialized parameters.
        """
        # Select model.
        # Operation flow sequence 3-2-1.

        self.model = SAKT(
            number_questions=int(1213),
            n=int(self.n),
            d=int(self.d),
            number_attention_heads=int(self.number_attention_heads),
            dropout=float(self.dropout)
        ).to(
            device
        )

        ckeckpoint = torch.load(self.ckeckpoint_path, map_location=device)

        # 연결
        self.model.load_state_dict(ckeckpoint)

        logging_sakt_config(number_epochs=self.number_epochs,
                            batch_size=self.batch_size,
                            optimizer=self.optimizer,
                            train_ratio=self.train_ratio,
                            learning_rate=self.learning_rate,
                            sequence_length=self.sequence_length,
                            d=self.d,
                            n=self.n,
                            dropout=self.dropout,
                            number_attention_heads=self.number_attention_heads,
                            model=self.model)

        logging.debug(
            "\n" + "\n" + " ############################" + \
            "\n" + " ### start evaluate model ###" + \
            "\n" + " ############################" + "\n"
        )

        start = time.time()

        for epoch in range(1, self.number_epochs + 1):

            train_loader = DataLoader(
                self.dataset,
                batch_size=int(self.dataset.length),
                generator=torch.Generator(device=device),
                shuffle=True,
                collate_fn=collate_fn
            )

            eval_model(
                model=self.model,
                loader=train_loader,
                epoch=epoch
            )

        end = time.time()
        sec = (end - start)
        result_list = str(datetime.timedelta(seconds=sec)).split(".")
        # 필요하면
        # logging.debug(" total train time : %s", result_list[0])
