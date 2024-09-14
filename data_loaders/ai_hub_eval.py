import pandas as pd
from torch.utils.data import Dataset
import logging

# 패키지 초기화 함수 불러오기
from __init__ import find_datasets_path


class AI_HUB(Dataset):

    def __init__(self) -> None:
        super().__init__()

        # datasets_path 가져오기
        self.dataset_dir, self.dataset_path = \
            find_datasets_path(file_name="ai_test_data")

        self.question_sequences, self.response_sequences, = self.preprocess()
        self.length = len(self.question_sequences)

    def __getitem__(self, index):
        return self.question_sequences[index], self.response_sequences[index]

    def __len__(self):
        return self.length

    def preprocess(self):
        df = pd.read_csv(self.dataset_path)

        logging.debug(
            "\n" + "\n" + " ########################" + \
            "\n" + " ### start preprocess ###" + \
            "\n" + " ########################" + "\n" + \
            "\n" + " Number of Data : {}".format(str(len(df)))+"\n"
        )

        question_sequences = []
        response_sequences = []

        for index in df.index:
            question_first = df.iloc[index]["question_sequences"].replace("[", "").replace("]", "").replace(".", "")
            question_first = str(question_first).split()
            question_list = [int(number) for number in question_first]

            response_first = df.iloc[index]["response_sequences"].replace("'", "").replace("[", "").replace("]",
                                                                                                            "").replace(
                "\n", "").replace(".", "")
            response_list = [int(number) for number in response_first.split()]

            question_sequences.append(question_list)
            response_sequences.append(response_list)

        return question_sequences, response_sequences
