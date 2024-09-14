import glob
import os
import pickle
import logging


def make_check_points(model_name=str, dataset_name=str, date_info=str):
    """Function to make check point folder.

    Args:
        model_name: Model name.
        dataset_name: Dataset name.
        date_info: datetime now.

    Returns:
        directory.

    Note:
        Return detail.

        - check_points_model_data_datetime_path: Check point directory.

    Examples:
        >>> make_check_points(model_name=str, dataset_name=str, date_info=str)
        "check_points/sakt/ASSISTment2009/skill_builder_data.csv"
    """

    logging.debug("model_name :" + str(model_name))
    logging.debug("dataset_name :" + str(dataset_name))
    logging.debug("datetime :" + date_info.strftime("%y%m%d_%H_%M_%S"))

    # check_points 폴더 존재 여부
    if not os.path.isdir("check_points"):
        os.mkdir("check_points")
    # check_points/model 폴더 존재 여부
    check_points_model_path = str(os.path.join("check_points", str(model_name)))
    if not os.path.isdir(check_points_model_path):
        os.mkdir(check_points_model_path)
    # check_points/model/dataset 폴더 존재 여부
    check_points_model_data_path = os.path.join(check_points_model_path, str(dataset_name))
    if not os.path.isdir(check_points_model_data_path):
        os.mkdir(check_points_model_data_path)
    # check_points/model/dataset/datetime 폴더 존재 여부
    check_points_model_data_datetime_path = os.path.join(check_points_model_data_path,
                                                         date_info.strftime("%y%m%d_%H_%M_%S"))
    if not os.path.isdir(check_points_model_data_datetime_path):
        os.mkdir(check_points_model_data_datetime_path)
    return check_points_model_data_datetime_path


# Operation flow sequence 15.
def find_datasets_path(file_name=str):
    """Function to find dataset path.

    Args:
        file_name: The file name you are looking for.

    Returns:
        directories.

    Note:
        Return detail.

        - dataset_directory: Folder path.

        - dataset_path: Dataset path.

    Examples:
        >>> find_datasets_path(file_name=str)
        ("kt/datasets/ASSISTment2009", "kt/datasets/ASSISTment2009/skill_builder_data.csv")
    """
    join_dataset_name = "**/" + str(file_name) + "*"
   
    # 인자로 받은 패턴과 이름이 일치하는 모든 파일과 디렉토리의 리스트를 반환
    # recursive=True로 설정하고 "**"를 사용하면 모든 하위 디렉토리까지 탐색한다.
    if len(glob.glob(join_dataset_name, recursive=True)) > 1:
        dataset_path = glob.glob(join_dataset_name, recursive=True)[1]
    if len(glob.glob(join_dataset_name, recursive=True)) == 1:
        dataset_path = glob.glob(join_dataset_name, recursive=True)[0]
    last_index = dataset_path.rfind("/")    
    dataset_directory = dataset_path[:last_index+1]
    return dataset_directory, dataset_path


def get_pkl(
        dataset_directory=str
):
    """Function to import pkl file.

    Args:
        dataset_directory: File path.

    Returns:
        parameters.

    Note:
        Return detail.

        - question_sequences: A bundle of questions sliced according to the set sequence.

        - response_sequence: A bundle of question responses sliced according to the set sequence.

        - question_list: Question dictionary.

        - user_list: User list.

        - question_to_index: Question index.

        - user_to_index: User index.

    Examples:
        >>> get_pkl(dataset_directory=str)
        ([[67 18 67 18 67 18 18 67 67 -1 -1 -1 ... -1 -1 -1]...],
         [[0 0 1 1 0 0 0 0 0 0 0 0 0 0 1 1 -1 -1 -1 ... -1 -1 -1]...],
         ["Absolute Value", "Addition Whole Numbers", "Addition and Subtraction Fractions"...],
         [14 21825 51950 ... 96297 96298 96299],
         {"Absolute Value": 0, "Addition Whole Numbers": 1...},
         {14: 0, 21825: 1, 51950: 2, 52613: 3, 53167: 4...})
    """
    dataset_directory = str(dataset_directory)
    with open(os.path.join(dataset_directory, "question_sequences_eval.pkl"), "rb") as file:
        question_sequences = pickle.load(file)
    with open(os.path.join(dataset_directory, "response_sequences_eval.pkl"), "rb") as file:
        response_sequence = pickle.load(file)
    with open(os.path.join(dataset_directory, "question_list_eval.pkl"), "rb") as file:
        question_list = pickle.load(file)
    with open(os.path.join(dataset_directory, "user_list_eval.pkl"), "rb") as file:
        user_list = pickle.load(file)
    with open(os.path.join(dataset_directory, "question_to_index_eval.pkl"), "rb") as file:
        question_to_index = pickle.load(file)
    with open(os.path.join(dataset_directory, "user_to_index_eval.pkl"), "rb") as file:
        user_to_index = pickle.load(file)

    return question_sequences, response_sequence, question_list, user_list, question_to_index, user_to_index


def set_train_config(
        config=object,
        set_batch_size=None or int,
        set_number_epochs=None or int,
        set_train_ratio=None or float,
        set_learning_rate=None or float,
        set_optimizerimizer=None or str,
        set_sequence_length=None or int
):
    """Function that sets the train config value.

    Args:
        config: config.ini.
        set_batch_size: The size of the batch size.
        set_number_epochs: Number of epochs.
        set_train_ratio: Model training data rate.
        set_learning_rate: Model learning rate.
        set_optimizerimizer: optimizerimization name.
        set_sequence_length: Length of input data.

    Returns:
        set parameters.

    Note:
        Return detail.

        - batch_size: The size of the batch size.

        - number_epochs: Number of epochs.

        - train_ratio: Model training data rate.

        - learning_rate: Model learning rate.

        - optimizerimizer: optimizerimization name.

        - sequence_length: Length of input data.

    Examples:
        >>> set_train_config(config=config,
        >>>                  set_batch_size=int,
        >>>                  set_number_epochs=int,
        >>>                  set_train_ratio=float,
        >>>                  set_learning_rate=float,
        >>>                  set_optimizerimizer=str,
        >>>                  set_sequence_length=int)
        (256, 100, 0.2, 0.001, "adam", 50)
    """

    # config setting RawConfigParser.
    train_config = dict(config.items("train_config"))

    # train_config.
    # setting batch_size.
    if set_batch_size is None:
        batch_size = int(train_config["batch_size"])
    else:
        batch_size = set_batch_size
    # setting number_epochs.
    if set_number_epochs is None:
        number_epochs = int(train_config["number_epochs"])
    else:
        number_epochs = set_number_epochs
    # setting train_ratio.
    if set_train_ratio is None:
        train_ratio = float(train_config["train_ratio"])
    else:
        train_ratio = set_train_ratio
    # setting learning_rate.
    if set_learning_rate is None:
        learning_rate = float(train_config["learning_rate"])
    else:
        learning_rate = set_learning_rate
    # setting optimizerimizer.
    if set_optimizerimizer is None:
        optimizerimizer = train_config["optimizerimizer"]
    else:
        optimizerimizer = str(set_optimizerimizer)
    # setting sequence_length.
    if set_sequence_length is None:
        sequence_length = train_config["sequence_length"]
    else:
        sequence_length = int(set_sequence_length)

    return batch_size, number_epochs, train_ratio, learning_rate, optimizerimizer, sequence_length


def set_sakt_config(
        model_config=dict,
        set_d=None or int,
        set_n=None or int,
        set_number_attention_heads=None or int,
        set_dropout=None or float
):
    """Function that sets sakt config values.

    Args:
        model_config: model_config.
        set_d: Number of d.
        set_n: Number of n.
        set_number_attention_heads: Number of attention heads.
        set_dropout: Number of dropout.

    Returns:
        set parameters

    Note:
        Return detail.

        - d: Number of d.

        - n: Number of n.

        - number_attention_heads: Number of attention heads.

        - dropout: Number of dropout.

    Examples:
        >>> set_sakt_config(model_config=dict, set_d=int, set_n=int, set_number_attention_heads=int, set_dropout=float)
        (12, 12, 2, 0.2)
    """
    if set_d is None:
        d = int(str(model_config["d"]))
    else:
        d = int(set_d)
    if set_n is None:
        n = int(str(model_config["n"]))
    else:
        n = int(set_n)
    if set_number_attention_heads is None:
        number_attention_heads = int(str(model_config["number_attention_heads"]))
    else:
        number_attention_heads = int(str(set_number_attention_heads))
    if set_dropout is None:
        dropout = float(str(model_config["dropout"]))
    else:
        dropout = float(str(set_dropout))
    return d, n, number_attention_heads, dropout
