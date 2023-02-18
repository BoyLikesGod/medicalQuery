import time

from gensim.models import KeyedVectors
from utils.arguments import DataTrainArguments, TrainingArguments, ModelArguments
from utils.dataProcess import QQRProcessor, QQRDataset

data_args = DataTrainArguments()  # 数据集参数
training_args = TrainingArguments()
model_args = ModelArguments()

print('参数信息', data_args, training_args, model_args, sep='\n--->\n')

w2v_model = KeyedVectors.load_word2vec_format(data_args.w2v_file, binary=False)

processor = QQRProcessor(data_dir=data_args.data_dir)

train_dataset = QQRDataset(
    processor.get_train_examples(),
    processor.get_labels(),
    vocab_mapping=w2v_model.key_to_index,
    max_length=32
)
eval_dataset = QQRDataset(
    processor.get_dev_examples(),
    processor.get_labels(),
    vocab_mapping=w2v_model.key_to_index,
    max_length=32
)
test_dataset = QQRDataset(
    processor.get_test_examples(),
    processor.get_labels(),
    vocab_mapping=w2v_model.key_to_index,
    max_length=32
)
