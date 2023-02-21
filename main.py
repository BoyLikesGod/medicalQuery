import os.path
import time

import torch
from gensim.models import KeyedVectors

from model import SemLSTM
from utils.arguments import DataTrainArguments, TrainingArguments, ModelArguments
from utils.dataProcess import QQRProcessor, QQRDataset, DataCollator
from utils.runing_utils import train, predict, generate_commit

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

data_collator = DataCollator()

# 创建输出结果（模型、参数、预测结果）的文件夹
model_name = f'semlstm-{str(int(time.time()))}'
training_args.output_dir = os.path.join(training_args.output_dir, model_name)
if not os.path.exists(training_args.output_dir):
    os.makedirs(training_args.output_dir, exist_ok=True)

# 初始化模型
print('模型初始化')
mod = SemLSTM(in_feat=model_args.in_feat, num_labels=len(processor.get_labels()), dropout_prob=model_args.dropout_prob, w2v_state_dict=w2v_model)
mod.to(training_args.device)

print('Training...')
best_steps, best_metric = train(args=training_args, model=mod, train_dataset=train_dataset, dev_dataset=eval_dataset, data_collator=data_collator)
print(f'训练结束，最好结果：第{best_steps}步 准确率{best_metric}')
best_model_path = os.path.join(training_args.output_dir, f'checkpoint-{best_steps}.pt')
mod = SemLSTM(in_feat=model_args.in_feat, num_labels=len(processor.get_labels()), dropout_prob=model_args.dropout_prob, w2v_state_dict=w2v_model)
mod.load_state_dict(torch.load(best_model_path, map_location='cpu'))
mod.to(training_args.device)

# 保存最佳模型及超参数
torch.save(mod.state_dict(), os.path.join(training_args.output_dir, 'python_model_bin'))
torch.save(training_args, os.path.join(training_args.output_dir, 'training_args_bin'))

# 预测及生成预测结果
preds = predict(training_args, mod, test_dataset, data_collator)
generate_commit(training_args.output_dir, processor.TASK, test_dataset, preds)