import dataclasses
import json
from dataclasses import field, dataclass


@dataclass
class DataTrainArguments:
    w2v_file: str = field(
        default='embedding/tencent-ailab-embedding-zh-d100-v0.2.0-s.txt',
        metadata={'help': '预训练词向量的目录'})
    data_dir: str = field(
        default='KUAKE-QQR',
        metadata={'help': '数据目录'})

    def __str__(self):
        self_as_dict = dataclasses.asdict(self)
        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"

@dataclass
class TrainingArguments:
    output_dir: str = field(
        default='output_data/',
        metadata={'help': '输出目录，模型的预测输出和检查点将输出到这个目录'}
    )
    train_batch_size: int = field(
        default=64,
        metadata={'help': '训练batch size'}
    )
    eval_batch_size: int = field(
        default=64,
        metadata={'help': '测试batch size'}
    )
    num_train_epochs: int = field(
        default=27,
        metadata={"help": "训练的总epochs数"}
    )
    learning_rate: float = field(
        default=0.001,
        metadata={'help': '"AdamW的初始learning rate'}
    )
    wight_decay: float = field(
        default=5e-4,
        metadata={"help": "AdamW的权重衰减"}
    )
    logging_steps: int = field(
        default=50,
        metadata={'help': '每更新X步记录一次状态'}
    )
    eval_steps: int = field(
        default=100,
        metadata={'help': '每X步进行一次评估'}
    )
    divce: str = field(
        default='cpu',
        metadata={"help": '设备'}
    )

    def __str__(self):
        self_as_dict = dataclasses.asdict(self)
        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)}"

    def to_json_string(self):
        """将此对象变成JSON字符串."""
        return json.dumps(dataclasses.asdict(self), indent=2)

@dataclass
class ModelArguments:
    in_feat: int = field(
        default=100,
        metadata={'help': "输入样本的维度"}
    )
    dropout_prob: float = field(
        default=0.1,
        metadata={'help': "Dropout概率"}
    )

    def __str__(self):
        self_as_dict = dataclasses.asdict(self)
        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)}"

    def to_json_string(self):
        """将此对象变成JSON字符串."""
        return json.dumps(dataclasses.asdict(self), indent=2)

