import json
import math
import os
from typing import Union, Any, Mapping, List

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import numpy as np

from utils.arguments import TrainingArguments


def create_optimizer_and_scheduler(args: TrainingArguments, model: nn.Module):
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2, eta_min=1e-5)

    return optimizer, lr_scheduler


def _prepare_input(data: Union[torch.Tensor, Any], device: str = 'cuda'):
    if isinstance(data, Mapping):
        return type(data)({k: _prepare_input(v, device) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(_prepare_input(v, device) for v in data)
    elif isinstance(data, torch.Tensor):
        kwargs = dict(device=device)
        return data.to(**kwargs)
    return data


def train(args: TrainingArguments, model: nn.Module, train_dataset, dev_dataset, data_collator):
    # 初始化dataloader
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.train_batch_size, shuffle=True,
                                  collate_fn=data_collator)
    dev_dataloader = DataLoader(dataset=dev_dataset, batch_size=args.eval_batch_size, shuffle=True,
                                collate_fn=data_collator)

    num_examples = len(train_dataloader.dataset)
    num_update_steps_per_epoch = len(train_dataloader)
    num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)

    max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
    num_train_epochs = math.ceil(args.num_train_epochs)
    num_train_samples = len(train_dataset) * args.num_train_epochs

    # 优化器、scheduler
    optimizer, lr_scheduler = create_optimizer_and_scheduler(args, model)

    print("***** Running training *****")
    print(f" 样本总量 = {num_examples}")
    print(f" Epochs总数量 = {args.num_train_epochs}")
    print(f" 训练的batch大小 = {args.train_batch_size}")
    print(f" 最大优化步 = {max_steps}")

    model.zero_grad()
    model.train()
    global_step = 0
    best_metric = 0.0
    best_steps = -1

    for epoch in range(num_train_epochs):
        for step, item in enumerate(train_dataloader):
            inputs = _prepare_input(item, device=args.device)
            outputs = model(**inputs)
            loss = outputs[0]

            loss.backward()
            optimizer.step()
            lr_scheduler.step(epoch + step / num_update_steps_per_epoch)

            model.zero_grad()
            global_step += 1

            if global_step % args.logging_steps == 0:
                print(f'Training: Epoch {epoch + 1}/{num_train_epochs} - Step {(step + 1)} - Loss {loss}')

            if global_step % args.eval_steps == 0:
                loss, acc = evaluate(args, model, dev_dataloader)
                print(
                    f'Evaluation: Epoch {epoch + 1}/{num_train_epochs} - Step {(global_step + 1)} - Loss {loss} - Accuracy {acc}')

                if acc > best_metric:
                    best_metric = acc
                    best_steps = global_step

                    saved_path = os.path.join(args.output_dir, f'checkpoint-{best_steps}.pt')
                    torch.save(model.state_dict(), saved_path)

    return best_steps, best_metric


def evaluate(args: TrainingArguments, model: nn.Module, eval_dataloader):
    model.eval()
    loss_list, preds_list, labels_list = [], [], []

    for item in eval_dataloader:
        inputs = _prepare_input(item, device=args.device)
        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs[0]
            loss_list.append(loss.detach().cpu().item())

            preds = torch.argmax(outputs[1].cpu(), dim=-1).numpy()
            preds_list.append(preds)

            labels_list.append(inputs['labels'].cpu().numpy())

    preds = np.concatenate(preds_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    loss = np.mean(loss_list)
    accuracy = simple_accuracy(preds, labels)

    model.train()
    return loss, accuracy


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def predict(args: TrainingArguments, model: nn.Module, test_dataset, data_collator):
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.eval_batch_size, shuffle=False,
                                 collate_fn=data_collator)
    print(" ***** 预测 ***** ")
    model.eval()
    preds_list = []

    for item in test_dataloader:
        inputs = _prepare_input(item, device=args.device)
        with torch.no_grad():
            outputs = model(**inputs)
            preds = torch.argmax(outputs.cpu(), dim=-1).numpy()
            preds_list.append(preds)

    print(f'预测结束')
    preds = np.concatenate(preds_list, axis=0).tolist()
    model.train()
    return preds

def generate_commit(output_dir, task_name, test_dataset, preds: List[int]):
    test_examples = test_dataset.examples
    pred_test_examples = []
    for idx in range(len(test_examples)):
        example = test_examples[idx]
        label = test_dataset.id2label[preds[idx]]
        pred_example = {'id': example.guid, 'qurry1': example.text_a, 'query2': example.text_b, 'label': label}
        pred_test_examples.append(pred_example)
    with open(os.path.join(output_dir, f'{task_name}_test.json'), 'w', encoding='utf-8') as f:
        json.dump(pred_test_examples, f, indent=2, ensure_ascii=False)
