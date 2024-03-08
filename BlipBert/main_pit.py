
import os
import sys
import argparse
import random

import torch
from torch import nn
import torch.utils.data
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from model import BlipBert
from dataloader import PitVis24VQA
from BLIP.models.blip import create_vit, init_tokenizer, load_checkpoint
from utils import save_clf_checkpoint, adjust_learning_rate, calc_acc, calc_precision_recall_fscore, calc_classwise_acc

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_arg():
    parser = argparse.ArgumentParser(description='VisualQuestionAnswerClassification')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--workers', type=int, default=8, help='for data-loading')

    parser.add_argument('--lr', type=float, default=0.00002, help='learning rate')
    parser.add_argument('--checkpoint_dir', default='checkpoints/pit/lr2e-5_', help='path for weights')
    parser.add_argument('--question_len', default=25, help='25')
    parser.add_argument('--num_class', default=59, help='59')

    args = parser.parse_args()
    return args


def blip_vqa_custom(pretrained='', **kwargs):
    model = BlipBert(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
    return model


def train(train_dataloader, model, criterion, optimizer, epoch, device):
    model.train()
    total_loss = 0.0
    label_true = None
    label_pred = None

    for i, (_, images, questions, labels) in enumerate(train_dataloader, 0):
        # labels
        labels = labels.to(device)
        # forward
        outputs = model(image=images.to(device), question=questions)
        # compute loss
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        _, predicted = torch.max(F.softmax(outputs, dim=1).data, 1)
        if label_true is None:  # accumulate true labels of the entire training set
            label_true = labels.data.cpu()
        else:
            label_true = torch.cat((label_true, labels.data.cpu()), 0)
        if label_pred is None:  # # accumulate pred labels of the entire training set
            label_pred = predicted.data.cpu()
        else:
            label_pred = torch.cat((label_pred, predicted.data.cpu()), 0)

    # compute metrics
    acc, c_acc = calc_acc(label_true, label_pred), calc_classwise_acc(label_true, label_pred)
    precision, recall, f_score = calc_precision_recall_fscore(label_true, label_pred)
    print(f'Epoch: {epoch}, train loss: {total_loss} | train acc: {acc} | '
          f'precision: {precision} | recall: {recall} | F1: {f_score}')
    return acc


def validate(val_loader, model, criterion, epoch, device):
    model.eval()
    total_loss = 0.0
    label_true = None
    label_pred = None

    with torch.no_grad():
        for i, (_, images, questions, labels) in enumerate(val_loader, 0):
            # label
            labels = labels.to(device)
            # forward
            outputs = model(image=images.to(device), question=questions)
            # loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(F.softmax(outputs, dim=1).data, 1)
            label_true = labels.data.cpu() if label_true is None else torch.cat((label_true, labels.data.cpu()), 0)
            label_pred = predicted.data.cpu() if label_pred is None else torch.cat((label_pred, predicted.data.cpu()), 0)

    # compute metrics
    c_acc = 0.0
    acc = calc_acc(label_true, label_pred)
    precision, recall, f_score = calc_precision_recall_fscore(label_true, label_pred)
    print(f'Epoch: {epoch}, test loss: {total_loss} | test acc: {acc} | '
          f'test precision: {precision} | test recall: {recall} | test F1: {f_score}')
    return acc, c_acc, precision, recall, f_score


if __name__ == '__main__':

    args = get_arg()
    os.makedirs('checkpoints/pit', exist_ok=True)

    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start_epoch = 1
    best_epoch = [0]
    best_results = [0.0]
    epochs_since_improvement = 0

    # data location
    train_seq = ['01', '03', '04', '05', '07', '08', '09', '10', '11', '14',
                 '15', '16', '17', '18', '19', '20', '21', '22', '23', '25']
    val_seq = ['02', '06', '12', '13', '24']
    folder_head = '/path/to/PitVis-24/QA/video_'  # change to your path
    folder_tail = '/*.txt'

    # dataloader
    train_dataset = PitVis24VQA(train_seq, folder_head, folder_tail)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_dataset = PitVis24VQA(val_seq, folder_head, folder_tail)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    model_url = \
        'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth'
    image_size = 224
    model = blip_vqa_custom(pretrained=model_url, num_class=args.num_class, image_size=image_size, vit='base')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    print('Start training.')
    for epoch in range(start_epoch, args.epochs+1):
        if epochs_since_improvement > 0 and epochs_since_improvement % 5 == 0:
            adjust_learning_rate(optimizer, 0.8)

        # train
        train_acc = train(train_dataloader=train_dataloader, model=model, criterion=criterion,
                          optimizer=optimizer, epoch=epoch, device=device)
        # validation
        test_acc, test_c_acc, test_precision, test_recall, test_f_score \
            = validate(val_loader=val_dataloader, model=model, criterion=criterion, epoch=epoch, device=device)

        if test_acc >= best_results[0]:
            print('Best Epoch:', epoch)
            epochs_since_improvement = 0
            best_results[0] = test_acc
            best_epoch[0] = epoch
            save_clf_checkpoint(args.checkpoint_dir, epoch, epochs_since_improvement,
                                model, optimizer, best_results[0], final_args=None)
    print('End training.')
