from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
import torch

from model import PitVQANet
from dataloader import Pit24VQAClassification

from torch.utils.data import DataLoader
import torch.nn.functional as F

import os
import random
import numpy as np


def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def calc_classwise_acc(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)
    classwise_acc = matrix.diagonal()/matrix.sum(axis=1)
    return classwise_acc


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    seed_everything()

    # preapre model
    model_path = '/path/to/weights/pit_lr1e-5-Best.pth.tar'
    checkpoint = torch.load(model_path, map_location='cpu')
    model = checkpoint['model']
    model.to(device)

    # prepare validation set
    val_seq = ['02', '06', '12', '13', '24']
    folder_head = '/path/to/dataset/PitVis-24/QA/video_'
    folder_tail = '/*.txt'
    val_dataset = Pit24VQAClassification(val_seq, folder_head, folder_tail)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    # evaluation
    label_true = None
    label_pred = None
    model.eval()
    with torch.no_grad():
        for i, (file_name, images, questions, labels) in enumerate(val_dataloader, 0):
            labels = labels.to(device)
            outputs = model(image=images.to(device), question=questions)

            scores, predicted = torch.max(F.softmax(outputs, dim=1).data, 1)

            if label_true is None:
                label_true = labels.data.cpu()
            else:
                label_true = torch.cat((label_true, labels.data.cpu()), 0)
            if label_pred is None:
                label_pred = predicted.data.cpu()
            else:
                label_pred = torch.cat((label_pred, predicted.data.cpu()), 0)

        print(label_true[:50])
        print(label_pred[:50])
        score = balanced_accuracy_score(label_true.numpy(), label_pred.numpy())
        classwise_acc = calc_classwise_acc(label_true.numpy(), label_pred.numpy())
        print(f'balanced_accuracy_score: {score}')
        print(f'classwise_acc: {classwise_acc}')
