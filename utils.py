import os
import sys
import json
import pickle
import random
from sklearn.metrics import classification_report
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, cohen_kappa_score, matthews_corrcoef
import matplotlib.pyplot as plt
import os
import random
import json
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast
import numpy as np
scaler = GradScaler()
def read_split_data(root: str, val_rate: float = 0.2, test_rate: float = 0.1):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(
        root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    class_num = [cla for cla in os.listdir(
        root) if os.path.isdir(os.path.join(root, cla))]
    class_num.sort()  # 排序，保证各平台顺序一致
    class_indices = dict((k, v) for v, k in enumerate(class_num))
    json_str = json.dumps(dict((val, key)
                               for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path, train_images_label = [], []
    val_images_path, val_images_label = [], []
    test_images_path, test_images_label = [], []  # 新增加的测试集路径和标签列表
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG",
                 ".jpeg", ".JPEG", ".bmp"]  # 支持的文件后缀类型

    for cla in class_num:
        cla_path = os.path.join(root, cla)
        images = [os.path.join(cla_path, i) for i in os.listdir(
            cla_path) if os.path.splitext(i)[-1] in supported]
        images.sort()

        # 分配测试集
        test_size = int(len(images) * test_rate)
        test_subset = random.sample(images, k=test_size)

        # 从剩余图片中分配验证集
        remain_images = list(set(images) - set(test_subset))
        val_size = int(len(remain_images) * val_rate)
        val_subset = random.sample(remain_images, k=val_size)

        # 剩余的为训练集
        train_subset = list(set(remain_images) - set(val_subset))

        for img_path in images:
            if img_path in test_subset:
                test_images_path.append(img_path)
                test_images_label.append(class_indices[cla])
            elif img_path in val_subset:
                val_images_path.append(img_path)
                val_images_label.append(class_indices[cla])
            else:
                train_images_path.append(img_path)
                train_images_label.append(class_indices[cla])

    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    print("{} images for testing.".format(len(test_images_path)))  # 新增加打印测试集信息
    assert len(
        train_images_path) > 0, "number of training images must be greater than 0."
    assert len(
        val_images_path) > 0, "number of validation images must be greater than 0."
    assert len(
        test_images_path) > 0, "number of test images must be greater than 0."

    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(class_num)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(class_num)), class_num)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label, test_images_path, test_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list
    
def check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_max = param.grad.abs().max()
            grad_min = param.grad.abs().min()
            print(f"{name} -- Max Grad: {grad_max:.6f}, Min Grad: {grad_min:.6f}")
            if torch.isnan(grad_max) or grad_max > 1e5:  # 设定阈值，检测异常梯度
                print(f"Gradient explosion detected in {name}")
        else:
            print(f"{name} -- No gradient")

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # Cumulative loss
    accu_num = torch.zeros(1).to(device)   # Cumulative correct predictions
    optimizer.zero_grad()

    sample_num = 0
    all_preds = []
    all_labels = []
    all_probs = []  # For AUC calculation

    data_loader = tqdm(data_loader, file=sys.stdout)
    
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        if isinstance(pred, list):
            pred = torch.stack(pred, dim=0)  # [2, 16, 8]
            pred, _ = torch.max(pred, dim=0)  # [16, 8]

        pred_classes = torch.max(pred, dim=1)[1]  # Get the class with the max probability

        # Collect all predictions and true labels for evaluation
        all_preds.extend(pred_classes.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Collect probabilities for AUC calculation
        all_probs.extend(pred.softmax(dim=1).detach().cpu().numpy())  # Detach before converting to numpy

        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.item() * images.size(0)  # Scale the loss by batch size

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

        # Update progress bar with current epoch loss and accuracy
        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch, accu_loss.item() / sample_num, accu_num.item() / sample_num)

    # Weighted  F1
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')

    # Weighted accuracy (using class-wise accuracy)
    class_accuracy = precision_score(all_labels, all_preds, average=None)
    class_weights = np.bincount(all_labels)
    acc_weighted = np.sum(class_accuracy * class_weights) / np.sum(class_weights)

    # Compute AUC (average weighted AUC for multi-class classification)
    auc_weighted = roc_auc_score(all_labels, all_probs, average='weighted', multi_class='ovr')

    # Compute Kappa
    kappa = cohen_kappa_score(all_labels, all_preds)

    # Compute MCC
    mcc = matthews_corrcoef(all_labels, all_preds)

    # Print all evaluation metrics
    print(f"Epoch {epoch} -  Weighted Accuracy: {acc_weighted:.3f}, Weighted F1: {f1_weighted:.3f}, AUC: {auc_weighted:.3f}, Kappa: {kappa:.3f}, MCC: {mcc:.3f}")

    return accu_loss.item() / sample_num, acc_weighted, f1_weighted, auc_weighted, kappa, mcc



@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    model.eval()
    loss_function = torch.nn.CrossEntropyLoss()

    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)

    all_preds = []
    all_labels = []
    all_probs = []  # For AUC calculation

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        if isinstance(pred, list):
            pred = torch.stack(pred, dim=0)  # [2, 16, 8]
            pred, _ = torch.max(pred, dim=0)  # [16, 8]
        loss = loss_function(pred, labels.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        accu_loss += loss.item() * images.size(0)  # Scale loss by batch size

        # Accumulate predictions and true labels for further evaluation
        all_preds.extend(pred_classes.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Collect probabilities for AUC calculation
        all_probs.extend(pred.softmax(dim=1).cpu().numpy())  # Softmax for multi-class AUC

    # Calculate weighted F1
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')

    # Calculate weighted accuracy
    class_accuracy = precision_score(all_labels, all_preds, average=None)
    class_weights = np.bincount(all_labels)
    acc_weighted = np.sum(class_accuracy * class_weights) / np.sum(class_weights)

    # Compute AUC (average weighted AUC for multi-class classification)
    auc_weighted = roc_auc_score(all_labels, all_probs, average='weighted', multi_class='ovr')

    # Compute Kappa
    kappa = cohen_kappa_score(all_labels, all_preds)

    # Compute MCC
    mcc = matthews_corrcoef(all_labels, all_preds)

    # Print the evaluation metrics
    print(f"Epoch {epoch} -Weighted Accuracy: {acc_weighted:.3f}, Weighted F1: {f1_weighted:.3f},  AUC: {auc_weighted:.3f}, Kappa: {kappa:.3f}, MCC: {mcc:.3f}")
    
    # Generate and print classification report for each class
    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=1)
    
    # Convert the classification report to a DataFrame and return it for saving
    report_df = pd.DataFrame(report).transpose()

    return accu_loss.item() / sample_num, acc_weighted, f1_weighted, auc_weighted, kappa, mcc, report_df
