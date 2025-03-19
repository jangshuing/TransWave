import os
import argparse
from torch.cuda.amp import GradScaler
import torch
import torch.optim as optim
from torchvision import transforms
import torchvision.transforms.functional as F
import pandas as pd
from ptflops import get_model_complexity_info
from utils import read_split_data, train_one_epoch, evaluate
from my_dataset import MyDataSet
from model.TransWave import transwave as create_model

# Initialize to 0 or None for comparison later
best_val_acc_weighted = 0
best_val_f1_weighted = 0
best_val_auc_weighted = 0
best_val_kappa = 0
best_val_mcc = 0

best_val_acc_weighted_path = None
best_val_f1_weighted_path = None
best_val_auc_weighted_path = None
best_val_kappa_path = None
best_val_mcc_path = None

def main(args):
    global best_val_acc_weighted, best_val_f1_weighted, best_val_auc_weighted, best_val_kappa, best_val_mcc
    global best_val_acc_weighted_path, best_val_f1_weighted_path, best_val_auc_weighted_path, best_val_kappa_path, best_val_mcc_path

    save_path = args.savepath if args.savepath else "./models"
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_images_path, train_images_label, val_images_path, val_images_label, test_images_path, test_images_label = read_split_data(
        args.data_path)

    img_size = 224
    # augmented dataset
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.1),  # Random horizontal flip
            transforms.RandomVerticalFlip(p=0.1),  # Random vertical flip
            transforms.RandomRotation(10),  # Random rotation
            transforms.ColorJitter(contrast=0.2),  # Random contrast adjustment
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),  # Random Gaussian blur
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [
                                 0.229, 0.224, 0.225])  # Normalize the image
        ]),
        "val": transforms.Compose([
            transforms.Resize(int(img_size * 1.143)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # Create MyDataSet instances with augmented dataset
    train_dataset = MyDataSet(
        images_path=train_images_path,
        images_class=train_images_label,
        transform=data_transform["train"]
    )

    val_dataset = MyDataSet(
        images_path=val_images_path,
        images_class=val_images_label,
        transform=data_transform["val"]
    )

    test_dataset = MyDataSet(
        images_path=test_images_path,
        images_class=test_images_label,
        transform=data_transform["val"]
    )

    num_labels_train = {}
    num_labels_val = {}
    num_labels_test = {}
    for label in range(3):  # Assuming there are 12 classes
        num_labels_train[label] = train_dataset.images_class.count(label)
        num_labels_val[label] = val_dataset.images_class.count(label)
        num_labels_test[label] = test_dataset.images_class.count(label)
        print(f"Training, validation, and testing dataset class {label + 1} class counts: {num_labels_train[label], num_labels_val[label], num_labels_test[label]}")

    batch_size = args.batch_size
    # Number of workers
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print(f'Using {nw} dataloader workers per process')

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=nw,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=nw,
    )
    
    test_loader = torch.utils.data.DataLoader(  # Newly added test data loader
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=nw,
    )

    model = create_model(num_classes=args.num_classes).to(device)

    # Calculate parameters
    input_size1 = (3, img_size, img_size)  # Assuming input is RGB image
    macs, params = get_model_complexity_info(model, input_size1, as_strings=False)
    print(macs, params)

    if args.weights != "":
        assert os.path.exists(
            args.weights), f"weights file: '{args.weights}' does not exist."
        weights_dict = torch.load(
            args.weights, map_location=device)  # crossvit transnext
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # Freeze all weights except the head
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print(f"Training {name}")

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5E-2)
    # Initialize GradScaler for mixed precision training
    scaler = GradScaler()
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    # Training + Validation
    # Initialize an empty DataFrame to store reports for each epoch
    all_val_reports_df = pd.DataFrame()
    for epoch in range(args.epochs):
        # Train one epoch
        train_loss, train_acc_weighted, train_f1_weighted, train_auc_weighted, train_kappa, train_mcc = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch
        )
        
        # Evaluate on validation set
        val_loss, val_acc_weighted,  val_f1_weighted, val_auc_weighted, val_kappa, val_mcc, val_report_df = evaluate(
            model=model,
            data_loader=val_loader,
            device=device,
            epoch=epoch
        )

        # Update learning rate
        scheduler.step(val_loss)

        new_directory = f"{save_path}/weight"
        if not os.path.exists(new_directory):
            os.makedirs(new_directory)
        # Save the best performing model
        if val_acc_weighted > best_val_acc_weighted:
            best_val_acc_weighted = val_acc_weighted
            best_val_acc_weighted_path = f"{new_directory}/best_val_acc_weighted.pth"
            torch.save(model.state_dict(), best_val_acc_weighted_path)
            print(f"New best val_acc_weighted model saved at epoch {epoch} with validation best_val_acc_weighted {best_val_acc_weighted:.5f}")

        if val_f1_weighted > best_val_f1_weighted:
            best_val_f1_weighted = val_f1_weighted
            best_val_f1_weighted_path = f"{new_directory}/best_val_f1_weighted.pth"
            torch.save(model.state_dict(), best_val_f1_weighted_path)
            print(f"New best val_f1_weighted model saved at epoch {epoch} with validation best_val_f1_weighted {best_val_f1_weighted:.5f}")

        if val_auc_weighted > best_val_auc_weighted:
            best_val_auc_weighted = val_auc_weighted
            best_val_auc_weighted_path = f"{new_directory}/best_val_auc_weighted.pth"
            torch.save(model.state_dict(), best_val_auc_weighted_path)
            print(f"New best val_auc_weighted model saved at epoch {epoch} with validation best_val_auc_weighted {best_val_auc_weighted:.5f}")

        if val_kappa > best_val_kappa:
            best_val_kappa = val_kappa
            best_val_kappa_path = f"{new_directory}/best_val_kappa.pth"
            torch.save(model.state_dict(), best_val_kappa_path)
            print(f"New best val_kappa model saved at epoch {epoch} with validation best_val_kappa {best_val_kappa:.5f}")
            
        if val_mcc > best_val_mcc:
            best_val_mcc = val_mcc
            best_val_mcc_path = f"{new_directory}/best_val_mcc.pth"
            torch.save(model.state_dict(), best_val_mcc_path)
            print(f"New best val_mcc model saved at epoch {epoch} with validation best_val_mcc {best_val_mcc:.5f}")

    print(f"Best ACC model saved at: {best_val_acc_weighted_path} with validation accuracy: {best_val_acc_weighted}")
    print(f"Best F1 model saved at: {best_val_f1_weighted_path} with validation F1 score: {best_val_f1_weighted}")
    print(f"Best AUC model saved at: {best_val_auc_weighted_path} with validation AUC: {best_val_auc_weighted}")
    print(f"Best Kappa model saved at: {best_val_kappa_path} with validation Kappa: {best_val_kappa}")
    print(f"Best MCC model saved at: {best_val_mcc_path} with validation MCC: {best_val_mcc}")


    # Test (Test using the best models based on different metrics)
    # Dictionary of model paths, key is the metric, value is the model path
    model_paths = {
        "Best Accuracy Model": best_val_acc_weighted_path,
        "Best F1 Model": best_val_f1_weighted_path,
        "Best AUC Model": best_val_auc_weighted_path,
        "Best Kappa Model": best_val_kappa_path,
        "Best MCC Model": best_val_mcc_path
    }

    # Initialize variables to store the best test metrics
    best_test_acc_weighted = 0
    best_test_f1_weighted = 0
    best_test_auc_weighted = 0
    best_test_kappa = 0
    best_test_mcc = 0
    best_test_loss = 100

    global best_metrics

    best_metrics = {
        "best_test_acc_weighted": best_test_acc_weighted,
        "best_test_f1_weighted": best_test_f1_weighted,
        "best_test_auc_weighted": best_test_auc_weighted,
        "best_test_kappa": best_test_kappa,
        "best_test_mcc": best_test_mcc,
        "best_test_loss": best_test_loss,
    }

    # Loop through the models and evaluate
    for model_name, model_path in model_paths.items():
        print(f"testing on test set with {model_name}...")

        # Test instantiate model
        model = create_model(num_classes=args.num_classes)  # Replace with your model class

        # Load the model's state dictionary
        model.load_state_dict(torch.load(model_path))

        # Move the model to device (CPU or GPU)
        model.to(device)

        # Evaluate the model
        test_loss, test_acc_weighted, test_f1_weighted, test_auc_weighted, test_kappa, test_mcc, test_report_df = evaluate(
            model=model,
            data_loader=test_loader,
            device=device,
            epoch=epoch
        )

        # Keep only the best metrics
        if test_acc_weighted > best_metrics["best_test_acc_weighted"]:
            best_metrics["best_test_acc_weighted"] = test_acc_weighted

        if test_f1_weighted > best_metrics["best_test_f1_weighted"]:
            best_metrics["best_test_f1_weighted"] = test_f1_weighted

        if test_auc_weighted > best_metrics["best_test_auc_weighted"]:
            best_metrics["best_test_auc_weighted"] = test_auc_weighted

        if test_kappa > best_metrics["best_test_kappa"]:
            best_metrics["best_test_kappa"] = test_kappa

        if test_mcc > best_metrics["best_test_mcc"]:
            best_metrics["best_test_mcc"] = test_mcc

        if test_loss < best_metrics["best_test_loss"]:
            best_metrics["best_test_loss"] = test_loss

    # Output the best metrics
    print("Best test metrics.......", best_metrics)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--data-path', type=str,
                        default="", help="Path to the dataset directory")
    parser.add_argument('--weights', type=str, default="",
                        help='initial weights path')
    parser.add_argument('--savepath', type=str, default="",
                        help='initial save path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0',
                        help='device id (i.e. 0 or 0,1 or cpu)')
    opt = parser.parse_args()
    main(opt)
