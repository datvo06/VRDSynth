import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pickle as pkl
from tqdm import tqdm
from train_gnn.models import MPNNModel
from layout_extraction.gnn_utils import calc_feature_size
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import numpy as np

def k_fold_split(dataset, k):
    """
    Splits the dataset into k-fold.
    
    Args:
    - dataset: A list or array of PyTorch Geometric Data objects.
    - k: Number of splits.
    
    Returns:
    - splits: A list of (train_idx, test_idx) pairs for each fold.
    """
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    splits = []
    
    for train_idx, test_idx in kfold.split(dataset):
        splits.append((train_idx, test_idx))
    
    return splits

def train(model, dataset, criterion, optimizer, device):
    model.train()
    total_loss = 0
    bar = tqdm(enumerate(dataset))
    avg_loss = 0
    avg_acc = 0
    for i, (data, _) in bar:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[0], data.y)
        acc = (out[0].argmax(dim=1) == data.y).sum().item() / data.y.shape[0]
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        avg_loss = (avg_loss * i + loss.item()) / (i + 1)
        avg_acc = (avg_acc * i + acc) / (i + 1)
        bar.set_description(f"Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")
        bar.set_description(f"Loss: {loss.item():.4f}, Acc: {acc:.4f}")
    return total_loss / len(dataset)


def test(model, dataset, criterion, device):
    model.eval()
    total_loss = 0
    bar = tqdm(enumerate(dataset))
    avg_loss = 0
    avg_acc = 0
    for i, (data, _) in bar:
        data = data.to(device)
        out = model(data.x, data.edge_index)
        loss = criterion(out[0], data.y)
        acc = (out[0].argmax(dim=1) == data.y).sum().item() / data.y.shape[0]
        total_loss += loss.item()
        avg_loss = (avg_loss * i + loss.item()) / (i + 1)
        avg_acc = (avg_acc * i + acc) / (i + 1)
        bar.set_description(f"Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")
    return total_loss / len(dataset)

def main(args):
    # Load the dataset
    with open(args.data_path, "rb") as f:
        dataset = pkl.load(f)
    with open(args.word_dict_path, "rb") as f:
        word_dict = pkl.load(f)
    
    splits = k_fold_split(dataset, args.k)
    model = MPNNModel(128, 4, calc_feature_size(w2i=word_dict), 5, 7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    best_f1 = 0
    for train_idx, test_idx in splits:
        train_subset = [dataset[i] for i in train_idx]
        test_subset = [dataset[i] for i in test_idx]
        for epoch in range(args.epochs):
            print(f"Epoch {epoch+1}")
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            train_loss = train(model, train_subset, criterion, optimizer, device)
            print(f"Train loss: {train_loss:.4f}")
            test_loss = test(model, test_subset, criterion, device)
            print(f"Test loss: {test_loss:.4f}")
            all_preds = []
            for data in test_subset:
                data = data.to(device)
                out = model(data.x, data.edge_index)
                all_preds.append(out[0].argmax(dim=1).cpu().numpy())
            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate([data.y.cpu().numpy() for data in test_subset])
            f1 = f1_score(all_labels, all_preds, average="macro")
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), args.model_path)
                # Full classification report
                print(classification_report(all_labels, all_preds))
                print("Saved model")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training script.")
    parser.add_argument('--data_path', type=str, default="all_data_processed_gnn/data_encoded.pkl",
                        help='path to the preprocessed data')

    parser.add_argument('--word_dict_path', type=str, default="all_data_processed_gnn/word_dict.pkl",
                        help='path to the preprocessed data')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--use_cuda', action='store_true', help='use CUDA if available')
    parser.add_argument('--save_interval', type=int, default=10, 
                        help='save model every save_interval epochs')
    parser.add_argument('--save_dir', type=str, default='./saved_models_gnn',
                        help='directory to save trained models')
    parser.add_argument('--k', type=int, default=5)
    
    args = parser.parse_args()
    main(args)
