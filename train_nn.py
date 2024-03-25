import os
from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle5 as pickle
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
from  matplotlib.ticker import FuncFormatter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
seed = 663
np.random.seed(seed)
torch.manual_seed(seed)

load_dotenv()

def transform_data_for_processing(df, num_pca, is_train):
    print(f"transforming {'training' if is_train else 'testing'} data")
    # transform arrival_date from datetime to int
    df["arrival_date"] = df["arrival_date"].astype("int64")
    # explode the vec_product_details column 
    vecs = df["vec_product_details"].explode().to_frame()
    vecs["obs_id"] = vecs.groupby(level=0).cumcount()
    vecs = vecs.pivot(columns="obs_id", values="vec_product_details").fillna(0)
    vecs = vecs.add_prefix("vec_product_details_")
    # put exploded data and rest together
    df = pd.concat([df, vecs], axis=1)
    df.drop(columns=["vec_product_details"], inplace=True)
    X = df.copy(deep=True)
    X.drop(columns=["country_of_origin_labels"], inplace=True)
    y = df.copy(deep=True)
    y = y[["country_of_origin_labels"]]
    # scale features for PCA
    standard_scaler = StandardScaler()
    if is_train:
        standard_scaler.fit(X)
        pickle.dump(standard_scaler, open("models/standard_scaler_pca.pk", "wb"))
    standard_scaler = pickle.load(open("models/standard_scaler_pca.pk", "rb"))
    X = standard_scaler.transform(X)
    # pca
    pca = PCA(n_components=num_pca)
    if is_train:
        pca.fit(X, y)
        pickle.dump(pca, open("models/pca.pk", "wb"))
    pca = pickle.load(open("models/pca.pk", "rb"))
    X = pca.transform(X)
    X =  torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y.to_numpy(), dtype=torch.float32).reshape(-1, 1)
    y = torch.flatten(y)
    y = y.type(torch.LongTensor)
    return X, y


class MLP(nn.Module):
    def __init__(self, num_features, num_targets):
        super().__init__()
        self.linear1 = nn.Linear(num_features, 120, bias=True)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(120, num_targets, bias=True)
        # don't need softmax, since using cross-entropy loss

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


def get_accuracy(model, X, y, test_batch_size=5000):
    model.eval()
    test_loader = DataLoader(list(zip(X,y)), shuffle=True, batch_size=test_batch_size)
    accuracies = []
    for X_batch, y_batch in test_loader:
        y_hat = model(X_batch)
        acc = (torch.argmax(y_hat, 1) == y_batch).float().mean()
        accuracies.append(float(acc))
    return sum(accuracies)/len(accuracies)


def gradient_descent(model, X_train, y_train, X_test, y_test, batch_size=64, lr=0.1, weight_decay=0.01, num_epochs=10):
    print(f"begin model training.")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(list(zip(X_train,y_train)), shuffle=True, batch_size=batch_size)
    train_accs, test_accs, avg_loss, epochs, losses = [], [], [], [], []

    # train model
    for epoch in range(num_epochs):
        for X_batch, y_batch in iter(loader):
            y_hat = model(X_batch)
            loss = criterion(y_hat, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss)/batch_size) 

        # evaluate model
        train_acc = get_accuracy(model, X_train, y_train)
        train_accs.append(train_acc)
        test_acc = get_accuracy(model, X_test, y_test)
        test_accs.append(test_acc)
        epochs.append(epoch)
        avg_loss.append(sum(losses)/len(losses))
        print(f"Epoch: {epoch} => avg train acc: {sum(train_accs)/len(train_accs)} | avg test acc: {sum(test_accs)/len(test_accs)}")

    # plot loss
    loss_df = pd.DataFrame({
        "epochs": epochs,
        "loss": avg_loss
    })
    loss_df.head()
    sns.lineplot(x="epochs", y="loss", data=loss_df)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))
    plt.title("Training Curve (batch_size={}, lr={})".format(batch_size, lr))
    plt.show()

    # plot acc
    acc_df = pd.DataFrame({
        "epochs": epochs,
        "train acc": train_accs,
        "test acc": test_accs
    })
    sns.lineplot(x="epochs", y="value", hue="variable", data=pd.melt(acc_df, ["epochs"]))
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))
    plt.ylabel("accuracy")
    plt.legend(loc='best')
    plt.show()
    return model

if __name__ == "__main__":
    train_file = os.getenv("OUT_DATA_FILE_TRAIN")
    test_file = os.getenv("OUT_DATA_FILE_TEST")
    trained_model_file = os.getenv("TRAINED_MODEL_FILE")
    train_df = pd.read_pickle(train_file)
    train_df = train_df.sample(frac=1)
    test_df = pd.read_pickle(test_file)
    test_df = test_df.sample(frac=1)

    # load train and test data for torch processing
    X_train, y_train = transform_data_for_processing(train_df, num_pca=280, is_train=True)
    X_test, y_test = transform_data_for_processing(test_df, num_pca=280, is_train=False)

    # sanity check
    print(f"sanity check:")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_test: {y_test.shape}")
    num_features_train = X_train.shape[1]
    num_instances_train = X_train.shape[0]
    num_features_test = X_test.shape[1]
    num_instances_test = y_test.shape[0]
    num_targets = len(train_df["country_of_origin_labels"].unique())
    print(f"num_targets: {num_targets}")

model = MLP(num_features_train, num_targets)

batch_size = int(os.getenv("BATCH_SIZE"))
learning_rate = float(os.getenv("LEARNING_RATE"))
weight_decay = float(os.getenv("WEIGHT_DECAY"))
n_epochs = int(os.getenv("NUM_EPOCHS"))
model = gradient_descent(model, X_train, y_train, X_test, y_test, 
                         batch_size=batch_size, lr=learning_rate, weight_decay=weight_decay, num_epochs=n_epochs)

torch.save(model.state_dict(), trained_model_file)