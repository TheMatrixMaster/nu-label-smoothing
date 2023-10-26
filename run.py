import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from torchvision import datasets, transforms

import umap
import math
from tqdm import tqdm
from sklearn.preprocessing import scale

np.random.seed(49)
torch.manual_seed(49)

from models import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class NULabelSmoothing():
    def __init__(self, dataset: str, model: str) -> None:
        # Load the dataset
        self.dataset = dataset
        self.train, self.test = self.load_dataset()
        self.x_size = math.prod([x for x in self.train.data.shape[1:]])
        self.hw_size = math.prod([x for x in self.train.data.shape[1:-1]])
        self.num_classes = len(np.unique(self.train.targets))

        self.train_umap = self.compute_umap_embeddings(save=True)
        self.class_centers = self.obtain_class_centers()
        self.distances = self.dist_to_centers()
        self.smoothed_y = self.get_smoothing_values()

        assert model in ["mlp", "vgg", "cnn"]
        self.model = model

    def load_dataset(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

        if self.dataset == "mnist":
            trainset = datasets.MNIST('data/MNIST_data/', download=True, train=True, transform=transform)
            testset = datasets.MNIST('data/MNIST_data/', download=True, train=False, transform=transform)
        elif self.dataset == "cifar10":
            trainset = datasets.CIFAR10('data/CIFAR10/', download=True, train=True, transform=transform)
            testset = datasets.CIFAR10('data/CIFAR10/', download=True, train=False, transform=transform)
        
        return trainset, testset 

    def compute_umap_embeddings(self, save=False):
        print("Computing UMAP embeddings...")
        # Check if we have already computed the embeddings
        try:
            images_map = np.load(f'data/umap_{self.dataset}.npy')
            return images_map
        except:
            pass

        # Flatten the images
        X = self.train.data.reshape(-1, self.x_size)
        y = np.array(self.train.targets)

        # Scale the images
        images = scale(X)

        # Perform umap
        fit = umap.UMAP(n_neighbors=15, random_state=42, metric='euclidean', init="random")
        images_map = fit.fit_transform(images)

        plt.scatter(images_map[:, 0], images_map[:, 1], c=y, alpha=0.1, s=5)
        plt.savefig(f'plots/umap_{self.dataset}.png')

        if save:
            np.save(f'data/umap_{self.dataset}.npy', images_map)

        return images_map

    def obtain_class_centers(self):
        print("Obtaining class centers...")
        # make cluster centers
        cluster_centers = []
        y = np.array(self.train.targets)

        for c in range(10):
            x = self.train_umap[y == c]
            avg_x = np.average(x, axis=0)
            cluster_centers.append(avg_x)

        cluster_centers = np.array(cluster_centers)
        return cluster_centers
    
    def dist_to_centers(self):
        print("Computing distances of each point to cluster center...")
        # Compute distance of each image to the class centers
        distances = []
        for emb in tqdm(self.train_umap):
            d = [np.linalg.norm(emb-c) for c in self.class_centers]
            distances.append(d)

        distances = np.array(distances)
        return distances
    
    def get_smoothing_values(self):
        print("Computing softmax of distances to obtain smoothing values...")
        # Compute the softmax of the distances
        y = np.array(self.train.targets)
        count = 0
        
        for i, distance in enumerate(tqdm(self.distances)):
            label = y[i]
            argmax = np.argmin(distance)
            if argmax != label:
                count += 1
                tmp = distance[label]
                distance[label] = distance[argmax]
                distance[argmax] = tmp

        distances_softmax = NULabelSmoothing.softmax(-self.distances, temperature=.05)

        print(f"There were {count} mismatched labels to cluster centers.")
        print(f"Mean softmaxed distance is {np.average(np.max(distances_softmax, axis=1))}")
        return distances_softmax
    
    def run(self):
        print("Transforming data to tensors...")
        
        if self.model == "mlp":
            X = torch.tensor(self.train.data, dtype=torch.float).reshape(-1, self.x_size).to(device)
            X_val = torch.tensor(self.test.data, dtype=torch.float).reshape(-1, self.x_size).to(device)
        else:
            X = torch.tensor(self.train.data, dtype=torch.float)
            X_val = torch.tensor(self.test.data, dtype=torch.float)

            if X.ndim == 3:
                X = X.unsqueeze(1)
                X_val = X_val.unsqueeze(1)

            if X.shape[1] > 3:
                X = X.permute(0, 3, 1, 2).to(device)
                X_val = X_val.permute(0, 3, 1, 2).to(device)
        
        y = F.one_hot(torch.tensor(self.train.targets), num_classes=self.num_classes).to(torch.float).to(device)
        y_val = F.one_hot(torch.tensor(self.test.targets), num_classes=self.num_classes).to(torch.float).to(device)
        smoothed_y = torch.tensor(self.smoothed_y, dtype=torch.float).to(device)

        trainloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, torch.stack((y, smoothed_y), dim=1)), batch_size=64, shuffle=True)
        validloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_val, y_val), batch_size=64, shuffle=False)

        if self.model == "mlp":
            model1 = MLP(self.x_size, self.num_classes).to(device)
            model2 = MLP(self.x_size, self.num_classes).to(device)
        elif self.model == "vgg":
            model1 = VGG16().to(device)
            model2 = VGG16().to(device)
        else:
            model1 = CNN(*X.shape[-3:]).to(device)
            model2 = CNN(*X.shape[-3:]).to(device)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        criterion2 = nn.CrossEntropyLoss()

        optimizer = optim.SGD(model1.parameters(), lr=0.01)
        optimizer2 = optim.SGD(model2.parameters(), lr=0.01)

        epochs = 40
        num_iterations = len(trainloader)

        train_loss = []
        train_loss1 = []
        train_accs = []
        train_accs1 = []
        test_loss = []
        test_loss1 = []
        test_accs = []
        test_accs1 = []

        print("Start training...")
        for e in range(epochs):
            print(f'Epoch {e+1}/{epochs}')
            running_loss = 0
            running_loss2 = 0
            running_accuracy = 0
            running_accuracy2 = 0

            model1.train()
            model2.train()
            
            for i, (x_batch, y) in enumerate(trainloader):
                optimizer.zero_grad()
                optimizer2.zero_grad()

                x_batch = x_batch.to(device)
                y_batch = y[:, 0].to(device)
                y_batch_sm = y[:, 1].to(device)

                y_hat_batch = model1.forward(x_batch)
                y_hat_batch2 = model2.forward(x_batch)
                loss = criterion(y_hat_batch, y_batch)
                loss2 = criterion2(y_hat_batch2, y_batch_sm)

                accuracy = (y_batch.argmax(dim=1) == y_hat_batch.argmax(dim=1)).float().mean()
                accuracy2 = (y_batch.argmax(dim=1) == y_hat_batch2.argmax(dim=1)).float().mean()

                running_loss += loss.item()
                running_loss2 += loss2.item()
                running_accuracy += accuracy.item()
                running_accuracy2 += accuracy2.item()

                loss.backward()
                loss2.backward()

                optimizer.step()
                optimizer2.step()

            
            final_loss = running_loss / num_iterations
            final_loss2 = running_loss2 / num_iterations
            final_accuracy = running_accuracy / num_iterations
            final_accuracy2 = running_accuracy2 / num_iterations

            train_loss.append(final_loss)
            train_loss1.append(final_loss2)
            train_accs.append(final_accuracy)
            train_accs1.append(final_accuracy2)

            print(f'Accuracy: {final_accuracy} vs {final_accuracy2}')
            print(f'Loss: {final_loss} vs {final_loss2}')
            print()

            model1.eval()
            model2.eval()

            val_loss, val_loss2 = 0, 0
            val_acc, val_acc2 = 0, 0

            for i, (x_batch, y_batch) in enumerate(validloader):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                y_hat_batch = model1.forward(x_batch)
                y_hat_batch2 = model2.forward(x_batch)

                loss = criterion(y_hat_batch, y_batch)
                loss2 = criterion2(y_hat_batch2, y_batch)

                accuracy = (y_batch.argmax(dim=1) == y_hat_batch.argmax(dim=1)).float().mean()
                accuracy2 = (y_batch.argmax(dim=1) == y_hat_batch2.argmax(dim=1)).float().mean()

                val_acc += accuracy.item()
                val_acc2 += accuracy2.item()

                val_loss += loss.item()
                val_loss2 += loss2.item()

            accuracy = val_acc / len(validloader)
            accuracy2 = val_acc2 / len(validloader)
            loss = val_loss / len(validloader)
            loss2 = val_loss2 / len(validloader)

            test_loss.append(loss)
            test_loss1.append(loss2)
            test_accs.append(accuracy)
            test_accs1.append(accuracy2)

            print('Evaluating...')
            print('Accuracy: ', accuracy, accuracy2)
            print('Loss: ', loss, loss2)
            print()

        self.plot_losses(train_loss, train_loss1, test_loss, test_loss1)
        self.plot_accs(train_accs, train_accs1, test_accs, test_accs1)


    @staticmethod
    def softmax(x, temperature=1.0):
        # softmax with temperature
        e_x = np.exp(x/temperature - np.max(x/temperature, axis=1)[:, None])
        return e_x / e_x.sum(axis=1)[:, None]
    
    def plot_losses(self, train_loss, train_loss1, test_loss, test_loss1):
        plt.plot(train_loss, label="Train loss with uniform smoothing")
        plt.plot(train_loss1, label="Train loss with nu-smoothing")
        plt.plot(test_loss, label="Test loss with uniform smoothing")
        plt.plot(test_loss1, label="Test loss with nu-smoothing")
        plt.legend()
        plt.savefig(f"plots/{self.dataset}-{self.model}-losses.png")
        plt.clf()

    def plot_accs(self, train_accs, train_accs1, test_accs, test_accs1):
        plt.plot(train_accs, label="Train accuracy with uniform smoothing")
        plt.plot(train_accs1, label="Train accuracy with nu-smoothing")
        plt.plot(test_accs, label="Test accuracy with uniform smoothing")
        plt.plot(test_accs1, label="Test accuracy with nu-smoothing")
        plt.legend()
        plt.savefig(f"plots/{self.dataset}-{self.model}-accs.png")
        plt.clf()


if __name__ == "__main__":
    NULabelSmoothing("cifar10", "vgg").run()
    # NULabelSmoothing("mnist", "cnn").run()
    