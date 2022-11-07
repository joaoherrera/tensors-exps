import torch

from models.dataset.dataset import CocoDataset
from models.dataset.models import DummyClassifier
from models.operations import Supervision
from torch.utils.data import random_split


def main():
    dataset_root = "/home/joaoherrera/datasets/loose_screw_dataset"
    dataset_images_directory = f"{dataset_root}/images"
    dataset_annotations_file = f"{dataset_root}/annotations/annotations.json"
    
    model = DummyClassifier()
    dataset = CocoDataset(dataset_images_directory, dataset_annotations_file)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    loss = torch.nn.BCELoss()
    
    train_set, test_set = random_split(dataset, lengths=[0.8, 0.2])
    trainer = Supervision(torch.device("cuda:0"), model)
    trainer.fit(train_set, test_set, optimizer, loss, loss)
        
if __name__ == "__main__":
    main()