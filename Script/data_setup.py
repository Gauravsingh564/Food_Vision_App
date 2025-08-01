from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import os
NUM_WORKERS=os.cpu_count()
def create_dataloaders(train_dir:str,
                       test_dir:str,
                       transform:transforms.Compose,
                       batch_size:int,
                       num_workers:int=NUM_WORKERS):
  train_data=datasets.ImageFolder(root=train_dir,
                                  transform=transform
                                  )
  test_data=datasets.ImageFolder(root=test_dir,
                                  transform=transform
                                )
  class_names=train_data.classes
  train_dataloader=DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  shuffle=True,
                                  pin_memory=True)
  test_dataloader=DataLoader(dataset=test_data,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  shuffle=False,
                                  pin_memory=True)
  return train_dataloader,test_dataloader,class_names
  
