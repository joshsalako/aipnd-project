import torch
from torchvision import datasets, transforms

def data_loader(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    data_transforms={'train_transforms':transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(256),
                                       transforms.Resize(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])]),
                'valid_transforms':transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
                    }
     #Load the datasets with ImageFolder
    image_data = {'train':datasets.ImageFolder(train_dir, transform=data_transforms['train_transforms']),
                  'valid':datasets.ImageFolder(valid_dir, transform=data_transforms['valid_transforms'])
                 }
    #save class to index
    class_to_idx=image_data['train'].class_to_idx
    BATCH=64
    dataloaders = {'train':torch.utils.data.DataLoader(image_data['train'], batch_size=BATCH, shuffle=True),
                  'valid':torch.utils.data.DataLoader(image_data['valid'], batch_size=BATCH)
                  }
    
    return dataloaders, class_to_idx
    
    