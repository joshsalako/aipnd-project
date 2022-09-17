#import modules and functions
from get_input_args_train import get_input_args_train
from data_loader import data_loader
import torch
from torchvision import models
from torch import nn, optim
import torch.nn.functional as F
from workspace_utils import active_session
from time import time

#get argurments
in_arg = get_input_args_train()

#get data loaded
dataloaders, class_to_idx=data_loader(in_arg.dir)

#build the model
def build_model(arch=in_arg.arch, lr=in_arg.learning_rate,output_units=in_arg.output_units,
          hidden_units=in_arg.hidden_units):
    if arch=='vgg16':
        model = models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = nn.Sequential(nn.Linear(25088, hidden_units), nn.ReLU(), nn.Dropout(0.3),
                                         nn.Linear(hidden_units, int(hidden_units/2)), nn.ReLU(), nn.Dropout(0.2),
                                         nn.Linear(int(hidden_units/2), int(hidden_units/4)), nn.ReLU(), nn.Dropout(0.1),
                                         nn.Linear(int(hidden_units/4), output_units),
                                         nn.LogSoftmax(dim=1))
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
        return model, criterion, optimizer, arch
    
    elif arch=='resnet50':
        model = models.resnet50(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Sequential(nn.Linear(2048, hidden_units), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(hidden_units, int(hidden_units/2)), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(int(hidden_units/2), int(hidden_units/4)), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(int(hidden_units/4), output_units),
                                 nn.LogSoftmax(dim=1))
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.fc.parameters(), lr=lr)
        return model, criterion, optimizer, arch
        
    else:
        raise NameError('Architecture is not recognized..')

def main(dataloaders=dataloaders, epochs=in_arg.epochs, gpu=in_arg.gpu, save_path=in_arg.save_dir):
    #trainning loop
    t1=time()
    model, criterion, optimizer, arch= build_model()
    device = torch.device("cuda" if gpu else "cpu")
    print_every=20 #print something after this number of weight updates
    model = model.to(device)
    steps = 0
    running_loss = 0
    with active_session():
        for epoch in range(epochs):
            for inputs, labels in dataloaders['train']:
                steps += 1
                # Move input and label tensors to the default device
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                logps = model.forward(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                if steps % print_every == 0:
                    validation_loss = 0
                    validation_accuracy = 0
                    model.eval()
                    with torch.no_grad():
                        for inputs, labels in dataloaders['valid']:
                            inputs, labels = inputs.to(device), labels.to(device)
                            logps = model.forward(inputs)
                            batch_loss = criterion(logps, labels)

                            validation_loss += batch_loss.item()
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            validation_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Training loss: {running_loss/print_every:.3f}.. "
                          f"Validation loss: {validation_loss/len(dataloaders['valid']):.3f}.. "
                          f"Validation accuracy: {validation_accuracy/len(dataloaders['valid']):.3f}")
                    running_loss = 0
                    model.train()
        time_elapsed=time()-t1
        print("\n** Total Elapsed Runtime:",
              str(int((time_elapsed/3600)))+":"+str(int((time_elapsed%3600)/60))+":"
              +str(int((time_elapsed%3600)%60)) )                     
    
    #saving the model
    model.class_to_idx = class_to_idx
    model.cpu()

    checkpoint = {'arch': arch,
                  'hidden_units': in_arg.hidden_units,#saving hidding layers for reloading model
                  'output_units': in_arg.output_units,
                  'class_to_idx': model.class_to_idx,
                  'state_dict': model.state_dict()}
    torch.save(checkpoint, save_path)

# Call to main function to run the program
if __name__ == "__main__":
    main()
    
    
    
    