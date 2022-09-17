from image_preprocess import process_image
from get_input_args_predict import get_input_args_predict
import torch
from torchvision import models
from torch import nn
import torch.nn.functional as F
import json

    
#get argurments
in_arg = get_input_args_predict()

#load in mapping of classes to name
with open(in_arg.category_names, 'r') as f:
    cat_to_name = json.load(f)

#load saved model
def load_checkpoint(filepath=in_arg.checkpoint):
    checkpoint = torch.load(filepath)
    
    #load hidden and output units
    hidden_units=checkpoint['hidden_units']
    output_units=checkpoint['output_units']
    
    #reloading vgg16 model
    if checkpoint['arch']=='vgg16':
        model=models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.class_to_idx = checkpoint['class_to_idx']
        model.classifier = nn.Sequential(nn.Linear(25088, hidden_units), nn.ReLU(), nn.Dropout(0.3),
                                         nn.Linear(hidden_units, int(hidden_units/2)), nn.ReLU(), nn.Dropout(0.2),
                                         nn.Linear(int(hidden_units/2), int(hidden_units/4)), nn.ReLU(), nn.Dropout(0.1),
                                         nn.Linear(int(hidden_units/4), output_units),
                                         nn.LogSoftmax(dim=1))
        model.load_state_dict(checkpoint['state_dict'])
        return model
    
    #loding resnet50 model
    elif checkpoint['arch']=='resnet50':
        model = models.resnet50(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.class_to_idx = checkpoint['class_to_idx']
        model.fc = nn.Sequential(nn.Linear(2048, hidden_units), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(hidden_units, int(hidden_units/2)), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(int(hidden_units/2), int(hidden_units/4)), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(int(hidden_units/4), output_units),
                                 nn.LogSoftmax(dim=1))        
        model.load_state_dict(checkpoint['state_dict'])
        return model
    
    else:
        raise NameError('Architecture is not recognized..')
        
        
def main(image_path=in_arg.dir, model=load_checkpoint(), topk=in_arg.top_k, gpu=in_arg.gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    #preprocess inputted image
    img = process_image(image_path) 
    img = torch.from_numpy(img).type(torch.FloatTensor) #convert to tensor from numpy
    img = torch.unsqueeze(img, 0) #Due to batch size
    
    #move model and data to gpu
    device = torch.device("cuda" if gpu else "cpu")
    model = model.to(device)
    img = img.to(device)
    
    # Predict top 5
    probs = torch.exp(model.forward(img)) 
    top_probs, top_labs = probs.topk(topk, dim=1)
    top_probs, top_labs = top_probs.cpu(), top_labs.cpu()
    
    #converting index to classes
    top_probs = list(top_probs.detach().numpy())[0]
    top_labs = list(top_labs.detach().numpy())[0]
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [idx_to_class[lab] for lab in top_labs]
    top_flowers = [cat_to_name[classes] for classes in top_classes]   
    print('Predicted Values')
    print('-'*55)
    print('Classes : Certainty')
    for key, value in dict(zip(top_classes, top_probs)).items():
        print(str(key) +' : '+ str(value))
    print()
    print(top_flowers[0])


# Call to main function to run the program
if __name__ == "__main__":
    main()