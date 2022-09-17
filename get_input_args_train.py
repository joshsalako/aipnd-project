# Imports python modules
import argparse


def get_input_args_train():
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()

    # Creating command line arguments as mentioned above using add_argument() from ArguementParser method
    parser.add_argument('dir', type = str, default = 'flowers', 
                    help = 'path to the data directory') 
    parser.add_argument('--arch', type = str, default = 'vgg16', 
                    help = 'Model Architechture (Either vgg16 or resnet50)')
    parser.add_argument('--save_dir', type = str, default = 'model.pth', 
                    help = 'path to save Model')
    parser.add_argument('--learning_rate', type = float, default = 0.001, 
                    help = 'learning rate hyperparameter')
    parser.add_argument('--output_units', type = int, default = 102, 
                        help = 'number of classes in the output layer')
    parser.add_argument('--hidden_units', type = int, default = 4096, 
                    help = 'number of neurons in the first hidden layer(subsequent hidden layers goes down by factor of 2)')
    parser.add_argument('--epochs', type = int, default = 3, 
                    help = 'number of epochs')
    parser.add_argument('--gpu', type = bool, default = True, 
                    help = 'Choose to use gpu or not(if gpu is available)') 

    return parser.parse_args()