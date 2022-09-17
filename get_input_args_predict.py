# Imports python modules
import argparse


def get_input_args_predict():
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()

    # Creating command line arguments as mentioned above using add_argument() from ArguementParser method
    parser.add_argument('dir', type = str, default = 'flowers/test/100/image_07896.jpg', 
                    help = 'path to image') 
    parser.add_argument('checkpoint', type = str, default = 'model.pth', 
                    help = 'path to saved Model')
    parser.add_argument('--top_k', type = int, default = 3, 
                    help = ' top K most likely classes(predicted)')
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', 
                        help = 'path to the JSON file(mapping of categories to real names)')
    parser.add_argument('--gpu', type = bool, default = True, 
                    help = 'Choose to use gpu or not(if gpu is available)') 

    return parser.parse_args()