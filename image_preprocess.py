from PIL import Image
import numpy as np

#defining the image proprocessing function
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    #loading the image
    img = Image.open(image)
    
    #Resizing and cropping
    img=img.resize((256, 256))
    left_margin = (img.width-224)/2
    bottom_margin = (img.height-224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    img = img.crop((left_margin, bottom_margin, right_margin, top_margin))
    
    #Scaling and transposing
    np_img = np.array(img)/225
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    np_img = (np_img-mean)/std
    np_img = np_img.transpose((2, 0, 1))
    
    return np_img
