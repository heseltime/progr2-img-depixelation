import sys
import os

from PIL import Image
import numpy as np

def to_grayscale(pil_image: np.ndarray) -> np.ndarray:
    # dimension checking/raise value errors appropriately
    grayscale_input = False
    if len(pil_image) == 2:
        grayscale_input = True # assumed
    elif len(pil_image == 3):
        if pil_image.shape[2] == 3:
            R, G, B = pil_image[:,:,0], pil_image[:,:,1], pil_image[:,:,2]
        else:
            raise ValueError('to_grayscale() needs the third dimension \
                             of the image to have value 3 for RGB conversion')
    else:
        raise ValueError('to_grayscale() only converts 1- and 3-dimensional images')
    
    if not grayscale_input:
        # normalization
        R_normalized = R / 255
        G_normalized = G / 255
        B_normalized = B / 255

        # linearization
        R_linear = np.where(R_normalized <= 0.04045, R_normalized/12.92, ((R_normalized + 0.055)/1.055)**2.4)
        G_linear = np.where(G_normalized <= 0.04045, G_normalized/12.92, ((G_normalized + 0.055)/1.055)**2.4)
        B_linear = np.where(B_normalized <= 0.04045, B_normalized/12.92, ((B_normalized + 0.055)/1.055)**2.4)

        # brightness
        Y_linear = 0.2126 * R_linear + 0.7152 * G_linear + 0.0722 * B_linear
        Y_linear_1d = Y_linear.ravel()

        # grayscale output
        Y = np.where(Y_linear <= 0.0031308, 12.92*R_normalized, 1.055*R_normalized**(1/2.4) - 0.055)

        Y_denormalized = Y * 255

        # integer case: round and cast to integer
        if np.issubdtype(pil_image.dtype, np.integer):
            Y_denormalized = np.round(Y_denormalized, 0).astype(int)
        # otherwise float values returned

        arr_3d = Y_denormalized[None, :]

    else: 
        # grayscale case
        Gr_normalized = pil_image[:,:] / 255
        Gr_linear = np.where(Gr_normalized <= 0.04045, Gr_normalized/12.92, ((Gr_normalized + 0.055)/1.055)**2.4)
        # assuming Y_linear is equivalent to the above line 
        Y_linear_1d = Gr_linear.ravel()
        arr_3d = pil_image[None, :]

    #np.insert(arr_3d, 0, Y_linear_1d)
    #print(arr_3d)
    return arr_3d
        

if __name__ == '__main__':
    input_args = sys.argv

    # in case want to process mutliple
    image_paths = []
    for i in range(1, len(input_args)):
        image_paths.append(input_args[i])

    for image_path in image_paths:
        image = np.asarray(Image.open(image_path))
        #print(image)
        gray_image_array = to_grayscale(image)
        #print(gray_image[0].shape)

        gray_image = Image.fromarray(gray_image_array[0])
        # RGB mode needed for saving in this main function
        if gray_image.mode != 'RGB':
            gray_image = gray_image.convert('RGB')

        file_name, file_ext = os.path.splitext(image_path)
        gray_image.save(file_name + '_gray.jpg')
