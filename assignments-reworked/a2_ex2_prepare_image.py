import sys
import os
import math

from PIL import Image
import numpy as np

import a2_ex1 as ex1

def prepare_image(image: np.ndarray, 
                  x: int, 
                  y: int, 
                  width: int, 
                  height: int, 
                  size: int
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # precede with value checks
    if (len(image.shape) != 3 or image.shape[0] != 1):
        raise ValueError('Problem with dimension of image passed \
                         (dimension must be three, only one channel must be present)')
    
    if (width < 2 or height < 2 or size < 2 ):
        raise ValueError('Values passed are too small')
    
    if (x < 0 or x > x + width or y < 0 or y > y + width):
        raise ValueError('Coordinate parameters are implausible \
                         (x or y is too small or too big)')

    # start preparing
    #
    img_copy = np.copy(image[0]) # image in channel 1
    
    # consider the section of the image to pixelate
    img_section = img_copy[y:y+height, x:x+width]
    
    section_height, section_width = img_section.shape
    
    # number of pixelations in each dimension
    pixel_count_x = math.ceil(section_width / size) 
        # ceiling to iterate to end of the section below
    pixel_count_y = math.ceil(section_height / size)
    
    # Calculate the dimensions of the pixelated section
    pixelated_width = pixel_count_x * size
    pixelated_height = pixel_count_y * size
    
    # Create a blank image to store the pixelated section
    pixelated_section = np.zeros((pixelated_height, pixelated_width), dtype=np.uint8) 
        # should be shape (width, height)
    
    # Iterate over the pixels in the pixelated section
    for i in range(pixel_count_y): # rows
        for j in range(pixel_count_x): # columns
            # Calculate the coordinates of the pixel in the original section
            y_start = i * size
            x_start = j * size

            # calculate end of the pixel block, 
                # can be cut off to match height/width specification exactly
            if y_start + size <= y + height:
                y_end = y_start + size
            else:
                y_end = y + height

            if y_start + size <= y + width:
                x_end = x_start + size
            else:
                x_end = x + width         
            
            # Get the color of the pixel in the original section
            pixel_color = np.mean(img_section[y_start:y_end, x_start:x_end], axis=(0, 1))
            
            # Set the color of the corresponding pixels in the pixelated section
            pixelated_section[y_start:y_end, x_start:x_end] = pixel_color
    
    # binary mask pixels in the pixelated area vs original
    mask = np.ones((img_copy.shape[0], img_copy.shape[1]), dtype=bool)
    mask[height:(height+pixelated_height), width:(width+pixelated_width)] = False

    mask_arr_3d = mask[None, :]

    # insert pixelated section back into image copy 
    img_copy[y:y+height, x:x+width] = pixelated_section

    pixelated_img_arr_3d = img_copy[None, :]
    
    # return the pixelated version, the mask, 
    # and the original image (originally 3D) as target
    return pixelated_img_arr_3d, mask_arr_3d, image

if __name__ == '__main__':
    input_args = sys.argv

    # in case want to process mutliple
    image_paths = []
    for i in range(1, len(input_args)):
        image_paths.append(input_args[i])     

    for image_path in image_paths:
        image = np.asarray(Image.open(image_path))

        # part 1 of this assignment
        gray_image_array = ex1.to_grayscale(image)

        # part 2 of this assignment
        pixelated_image_array, known_array, target_array \
            = prepare_image(gray_image_array, 100, 200, 180, 150, 5)

        # print shapes to check
        print(pixelated_image_array.shape)
        print(known_array.shape)
        print(target_array.shape)

        # save pxielated version
        pixelated_image = Image.fromarray(pixelated_image_array[0])
        # RGB mode needed for saving in this main function
        if pixelated_image.mode != 'RGB':
            pixelated_image = pixelated_image.convert('RGB')

        file_name, file_ext = os.path.splitext(image_path)
        pixelated_image.save(file_name + '_pixelated.jpg')

        # print the known array to check
        print(known_array)
        
        # target is the original image (input)