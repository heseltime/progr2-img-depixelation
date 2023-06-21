"""
Author: Jack Heseltine
Matr.Nr.: 01409574
Assignment 1, Exercise 2 - Spring (Summer) Semester 2023
"""

import sys
import os
import shutil
import glob
from PIL import Image
import numpy as np
import hashlib


def validate_images(input_dir: str, output_dir: str, log_file: str, formatter: str = "07d"):
    # create necessary file structure for ...
    # ... output
    os.makedirs(output_dir, exist_ok=True)

    # ... log
    log_file = os.path.abspath(log_file)
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)

    # ... and input + get input
    input_dir = os.path.abspath(input_dir)
    inp_files = glob.glob(os.path.join(input_dir, '**', '*'), recursive=True)
    inp_files.sort()

    # variable to hold the work + central processing loop
    out_files = []
    hashes = []
    serial = 0
    # loop
    with open(log_file, "w") as log:
        for f in inp_files:
            # prepare basename, for BOTH log and output
            temp = '{:' + formatter + '}.jpg'
            new_name = temp.format(serial)
            serial += 1

            # prepare error checking
            err_code = 0

            try:
                # error checking

                # 1
                if not f.endswith(('.jpg', '.JPG', '.jpeg', '.JPEG')):
                    err_code = 1
                    raise ValueError(err_code)

                # 2
                if os.path.getsize(f) > 250000:
                    err_code = 2
                    raise ValueError(err_code)

                # 3
                # at this point in the error cascade image can be opened
                try:
                    im = Image.open(f)
                    bytes = im.tobytes()
                    w, h = im.size

                    # 4
                    np_image = np.array(im)
                    image_shape = np_image.shape
                    if not (len(image_shape) == 3 and image_shape[0] >= 100 and image_shape[1] >= 100):
                        err_code = 4
                        raise ValueError(err_code)

                    # 5
                    if np.std(np_image) <= 0:
                        err_code = 5
                        raise ValueError(err_code)

                    hash = hashlib.md5(bytes).hexdigest()

                    # duplicate check here
                    if hash in hashes:
                        err_code = 6
                        raise ValueError(err_code)
                    else:
                        hashes.append(hash)

                    im.close()
                except:
                    # PIL has issue with file = err code 3
                    err_code = 3
                    raise ValueError(err_code)
                

            # errors to log
            except ValueError:
                # catch error code errors
                #log.write(f + ";" + str(err_code) + "\n")
                log.write(new_name + ";" + str(err_code) + "\n")

            # copy to output directory if all good
            if err_code == 0:
                shutil.copy(f, os.path.join(output_dir, new_name))

if __name__ == '__main__':
    if len(sys.argv) < 4 or len(sys.argv) > 5:
        raise SyntaxError("Wrong number of arguments: arguments are input_dir, output_dir, log_file, (optional:) format (str).")
    elif len(sys.argv) == 4:
        validate_images(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        validate_images(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
