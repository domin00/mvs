import numpy as np
import re
import sys
from PIL import Image

def read_cam_file(filename):
    # TODO

    f = open(filename, 'r')

    data = f.readlines()

    array1 = data[1].split(" ")
    array2 = data[2].split(" ")
    array3 = data[3].split(" ")
    array4 = data[4].split(" ")

    extrinsics = [[float(i) for i in array1[0:-1]],
                 [float(i) for i in array2[0:-1]],
                 [float(i) for i in array3[0:-1]],
                 [float(i) for i in array4[0:-1]]]

    array7 = data[7].split(" ")
    array8 = data[8].split(" ")
    array9 = data[9].split(" ")

    intrinsics = [[float(i) for i in array7[0:-1]],
                 [float(i) for i in array8[0:-1]],
                 [float(i) for i in array9[0:-1]]]

    depth = data[11].split(" ")
    depth_min = float(depth[0])
    depth_max = float(depth[1])

    return np.float32(np.array(intrinsics)), np.float32(np.array(extrinsics)), np.float32(np.array(depth_min)), np.float32(np.array(depth_max))

def read_img(filename):
    #TODO
    img = Image.open(filename) # load image using PILLOW
    np_img = np.array(img, dtype=np.float32) # load as float array
    np_img = np_img/255.0 # divide the 0-255 values by 255.0 to get a range from 0 to 1

    return np_img

def read_depth(filename):
    # read pfm depth file
    return np.array(read_pfm(filename)[0], dtype=np.float32)

def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def save_pfm(filename, image, scale=1):
    file = open(filename, "wb")
    color = None

    image = np.flipud(image)

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n'.encode('utf-8') if color else 'Pf\n'.encode('utf-8'))
    file.write('{} {}\n'.format(image.shape[1], image.shape[0]).encode('utf-8'))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(('%f\n' % scale).encode('utf-8'))

    image.tofile(file)
    file.close()
