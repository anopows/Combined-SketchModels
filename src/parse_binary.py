import numpy as np
import struct
from struct import unpack
import cv2 as cv

# Folders for storage/retrival
data_directory = '../data/'
binaries_folder = data_directory + 'binaries/'

def _unpack_drawing(file_handle):
    key_id, = unpack('Q', file_handle.read(8))
    countrycode, = unpack('2s', file_handle.read(2))
    recognized, = unpack('b', file_handle.read(1))
    timestamp, = unpack('I', file_handle.read(4))
    n_strokes, = unpack('H', file_handle.read(2))
    image = []
    for i in range(n_strokes):
        n_points, = unpack('H', file_handle.read(2))
        fmt = str(n_points) + 'B'
        x = unpack(fmt, file_handle.read(n_points))
        y = unpack(fmt, file_handle.read(n_points))
        image.append((x, y))

    return {
#        'key_id': key_id,
#        'countrycode': countrycode,
#        'recognized': recognized,
#        'timestamp': timestamp,
        'image': image
    }

def _create_img(lines, imgsize = (255,255), output_size = (224,224), border = (5,5), thickness=3):
    """Create an image from given lines (format given by binaries)

    Keyword arguments:
    lines   -- array of lines. Each line consists of connected points: [(x1,x2,x2,..),(y1,y2,y3,..)]
    imgsize -- output size of an image (default (255,255))

    Assume point coordinates are between 0 and 255
    """
    # white image
    imgsizex, imgsizey = (imgsize[0] + 2*border[0], imgsize[1] + 2*border[1])
    img = 255*np.ones((imgsizey,imgsizex), np.uint8) # for opencv first y-coord, then x-coord
    # draw lines
    for line in lines:
        line = np.array(list(zip(*line)))
        line = np.array(border) + line
        line = np.reshape(line, [-1,1,2])
        cv.polylines(img,[line],True,(0,0,0),thickness,32)

    img = cv.resize(img, output_size, interpolation = cv.INTER_CUBIC)
    return img

# Parse from ink array to array of offsets to previous point + normalized
# 3rd dim: indicating w/ 0/1 if last point of a stroke
def _parse_sample(inkarray, preprocessing=False):
    """Parse an ndjson line and return ink (as np array)
    
    Adopted from: https://github.com/tensorflow/models/blob/master/tutorials/rnn/quickdraw/create_dataset.py
    """ 
    stroke_lengths = [len(stroke[0]) for stroke in inkarray]
    total_points = sum(stroke_lengths)
    
    if preprocessing:
        np_ink = np.zeros((total_points, 3), dtype=np.float32)
    else:
        np_ink = np.zeros((total_points, 3), dtype=np.uint8)
        
    current_t = 0
    if not inkarray:
        print("Empty inkarray")
        return None, None
    for stroke in inkarray:
        if len(stroke[0]) != len(stroke[1]):
            print("Inconsistent number of x and y coordinates.")
            return None, None
        for i in [0, 1]:
            np_ink[current_t:(current_t + len(stroke[0])), i] = stroke[i]
        current_t += len(stroke[0])
        np_ink[current_t - 1, 2] = 1  # stroke_end
    
    if preprocessing:
        # Preprocessing.
        # 1. Size normalization.
        lower = np.min(np_ink[:, 0:2], axis=0)
        upper = np.max(np_ink[:, 0:2], axis=0)
        scale = upper - lower
        scale[scale == 0] = 1
        np_ink[:, 0:2] = (np_ink[:, 0:2] - lower) / scale
        # 2. Compute deltas.
        np_ink[1:, 0:2] -= np_ink[0:-1, 0:2]
        np_ink = np_ink[1:, :]
    return np_ink
   
def unpack_binary(filename):
    with open(filename, 'rb') as f:
        while True:
            try:
                yield _unpack_drawing(f)
            except struct.error:
                break
                
# both create parsed sketch (& image) samples
def samples_from_binary(class_name, with_imgs=True):
    file_name = binaries_folder + class_name + '.bin'
    sketches = unpack_binary(file_name)
    
    while True:
        try:
            sketch = next(sketches)['image']
            sketch_parsed = _parse_sample(sketch)
            if with_imgs:
                img = _create_img(sketch)
                yield sketch_parsed,img
            else:
                yield sketch_parsed
        except StopIteration:
            break