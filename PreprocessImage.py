from numpy import expand_dims
from keras.preprocessing.image import load_img, img_to_array

def load_image_pixels(filename, shape):
    image = load_img(filename)
    width, height = image.size
    image = load_img(filename, interpolation = 'bilinear', target_size=shape)
    image = img_to_array(image)
    image = image.astype('float32')
    image /= 255.0
    image = expand_dims(image, 0)    
    return image, width, height


