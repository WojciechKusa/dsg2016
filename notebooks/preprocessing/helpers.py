import os
import itertools
import matplotlib.pyplot as plt
from skimage.io import imread

images_folder = None
default_extension = '.jpg'

def open_image_id(image_id, extension=None):
    if extension == None:
        extension = default_extension
    img_name = str(image_id)
    image_path = os.path.join(images_folder,img_name+extension)
    image = imread(image_path)
    return image

def view_grid_images(a,b, images_dict, imshow_args={}, **kwargs):
    fig, axes = plt.subplots(a,b, **kwargs)
    # iloczyn kartezjanski, zeby przeiterowac po gridzie plotu
    grid_list = list(itertools.product(*[range(a), range(b)]))
    for grid_coord, image_data in zip(grid_list, images_dict.iteritems()):
        Id, (img, label) = image_data
        x,y = grid_coord
        ax = axes[x][y]
        ax.imshow(img, **imshow_args)
        ax.set_xlabel(label)
    fig.tight_layout()
    
def df_to_dict(rows):
    return {Id: (open_image_id(Id),label) for  ind, (Id, label) in rows.iterrows()} 

def get_random_dict(n, ids):
    new_ids = ids.sample(n)
    return df_to_dict(new_ids)

def view_random_samples(a,b, ids, **kwargs):
    images_dict = get_random_dict(a*b, ids)
    view_grid_images(a,b, images_dict, **kwargs)
        