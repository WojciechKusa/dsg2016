import numpy as np
from skimage.io import imread
import os
import pandas as pd
from keras.utils import np_utils



def _maybe_float32(fun):
    """
    Don't overthink this
    """
    def wrapper(*args):
        if args[0].as_float32:
            return fun(*args).astype(np.float32)
        else: 
            return fun(*args)
    return wrapper
    
class Dataset(object):
    
    def _run_if_not_exist(self, obj_name, meth):
        if getattr(self, obj_name) is None:
            meth()
        return getattr(self, obj_name)
    
    
    @staticmethod
    def _create_image_opened(images_folder):   
        default_extension = '.bmp'
        def open_image_id(image_id, extension=None):
            if extension == None:
                extension = default_extension
            img_name = str(image_id)
            image_path = os.path.join(images_folder,img_name+extension)
            image = imread(image_path)
            return image
        return open_image_id
    
    def __init__(self, images_folder, ids_path, train_percent=0.7, as_float32=False):
        self.as_float32 = as_float32
        self.images_folder = images_folder
        self.ids_path = ids_path
        self.ids = pd.read_csv(ids_path)
        self.train_percent = train_percent
        self.nb_classes = 4
        self._X_train, self._Y_train = None, None
        self._X_test, self._Y_test = None, None
        self._X_data = None
        self._Y_data = None
        self._X_submition = None
        self.train_ind = None
        self.test_ind = None
        self.open_image_id = self._create_image_opened(self.images_folder)
    
    def set_as_float32(self):
        self.as_float32 = True
        self._X_data = self._X_data.astype(np.float32)
    
    @property
    def d_size(self):
        return len(self.ids)
    
    def read_dataset(self):
        print "Reading data..."
        X_data = []
        Y_data = []
        for ind, (Id, label) in self.ids.iterrows():
            X_data.append(self.open_image_id(Id))
            Y_data.append(label)
        self._X_data = np.array(X_data)
        self._Y_data = np.array(Y_data)
        self._Y_data = np_utils.to_categorical(self._Y_data-1, self.nb_classes)
        if self.as_float32:
            self._X_data = self._X_data.astype(np.float32)
        print "Done."
        
    def split_dataset(self,seed=1337):
        np.random.seed(seed)
        rnd = np.random.rand(self.d_size)
        self.train_ind = rnd<self.train_percent
        self.test_ind = ~self.train_ind
        #X, Y = self.X_data, self.Y_data
        #self._X_train, self._Y_train = map(lambda o: o[self.train_ind], (X,Y))
        #self._X_test, self._Y_test = map(lambda o: o[self.test_ind], (X,Y))

    @staticmethod
    def apply_each_row(func, data):
        print "Applying function to data..."
        rval = [func(data[i]) for i in xrange(len(data))]
        print "Applied."
        return rval
       
    def apply_each_X_data_row(self, func):
        xtmp = self.apply_each_row(func, self.X_data)
        del self._X_data
        self._X_data = np.array(xtmp)
        self.split_dataset()
       
    def apply_each_X_submition_row(self, func):
        xtmp = self.apply_each_row(func, self.X_submition)
        del self._X_submition
        self._X_submition = None
        self._X_submition = np.array(xtmp)

    
    
    
    @property
    @_maybe_float32
    def X_data(self):
        return self._run_if_not_exist('_X_data',self.read_dataset)
        
    @property
    def Y_data(self):
        return self._run_if_not_exist('_Y_data',self.read_dataset)
        
    @property
    def Y_train(self):
        train_ind = self._run_if_not_exist('train_ind',self.split_dataset)
        return self.Y_data[train_ind]
        
    
    @property
    @_maybe_float32
    def X_train(self):
        train_ind = self._run_if_not_exist('train_ind',self.split_dataset)
        return self.X_data[train_ind]
        
    
    @property
    @_maybe_float32
    def X_test(self):
        test_ind = self._run_if_not_exist('test_ind',self.split_dataset)
        return self.X_data[test_ind]
    
    @property
    def Y_test(self):
        test_ind = self._run_if_not_exist('test_ind',self.split_dataset)
        return self.Y_data[test_ind]
    
    def read_submition(self, submition_ids_path, submition_pics_folder):
        self.submition_pics_folder = submition_pics_folder
        self.open_submition_id = self._create_image_opened(self.submition_pics_folder)
        self.submition_ids_path = submition_ids_path
        self.submition_ids = pd.read_csv(submition_ids_path)
        X_submition = []
        for ind, (Id, label) in self.submition_ids.iterrows():
            X_submition.append(self.open_submition_id(Id,extension='.jpg'))
        self._X_submition = np.array(X_submition)

    
    @property
    @_maybe_float32
    def X_submition(self):
        if self._X_submition is None:
            raise AttributeError('Use \'read_submition()\' first to read submition data.')
        else:
            return self._X_submition