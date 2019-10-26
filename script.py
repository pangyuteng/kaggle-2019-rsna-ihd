#************************************************************
vae_weight_file = 'vae_conv2d.h5'
mlp_weight_file = 'mlp.h5'
verbose = 1
# network parameters
DIM = (64,64)
input_shape = DIM+(1,)
batch_size = 2 #*********************************** maybe too big
latent_dim = 64 #*********************************** alsomaybe too big
epochs =  2 #*********************************** increase later
steps_per_epoch = 3 #*********************************** increase later
validation_steps = 3
#************************************************************
mlp_epochs = 2
mlp_batch_size = 2
mlp_steps_per_epoch = 3
mlp_validation_steps = 3
#************************************************************
import os
os.makedirs('~/.kaggle',exist_ok=True)
with open('~/.kaggle/kaggle.json','w') as f:
    f.write('{"username":"pangyuteng","key":"cd192c3df69daeca70484e6a0586ceb1"}')
    
import subprocess
out=subprocess.check_output(['pip','install','kaggle'])
print(out)

import traceback
import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# https://gist.github.com/pangyuteng/7f54dbfcd67fb9d43a85f8c6818fca7b
import SimpleITK as sitk

def imread(fpath):
    reader= sitk.ImageFileReader()
    reader.SetFileName(fpath)
    img = reader.Execute()
    arr = sitk.GetArrayFromImage(img)    
    spacing = img.GetSpacing()
    origin = img.GetOrigin()
    direction = img.GetDirection()    
    return arr,spacing,origin,direction

def imwrite(fpath,arr,spacing,origin,direction,use_compression=True):
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    img.SetDirection(direction)
    writer = sitk.ImageFileWriter()    
    writer.SetFileName(fpath)
    writer.SetUseCompression(use_compression)
    writer.Execute(img)
    
# https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
def chunkify(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

    
seed_value= 0
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
import random
random.seed(seed_value)
import numpy as np
np.random.seed(seed_value)
import tensorflow as tf
tf.random.set_seed(seed_value)


# https://towardsdatascience.com/keras-data-generators-and-how-to-use-them-b69129ed779c

import numpy as np
import cv2
from keras.utils import Sequence
import scipy.ndimage as ndi

class DataGenerator(Sequence):
    def __init__(self, image_path_list,to_fit=True, batch_size=32, dim=(64,64),
                 n_channels=1, shuffle=True):
        self.list_IDs = [x for x in range(len(image_path_list))]
        self.image_path_list = image_path_list
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X = self._generate_x(list_IDs_temp)

        if self.to_fit:
            return X, X
        else:
            return X

    def _generate_x(self, list_IDs_temp):
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            X[i,:] = self._load_grayscale_image(self.image_path_list[ID])

        return X
    
    def _load_grayscale_image(self, image_path):
        img,spacing,origin,direction = imread(image_path)
        img = img.astype(np.float32)
        img = np.squeeze(img)
        factor = np.array(self.dim)/np.array(img.shape)
        img = ndi.zoom(img,factor,order=3, mode='constant', cval=-1000)
        img = np.expand_dims(img,axis=-1)        
        
        # https://radiopaedia.org/articles/ct-head-subdural-window-1?lang=us
        # window-level,center 100, window 400
        minval = 100-200.
        maxval = 100+200.
        img = (img-minval)/(maxval-minval)
        # clip
        img = np.clip(img,0.,1.)
        return img
        # *************************************************
        
        
        
class LatentDataGenerator(DataGenerator):
    def __init__(self,encoder,ydf,*args,enc_batch_size=32,latent_dim=64,**kwargs):
        super().__init__(*args,**kwargs)
        
        self.ydf = ydf
        self.encoder = encoder
        self.latent_dim = latent_dim
        self.enc_batch_size = enc_batch_size
        
    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        Z,Y = self._generate(list_IDs_temp)
        if self.to_fit:
            return Z, Y
        else:
            return Z
        
    def _generate(self, list_IDs_temp):
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        Y = np.empty((self.batch_size, 6))
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            X[i,:] = self._load_grayscale_image(self.image_path_list[ID])
            actual_id = self.image_path_list[ID].split('ID_')[-1].split('.')[0]
            ys = self.ydf[self.ydf['ID']==actual_id].values.squeeze()[1:]
            Y[i,:] = ys
        encoderoutput = self.encoder.predict(X,batch_size=self.enc_batch_size)
        Z = encoderoutput[2]
        return Z, Y
    

from keras import backend as K
K.clear_session()

# https://keras.io/examples/variational_autoencoder/
from keras.layers import Lambda, Input, Dense, Flatten
from keras.layers import LeakyReLU
from keras.models import Model
from keras.layers import Conv2D,MaxPooling2D,Reshape,Conv2DTranspose,UpSampling2D
from keras.datasets import mnist
from keras.optimizers import Adam
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os



# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim)) # *********** vary the noise level
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

inputs = Input(input_shape)
x = Conv2D(64, 3, activation = 'linear', padding = 'same', kernel_initializer = 'he_normal')(inputs)
x = LeakyReLU()(x)
x = Conv2D(64, 3, activation = 'linear', padding = 'same', kernel_initializer = 'he_normal')(x)
x = LeakyReLU()(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(128, 3, activation = 'linear', padding = 'same', kernel_initializer = 'he_normal')(x)
x = LeakyReLU()(x)
x = Conv2D(128, 3, activation = 'linear', padding = 'same', kernel_initializer = 'he_normal')(x)
x = LeakyReLU()(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(256, 3, activation = 'linear', padding = 'same', kernel_initializer = 'he_normal')(x)
x = LeakyReLU()(x)
x = Conv2D(256, 3, activation = 'linear', padding = 'same', kernel_initializer = 'he_normal')(x)
x = LeakyReLU()(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
nonflattenshape = np.array(x.get_shape()[1:]).astype(np.int)
x = Flatten()(x)
flatten_shape = int(np.prod(nonflattenshape))

x = Dense(1024, activation='linear')(x)
x = LeakyReLU()(x)
x = Dense(1024, activation='linear')(x)
x = LeakyReLU()(x)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()

latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(1024, activation='linear')(latent_inputs)
x = LeakyReLU()(x)
x = Dense(1024, activation='linear')(x)
x = LeakyReLU()(x)
x = Dense(flatten_shape, activation='linear')(x)
x = LeakyReLU()(x)
x = Reshape(nonflattenshape)(x)
x = Conv2DTranspose(256,3, activation='linear', padding='same', kernel_initializer='he_normal')(x)
x = LeakyReLU()(x)
x = Conv2DTranspose(256,3, activation='linear', padding='same', kernel_initializer='he_normal')(x)
x = LeakyReLU()(x)
x = UpSampling2D(size=(2, 2))(x)
x = Conv2DTranspose(128,3, activation = 'linear', padding = 'same', kernel_initializer = 'he_normal')(x)
x = LeakyReLU()(x)
x = Conv2DTranspose(128,3, activation = 'linear', padding = 'same', kernel_initializer = 'he_normal')(x)
x = LeakyReLU()(x)
x = UpSampling2D(size=(2, 2))(x)
x = Conv2DTranspose(64,3, activation = 'linear', padding = 'same', kernel_initializer = 'he_normal')(x)
x = LeakyReLU()(x)
x = Conv2DTranspose(64,3, activation = 'linear', padding = 'same', kernel_initializer = 'he_normal')(x)
x = LeakyReLU()(x)
x = UpSampling2D(size=(2, 2))(x)
x = Conv2DTranspose(32,3, activation = 'linear', padding = 'same', kernel_initializer = 'he_normal')(x)
x = LeakyReLU()(x)
outputs = Conv2D(1,3, activation = 'linear', padding = 'same', kernel_initializer = 'he_normal')(x)
# instantiate decoder model
decoder = Model(latent_inputs,outputs, name='decoder')
decoder.summary()

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='con2d_vae')
vae.summary()

models = (encoder, decoder)

def vae_loss(input_shape,z_mean,z_log_var):
    def loss(y_true,y_pred):
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        reconstruction_loss = K.mean(K.square(y_pred - y_true), axis=[1,2,3]) * np.prod(input_shape)
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        return vae_loss
    return loss


optimizer = Adam(learning_rate=1e-3,)
vae.compile(optimizer=optimizer,loss=vae_loss(input_shape,z_mean,z_log_var))
vae.summary()


from keras.models import Sequential
from keras.layers import Dropout, Dense
from keras.layers import LeakyReLU, BatchNormalization
from keras.optimizers import RMSprop,Adam

mlp = Sequential()
mlp.add(Dense(32, activation='relu', input_shape=(latent_dim,)))
mlp.add(BatchNormalization())
mlp.add(LeakyReLU())
mlp.add(Dense(16, activation='relu'))
mlp.add(BatchNormalization())
mlp.add(LeakyReLU())
mlp.add(Dense(16, activation='relu'))
mlp.add(BatchNormalization())
mlp.add(LeakyReLU())
mlp.add(Dense(6, activation='sigmoid')) # softmax
mlp.summary()

mlp.compile(loss='binary_crossentropy', # ?? categorical_crossentropy',
              optimizer=RMSprop(learning_rate=1E-4),
              metrics=['accuracy'])


print('Preparing Y...')
if os.path.exists('ydf.csv'):
    ydf = pd.read_csv('ydf.csv')
else:
    # generate Y 
    # generate dfs per kind, then join them to create the "Y" of the X,Y for the final training dataset.
    df = pd.read_csv('/kaggle/input/rsna-intracranial-hemorrhage-detection/stage_1_train.csv')
    df['KIND']= df.apply(lambda x: x['ID'].split('_')[2],axis=1)
    df['ID']= df.apply(lambda x: x['ID'].split('_')[1],axis=1)
    df.index=df['ID']
    dfs = []
    for col in list(df['KIND'].unique()):
        tmp = df[df['KIND']==col]
        tmp = tmp.rename(columns={"Label": col})
        tmp = tmp.drop(columns='KIND')
        tmp = tmp.drop(columns='ID')
        dfs.append(tmp)
        
    ydf = None
    for n,col in enumerate(list(df['KIND'].unique())):
        if n == 0:
            ydf = dfs[n]
        else:
            ydf = ydf.join(dfs[n],how='left',on=['ID'])
            
    pd.DataFrame(ydf).to_csv('ydf.csv')
    
# ******************
# TODO: reduce datatset to increase training
# balance the label
# redue "negative" scans
# eliminate "bad" scans

print('Preparing Y... Done')

KINDENUM = ['epidural','intraparenchymal','intraventricular','subarachnoid','subdural','any']

from sklearn.model_selection import train_test_split
sampledf=pd.read_csv('/kaggle/input/rsna-intracranial-hemorrhage-detection/stage_1_sample_submission.csv')
trainstage1 = list(set([x.split('_')[1] for x in sampledf['ID']]))
test_file_path_list =  ['/kaggle/input/rsna-intracranial-hemorrhage-detection/stage_1_test_images/ID_{}.dcm'.format(x) for x in trainstage1]
train_val_file_path_list = ['/kaggle/input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/ID_{}.dcm'.format(x) for x in ydf['ID']]
train_file_path_list, validation_file_path_list, _, _ = train_test_split(train_val_file_path_list,train_val_file_path_list,test_size=0.1, random_state=42)

    
try:
    train_dg = DataGenerator(train_file_path_list)
    validation_dg = DataGenerator(validation_file_path_list[-2:])

    if os.path.exists(vae_weight_file):
        vae.load_weights(vae_weight_file)
    else:
        print('training vae..............')
        history = vae.fit_generator(train_dg,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                verbose=verbose,
                validation_steps=validation_steps,
                validation_data=validation_dg)
        vae.save_weights(vae_weight_file)
        #score = vae.evaluate_generator()
        print(history)
        
    train_ldg = LatentDataGenerator(encoder,ydf,train_file_path_list[:1000],latent_dim=latent_dim,enc_batch_size=mlp_batch_size)
    validation_ldg = LatentDataGenerator(encoder,ydf,train_file_path_list[-1000:],latent_dim=latent_dim,enc_batch_size=mlp_batch_size)
    if os.path.exists(mlp_weight_file):
        mlp.load_weights(mlp_weight_file)
    else:
        print('training mlp..............')
        history = mlp.fit_generator(train_ldg,
                        epochs=mlp_epochs,
                        steps_per_epoch=mlp_steps_per_epoch,
                        verbose=verbose,
                        validation_steps=mlp_validation_steps,
                        validation_data=validation_ldg)
        mlp.save_weights(mlp_weight_file)
        #score = mlp.evaluate_generator()
        print(history)
        
        
    batch_size=2048
    tmp_file_path_list = test_file_path_list
    x_test=np.zeros((len(tmp_file_path_list),latent_dim))
    y_test=np.zeros((len(tmp_file_path_list),6))
    test_dg = DataGenerator(tmp_file_path_list,batch_size=batch_size,shuffle=False)
    
    chunks = [x for x in chunkify([x for x in range(len(tmp_file_path_list))],batch_size)]
    print(len(chunks))
    for lst in tqdm(chunks):
        tmp = test_dg._generate_X(lst)
        pred = encoder.predict(tmp)
        flist = [tmp_file_path_list[x] for x in lst]
        out = model.predict(pred[2])
        for n,index in enumerate(lst):
            y_test[index,:]=out[n,:]
        my_submission = []
        for index,theid in enumerate([x.split('ID_')[-1].split('.')[0] for x in test_file_path_list]):
            _predicted_ = y_test[index,:]
            for n,kind in enumerate(KINDENUM):
                predicted = _predicted_[n]
                my_submission.append({
                    'ID':'ID_{}_{}'.format(theid,kind),
                    'Label':predicted,
                })
    mysdf = pd.DataFrame(my_submission)
    mysdf.to_csv('submission.csv', index=False)
    
    out=subprocess.check_output(['kaggle','competitions','submit','-c','rsna-intracranial-hemorrhage-detection','-f','submission.csv','-m','init - script'])
    print(out)
except:
    traceback.print_exc()
    print('errrrrrrrrrrrrrrrrrrrrrr')

print('done')

