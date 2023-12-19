

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # showing and rendering figures
# io related
from skimage.io import imread
import os
from glob import glob


#%%



age_df = pd.read_csv('boneage_file.csv')



age_df['path'] = age_df['id'].map(lambda x: os.path.join('./data/', 
                                                         '{}.bmp'.format(x)))



age_df['exists'] = age_df['path'].map(os.path.exists)

age_df['exists'] = age_df['exists'].map(lambda x: True if x else np.nan )
age_df.dropna(axis= 'index', how='any')

for i in range (0,300):
    if age_df['exists'][i] == '':
        print(age_df['path'][i])
    
    
    

print('Total:', age_df['exists'].sum(), 'images')


age_df['gender'] = age_df['sex_male(1)'].map(lambda x: 'male' if x else 'female')


boneage_mean = age_df['bone_age'].mean()
boneage_div = 2*age_df['bone_age'].std()
# we don't want normalization for now
boneage_mean = 0
boneage_div = 1.0


age_df['bone_age_zscore'] = age_df['bone_age'].map(lambda x: (x-boneage_mean)/boneage_div)
#age_df.dropna(axis= 'index', how='any')
#age_df.size
age_df.sample(3)

age_df[['bone_age', 'sex_male(1)']].hist(figsize = (10, 5))

age_df['boneage_category'] = pd.cut(age_df['bone_age'], 10)



#%%



from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(age_df, 
                                   test_size = 0.1, 
                                   stratify = age_df['boneage_category'])
'''
valid_df, test_df = train_test_split(valid_df, 
                                   test_size = 0.5, 
                                   stratify = valid_df['boneage_category'])
'''

print('train', train_df.shape[0], '\nvalidation', test_df.shape[0])


#%%

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG19
#from tensorflow.keras.applications.Resnet152 import preprocess_input

from tensorflow.keras.applications.vgg19 import preprocess_input
image_size=128

data_gen=ImageDataGenerator(
                height_shift_range=0.2,
                width_shift_range=0.2,
                horizontal_flip=True,
                vertical_flip=False,
                preprocessing_function=preprocess_input,
                zoom_range=0.2,
                validation_split=0.20
            )

data_gen_test=ImageDataGenerator(
                preprocessing_function=preprocess_input,
            )


'''
train_generator=data_gen.flow_from_dataframe(
    dataframe=age_df,
    x_col="path",
    y_col="bone_age",
    subset="training",
    batch_size=10,
    seed=42,
    shuffle=True,
    target_size=(image_size,image_size),
    class_mode='raw'
    )
'''
train_generator=data_gen.flow_from_dataframe(
    dataframe=train_df,
    x_col="path",
    y_col="bone_age",
    subset="training",
    batch_size=10,
    seed=42,
    shuffle=True,
    color_mode = 'rgb',
    target_size=(image_size,image_size),
    class_mode='raw'
    )

validation_generator=data_gen.flow_from_dataframe(
    dataframe=train_df,
    x_col="path",
    y_col="bone_age",
    subset="validation",
    batch_size=10,
    seed=42,
    shuffle=True,
    color_mode = 'rgb',
    target_size=(image_size,image_size),
    class_mode='raw'
)

test_generator=data_gen_test.flow_from_dataframe(
    dataframe=test_df,
    x_col="path",
    y_col="bone_age",
    batch_size=10,
    seed=42,
    shuffle=True,
    color_mode = 'rgb',
    target_size=(image_size,image_size),
    class_mode='raw'
)

test_X, test_Y = next(test_generator)

#%%

import cv2
t_x, t_y = next(train_generator)
fig, m_axs = plt.subplots(2, 4, figsize = (16, 8))
for (c_x, c_y, c_ax) in zip(t_x, t_y, m_axs.flatten()):
    c_ax.imshow(c_x[:,:,-1], cmap = 'bone', vmin = -100, vmax = 100)
    c_ax.set_title('%2.0f months' % (c_y*boneage_div+boneage_mean))
    c_ax.axis('off')


#%%




'''

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
model=Sequential()
model.add(VGG16(input_shape=(image_size, image_size, 3), include_top=False, weights='imagenet'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='linear'))
model.layers[0].trainable=False
model.compile(loss='mse', optimizer='adam', metrics=['MeanSquaredError'])
model.summary()




model.fit_generator(train_generator,
                   validation_data=validation_generator,
                   epochs=30,
                   )


y_pred=model.predict_generator(test_generator)
preds=y_pred.flatten()
'''

#%%

#from tensorflow.keras.applications import ResNet101

from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, Input, Conv2D, multiply, LocallyConnected2D, Lambda
from tensorflow.keras.models import Model

in_lay = Input(t_x.shape[1:])
base_pretrained_model = VGG19(input_shape =  t_x.shape[1:], include_top = False, weights = 'imagenet')
base_pretrained_model.trainable = False
pt_depth = base_pretrained_model.get_output_shape_at(0)[-1]
pt_features = base_pretrained_model(in_lay)
from tensorflow.keras.layers import BatchNormalization
bn_features = BatchNormalization()(pt_features)

#custom

attn_layer = Conv2D(64, kernel_size = (3,3), padding = 'same', activation = 'relu')(bn_features)
attn_layer  = BatchNormalization()(bn_features)
attn_layer = Conv2D(16, kernel_size = (3,3), padding = 'same', activation = 'relu')(attn_layer)
attn_layer  = BatchNormalization()(attn_layer)
attn_layer = LocallyConnected2D(1, 
                                kernel_size = (1,1), 
                                padding = 'valid', 
                                activation = 'sigmoid')(attn_layer)
#combine

up_c2_w = np.ones((1, 1, 1, pt_depth))
up_c2 = Conv2D(pt_depth, kernel_size = (1,1), padding = 'same', 
               activation = 'linear', use_bias = False, weights = [up_c2_w])
up_c2.trainable = False
attn_layer = up_c2(attn_layer)

mask_features = multiply([attn_layer, bn_features])
gap_features = GlobalAveragePooling2D()(mask_features)
gap_mask = GlobalAveragePooling2D()(attn_layer)
# to account for missing values from the attention model
gap = Lambda(lambda x: x[0]/x[1], name = 'RescaleGAP')([gap_features, gap_mask])
gap_dr = Dropout(0.5)(gap)
dr_steps = Dropout(0.25)(Dense(512, activation = 'relu')(gap_dr))
out_layer = Dense(1, activation = 'linear')(dr_steps) # linear is what 16bit did
bone_age_model = Model(inputs = [in_lay], outputs = [out_layer])
from tensorflow.keras.metrics import mean_absolute_error



def mae_months(in_gt, in_pred):
    return mean_absolute_error(boneage_div*in_gt, boneage_div*in_pred)

bone_age_model.compile(optimizer = 'adam', loss = 'mse',
                           metrics = [mae_months])

bone_age_model.summary()
#%%


from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
weight_path="VGG19.hdf5".format('bone_age')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)


reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)
early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=15) # probably needs to be more patient, but kaggle time is limited
callbacks_list = [checkpoint, early, reduceLROnPlat]


history=bone_age_model.fit_generator(train_generator, 
                                  steps_per_epoch=50,
                                  validation_data = (test_X, test_Y), 
                                  epochs = 50, 
                                  callbacks = callbacks_list)



#%%




def plot_it(history):
    '''function to plot training and validation error'''
    fig, ax = plt.subplots( figsize=(20,10))
    ax.plot(history.history['mae_months'])
    ax.plot(history.history['val_mae_months'])
    plt.title('Model Error')
    plt.ylabel('error')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper right')
    ax.grid(color='black')
    plt.show()
    
    
import matplotlib.pyplot as plt
plot_it(history)
    
    
    
    
#%%

bone_age_model.save('VGG19.h5')
bone_age_model.load_weights(weight_path)    
    
    
#%%



for attn_layer in bone_age_model.layers:
    c_shape = attn_layer.get_output_shape_at(0)
    if len(c_shape)==4:
        if c_shape[-1]==1:
            print(attn_layer)
            break
#%%





import tensorflow.keras.backend as K
rand_idx = np.random.choice(range(len(test_X)), size = 6)
attn_func = K.function(inputs = [bone_age_model.get_input_at(0), K.learning_phase()],
           outputs = [attn_layer.get_output_at(0)]
          )
fig, m_axs = plt.subplots(len(rand_idx), 2, figsize = (8, 4*len(rand_idx)))
[c_ax.axis('off') for c_ax in m_axs.flatten()]
for c_idx, (img_ax, attn_ax) in zip(rand_idx, m_axs):
    cur_img = test_X[c_idx:(c_idx+1)]
    attn_img = attn_func([cur_img, 0])[0]
    img_ax.imshow(cur_img[0,:,:,0], cmap = 'bone')
    attn_ax.imshow(attn_img[0, :, :, 0], cmap = 'viridis', 
                   vmin = 0, vmax = 1, 
                   interpolation = 'lanczos')
    real_age = boneage_div*test_Y[c_idx]+boneage_mean
    img_ax.set_title('Hand Image\nAge:%2.2fY' % (real_age/12))
    pred_age = boneage_div*bone_age_model.predict(cur_img)+boneage_mean
    attn_ax.set_title('Attention Map\nPred:%2.2fY' % (pred_age/12))
fig.savefig('attention_map.png', dpi = 300)



#%%





pred_Y = boneage_div*bone_age_model.predict(test_X, batch_size = 32, verbose = True)+boneage_mean
test_Y_months = boneage_div*test_Y+boneage_mean



fig, ax1 = plt.subplots(1,1, figsize = (6,6))
ax1.plot(test_Y_months, pred_Y, 'r.', label = 'predictions')
ax1.plot(test_Y_months, test_Y_months, 'b-', label = 'actual')
ax1.legend()
ax1.set_xlabel('Actual Age (Months)')
ax1.set_ylabel('Predicted Age (Months)')





ord_idx = np.argsort(test_Y)
ord_idx = ord_idx[np.linspace(0, len(ord_idx)-1, 8).astype(int)] # take 8 evenly spaced ones
fig, m_axs = plt.subplots(4, 2, figsize = (16, 32))
for (idx, c_ax) in zip(ord_idx, m_axs.flatten()):
    c_ax.imshow(test_X[idx, :,:,0], cmap = 'bone')
    
    c_ax.set_title('Age: %2.1fY\nPredicted Age: %2.1fY' % (test_Y_months[idx]/12.0, 
                                                           pred_Y[idx]/12.0))
    c_ax.axis('off')
fig.savefig('trained_img_predictions.png', dpi = 300)








