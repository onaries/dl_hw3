
# coding: utf-8

# In[1]:

import numpy as np
import tensorflow as tf


# In[2]:

# variables

batch_size = 128
num_classes = 7
epochs = 100


# In[3]:

# dataset loading

dataset_file = 'dataset/fer2013/fer2013.csv'
data = np.genfromtxt(dataset_file, dtype=None, delimiter=',', skip_header=1) 


# In[4]:

labels = []
images = []
usages = []

x_train = []
y_train = []
x_test1 = []
y_test1 = []
x_test2 = []
y_test2 = []


# In[5]:

def labeling(data):
    label = np.zeros(7, dtype=np.uint8)
    if data == 0:
        label[0] = 1
    elif data == 1:
        label[1] = 1
    elif data == 2:
        label[2] = 1
    elif data == 3:
        label[3] = 1
    elif data == 4:
        label[4] = 1
    elif data == 5:
        label[5] = 1
    elif data == 6:
        label[6] = 1
    
    return label.tolist()
        


# In[6]:

for i in range(len(data)):
    if data[i][2] == b'Training':
        x_train.append(data[i][1].split(sep=b' '))
        y_train.append(labeling(data[i][0]))
    elif data[i][2] == b'PublicTest':
        x_test1.append(data[i][1].split(sep=b' '))
        y_test1.append(labeling(data[i][0]))
    else:
        x_test2.append(data[i][1].split(sep=b' '))
        y_test2.append(labeling(data[i][0]))
               


# In[7]:

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test1 = np.array(x_test1)
y_test1 = np.array(y_test1)
x_test2 = np.array(x_test2)
y_test2 = np.array(y_test2)


# In[8]:

x_train.shape[0]


# In[9]:

x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
x_test1 = x_test1.reshape(x_test1.shape[0], 48, 48, 1)
x_test2 = x_test2.reshape(x_test1.shape[0], 48, 48, 1)


# In[10]:

print(x_train.shape)
print(x_test1.shape)
print(x_test2.shape)


# In[11]:

model = tf.contrib.keras.models.Sequential()


# In[12]:

x_train.shape[0:]


# In[13]:

model.add(tf.contrib.keras.layers.Conv2D(64, (3, 3), padding='same', input_shape=(48, 48, 1)))
model.add(tf.contrib.keras.layers.Activation('relu'))


# In[14]:

model.add(tf.contrib.keras.layers.Conv2D(64, (3, 3)))
model.add(tf.contrib.keras.layers.Activation('relu'))


# In[15]:

model.add(tf.contrib.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.contrib.keras.layers.Dropout(0.25))


# In[16]:

model.add(tf.contrib.keras.layers.Conv2D(128, (3, 3), padding='same'))
model.add(tf.contrib.keras.layers.Activation('relu'))


# In[17]:

model.add(tf.contrib.keras.layers.Conv2D(128, (3, 3)))
model.add(tf.contrib.keras.layers.Activation('relu'))


# In[18]:

model.add(tf.contrib.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.contrib.keras.layers.Dropout(0.25))


# In[19]:

model.add(tf.contrib.keras.layers.Flatten())
model.add(tf.contrib.keras.layers.Dense(512))
model.add(tf.contrib.keras.layers.Activation('relu'))
model.add(tf.contrib.keras.layers.Dropout(0.5))
model.add(tf.contrib.keras.layers.Dense(num_classes))
model.add(tf.contrib.keras.layers.Activation('softmax'))


# In[20]:

# optimizer

opt = tf.contrib.keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)


# In[21]:

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


# In[22]:

early_stopping = tf.contrib.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)


# In[23]:

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test1, y_test1), shuffle=True, callbacks=[early_stopping])


# In[24]:

import matplotlib.pyplot as plt


# In[40]:

emotion = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']


# In[23]:

im = np.array(x_test1[0], dtype=np.uint8)


# In[24]:

im = im.reshape(48, 48)


# In[30]:

plt.imshow(im, cmap='gray')
plt.show()


# In[25]:

model.save('model.h5')


# In[26]:

model.save_weights('model_weights.h5')


# In[27]:

pred = model.predict_classes(x_test1, batch_size=1)


# In[34]:

len(pred)


# In[28]:

pred[:100]


# In[43]:

accu = model.evaluate(x_test1, y_test1, batch_size=128)


# In[44]:

accu


# In[51]:

accu2 = model.evaluate(x_test2, y_test2, batch_size=128)


# In[52]:

accu2


# In[58]:

im = np.array(x_test1[1], dtype=np.uint8)
im = im.reshape(48, 48)
print(y_test1[1])


# In[46]:

plt.imshow(im, cmap='gray')
plt.xlabel(emotion[pred[1]])
plt.show()


# In[57]:

im = np.array(x_test1[2], dtype=np.uint8)
im = im.reshape(48, 48)
print(y_test1[2])


# In[48]:

plt.imshow(im, cmap='gray')
plt.xlabel(emotion[pred[2]])
plt.show()


# In[56]:

im = np.array(x_test1[3], dtype=np.uint8)
im = im.reshape(48, 48)
print(y_test1[3])


# In[50]:

plt.imshow(im, cmap='gray')
plt.xlabel(emotion[pred[3]])
plt.show()


# In[53]:

im = np.array(x_test1[4], dtype=np.uint8)
im = im.reshape(48, 48)


# In[55]:

plt.imshow(im, cmap='gray')
plt.xlabel(emotion[pred[4]])
plt.show()
print(y_test1[4])


# In[ ]:



