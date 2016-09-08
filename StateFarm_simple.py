import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import time
import sys

start_time = time.time()

#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def DownSample0x(image):
    return image

def DownSample2x(image):
    s2x = image
    s2x = cv2.pyrDown(image, s2x, (320,240))
    return s2x

def DownSample4x(image):
    s2x = image
    s2x = cv2.pyrDown(image, s2x, (320,240))
    s4x = s2x
    s4x = cv2.pyrDown(s2x, s4x, (160,120))
    return s4x

def DownSample8x(image):
    s2x = image
    s2x = cv2.pyrDown(image, s2x, (320,240))
    s4x = s2x
    s4x = cv2.pyrDown(s2x, s4x, (160,120))
    s8x = s4x
    s8x = cv2.pyrDown(s4x, s8x, (80,60))
    return s8x

def DownSample16x(image):
    s2x = image
    s2x = cv2.pyrDown(image, s2x, (320,240))
    s4x = s2x
    s4x = cv2.pyrDown(s2x, s4x, (160,120))
    s8x = s4x
    s8x = cv2.pyrDown(s4x, s8x, (80,60))
    s16x = s8x
    s16x = cv2.pyrDown(s8x, s16x, (40,30))
    return s16x

def down_sample_by(image, times):
    if times == 0:
        return image
    else:
        height = len(image)
        width = len(image[0])
        for i in range(times):
            height /= 2
            width /= 2
            image = cv2.pyrDown(image, image, (width,height))
        return image


down_sample_times = 2
lr = 5e-4
ep = 1e-10
k = 100
kth = 88
# initializes place holders
pixel_count = 640*480/((2 ** down_sample_times) ** 2)
#pixel_count = 640*480/16
x = tf.placeholder(tf.float32, shape=[None, pixel_count])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# initializes 'Variable' to have the dimensions desired
W = tf.Variable(tf.zeros([pixel_count,10]))
b = tf.Variable(tf.zeros([10]))

# start session
#sess.run(tf.initialize_all_variables())

# matrix multiplication
y = tf.nn.softmax(tf.matmul(x,W) + b)
# help print out y
y = tf.Print(y, [y], "prediction")

# cross_entropy tells us how bad the model is doing
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y,1e-10,1.0)),
                                              reduction_indices=[1]))
cross_entropy = tf.Print(cross_entropy, [cross_entropy], "cost")

# optimize the model by minimizing the cross entropy
#train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
#train_step = tf.train.AdagradOptimizer(0.01).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(lr, epsilon=ep).minimize(cross_entropy)

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

driver_list = pd.read_csv('input/driver_imgs_list.csv')
classname = driver_list['classname'].values
filename = driver_list['img'].values
number_of_images = len(classname)
shuffle_index = np.arange(number_of_images)
#np.random.seed(int(time.time()))


np.random.seed(1234)
np.random.shuffle(shuffle_index)
np.save('numpy_arr', shuffle_index)
print('last index is %i' % shuffle_index[-1])
sys.stdout.flush()
#shuffle_index = np.load('numpy_arr.npy')
classname = classname[shuffle_index]
filename = filename[shuffle_index]


# doing k fold cross validation

list_size = len(classname)
split_size = list_size/k

test_begin_index = kth*split_size
test_end_index = (kth+1)*split_size
train_begin1_index = 0
train_end1_index = kth*split_size
train_begin2_index = (kth+1)*split_size
train_end2_index = list_size
test_classname = classname[test_begin_index:test_end_index]
test_filename = filename[test_begin_index:test_end_index]
train_classname = np.append(classname[train_begin1_index:train_end1_index],
                            classname[train_begin2_index:train_end2_index])
train_filename = np.append(filename[train_begin1_index:train_end1_index],
                           filename[train_begin2_index:train_end2_index])
train_size = len(train_classname)
test_size = len(test_classname)
print(test_begin_index,test_end_index)
print(train_begin1_index, train_end1_index, train_begin2_index, train_end2_index)
sys.stdout.flush()
#print(test_size, train_size, test_size+train_size, test_classname[0])


batch_size = 100
batch_number, _ = divmod(train_size, batch_size)
batch_number += 1

train_start = time.time()

for i in range(batch_number):
    image_batch = np.zeros([batch_size,pixel_count])
    label_batch = np.zeros([batch_size,10])
    for j in range(batch_size):
        index = i*batch_size+j
        if index < train_size:
            train_class = train_classname[index]
            train_file = train_filename[index]
            image_batch[j] = (down_sample_by(cv2.imread('input/train/'+train_class+'/'+train_file,0),down_sample_times)/255.0).flatten()
            label_batch[j][int(train_class.split('c')[1])] = 1
    train_step.run(feed_dict={x: image_batch, y_: label_batch})
    print('processed batch %i, index %i, last filename %s' % (i,index,train_filename[-1]))
    sys.stdout.flush()

train_duration = time.time() - train_start
print('train time %f s' % train_duration)
sys.stdout.flush()

# calculate the correct predictions
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# calculate the accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# output the accuracy
#print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

test_image_batch = np.zeros([test_size,pixel_count])
test_label_batch = np.zeros([test_size,10])
for i in range(test_size):
    test_class = test_classname[i]
    test_file = test_filename[i]
    test_image_batch[i] = (down_sample_by(cv2.imread('input/train/'+test_class+'/'+test_file,0),down_sample_times)/255.0).flatten()
    test_label_batch[i][int(test_class.split('c')[1])] = 1
    if i%10 == 0:
        print('read in %i tests %s' % (i,test_filename[-1]))
        sys.stdout.flush()

eval_start = time.time()
print(sess.run(accuracy, feed_dict={x: test_image_batch, y_: test_label_batch}))
sys.stdout.flush()
eval_duration = time.time()-eval_start
print('eval test %f s' % eval_duration)

print('calculating prediction array')
sys.stdout.flush()
prediction = y
prediction_array = prediction.eval(feed_dict={x: test_image_batch}, session=sess)

np.save('prediction_array', prediction_array)

print('total time %f s' % (time.time() - start_time))
sys.stdout.flush()
