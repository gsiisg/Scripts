import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import pickle
import time

start_time = time.time()

#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# initializes place holders
pixel_count = 640*480
x = tf.placeholder(tf.float32, shape=[None, pixel_count])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# initializes 'Variable' to have the dimensions desired
W = tf.Variable(tf.zeros([pixel_count,10]))
b = tf.Variable(tf.zeros([10]))

# start session
#sess.run(tf.initialize_all_variables())

# matrix multiplication
y = tf.nn.softmax(tf.matmul(x,W) + b)

# cross_entropy tells us how bad the model is doing
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# optimize the model by minimizing the cross entropy
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

'''
# feeding in training data
for i in range(1000):
  batch = mnist.train.next_batch(50)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})
'''

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

driver_list = pd.read_csv('input/driver_imgs_list.csv')
classname = driver_list['classname'].values
filename = driver_list['img'].values
number_of_images = len(classname)
shuffle_index = np.arange(number_of_images)
#np.random.seed(int(time.time()))
np.random.seed(1234)
np.random.shuffle(shuffle_index)
#shuffle_index = pickle.load(open('shuffle_index','rb'))
print('last index is %i' % shuffle_index[-1])
classname = classname[shuffle_index]
filename = filename[shuffle_index]

# doing k fold cross validation
k=4
list_size = len(classname)
split_size = list_size/k
kth = 0
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

#print(test_size, train_size, test_size+train_size, test_classname[0])


batch_size = 1000
batch_number = 17

train_start = time.time()

for i in range(batch_number):
  image_batch = np.empty([batch_size,pixel_count])
  label_batch = np.empty([batch_size,10])
  for j in range(batch_size):
      index = i*batch_size+j
      if index < train_size:
        train_class = train_classname[index]
        train_file = train_filename[index]
        image_batch[j] = cv2.imread('input/'+train_class+'/'+train_file,0)
        label_batch[j][int(train_class.split('c')[1])] = 1
  sess.run(train_step, feed_dict={x: image_batch, y_: label_batch})
  print('processed batch %i' % i)

train_duration = time.time() - train_start
print('train time %f' % train_duration)

# calculate the correct predictions
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# calculate the accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# output the accuracy
#print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

test_image_batch = np.empty([test_size,pixel_count])
test_label_batch = np.empty([test_size,10])
for i in range(test_size):
  test_class = test_classname[i]
  test_file = test_filename[i]
  test_image_batch[i] = cv2.imread('input/'+test_class+'/'+test_file,0)
  test_label_batch[i][int(test_class.split('c')[1])] = 1
  if i%1000 == 0:
    print('read in %i tests' % i)

eval_start = time.time()
print(sess.run(accuracy, feed_dict={x: test_image_batch, y_: test_label_batch}))
eval_duration = time.time()-eval_start
print('eval test %f' % eval_duration)

print(time.time() - start_time)
#pickle.dump(shuffle_index,open('shuffle_index','wb'))
