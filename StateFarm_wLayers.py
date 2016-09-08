import pandas as pd
import numpy as np
import cv2
import time
import sys
import glob
import tensorflow as tf

start_time = time.time()

sess = tf.InteractiveSession()

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# down sampling non-linearly picking the max out of 2x2 matrix of the convolution
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

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

def train_test_sets_k_fold(csvFile, k, kth):
    driver_list = pd.read_csv(csvFile)
    classname = driver_list['classname'].values
    filename = driver_list['img'].values
    number_of_images = len(classname)
    shuffle_index = np.arange(number_of_images)
    #np.random.seed(int(time.time()))
    np.random.seed(seed)
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
    # determine the index
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
    print('test:',test_begin_index,test_end_index,'test_size:',len(test_classname))
    print('train:',train_begin1_index,train_end1_index,train_begin2_index,train_end2_index,
          'train_size:',len(train_classname))
    sys.stdout.flush()
    return train_classname, train_filename, test_classname, test_filename

def read_image(filename):
    return cv2.imread(filename, 0)/255.0

def train_and_test(csvFile):
    driver_list = pd.read_csv(csvFile)
    train_classname = driver_list['classname'].values
    train_filename = driver_list['img'].values
    number_of_images = len(train_classname)
    shuffle_index = np.arange(number_of_images)
    np.random.seed(seed)
    np.random.shuffle(shuffle_index)
    np.save('numpy_arr', shuffle_index)
    train_classname = train_classname[shuffle_index]
    train_filename = train_filename[shuffle_index]
    test_filename = glob.glob('input/test/*.jpg')
    return train_classname, train_filename, test_filename

#img = cv2.imread("image.jpg")
#cv2.startWindowThread()
#cv2.namedWindow("preview")
#cv2.imshow("preview", img)
#cv2.destroyAllWindows()

# initializes place holders
mode = 'test' # 'validate' or real 'test'
#mode = 'validate' # 'validate' or real 'test'
if mode == 'test':
    test_batch_size = 100
seed = 1234
lr = 1e-4
#lr = 1e-4
ep = 1e-10
k = 100
kth = 77
batch_size = 100
downsample_times = 3
downsample_x = 640 / (2 ** downsample_times)
downsample_y = 480 / (2 ** downsample_times)
pixel_count = 640 * 480 / ((2 ** downsample_times) ** 2)

x = tf.placeholder(tf.float32, shape=[None, pixel_count])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

x_image = tf.reshape(x, [-1,downsample_y,downsample_x,1])

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

fully_connected_dimension = np.int(downsample_x * downsample_y / 16.0)
print(downsample_x,downsample_y,fully_connected_dimension)
sys.stdout.flush()
W_fc1 = weight_variable([fully_connected_dimension * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, fully_connected_dimension * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
y_conv = tf.Print(y_conv, [y_conv], "prediction")

# initializes 'Variable' to have the dimensions desired
W = tf.Variable(tf.zeros([pixel_count, 10]))
b = tf.Variable(tf.zeros([10]))

# start session
saver = tf.train.Saver()
sess.run(tf.initialize_all_variables())

# matrix multiplication
#y = tf.nn.softmax(tf.matmul(x, W) + b)


# cross_entropy tells us how bad the model is doing
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ *
                tf.log(tf.clip_by_value(y_conv,1e-10,1.0)), reduction_indices=[1]))

cross_entropy = tf.Print(cross_entropy, [cross_entropy], "cost")
train_step = tf.train.AdamOptimizer(lr, epsilon=ep).minimize(cross_entropy)
#train_step = tf.train.FtrlOptimizer(lr).minimize(cross_entropy)


# learn_rate=5e-4 epsilon=1e-8 accuracy=18% loss=8 downsample_times=3
# learn_rate=1e-3 epsilon=1e-8 accuracy=15% loss=8 downsample_times=3
# learn_rate=1e-4 epsilon=1e-8 accuracy=26% loss=5 downsample_times=3
# learn_rate=1e-4 epsilon=1e-8 accuracy=90% loss=1 downsample_times=2 batch_size=50 k=100
# learn_rate=1e-5 epsilon=1e-8 accuracy=52% loss=3 downsample_times=2 batch_size=50 k=100
# learn_rate=1e-6 epsilon=1e-8 accuracy=8% loss=16 downsample_times=3 batch_size=50 k=100
# learn_rate=1e-6 epsilon=1e-10 accuracy=11% loss=14 downsample_times=3 batch_size=50 k=100
# learn_rate=1e-4 epsilon=1e-10 accuracy=13% loss=16 downsample_times=3 batch_size=100 k=100
# learn_rate=1e-4 epsilon=1e-8 accuracy=75% loss=1 downsample_times=2 batch_size=100 k=100,33
# learn_rate=1e-4 epsilon=1e-8 accuracy=89% loss=0.8 downsample_times=2 batch_size=100 k=100,77
# learn_rate=1e-4 epsilon=1e-8 accuracy=94% loss=0.6 downsample_times=2 batch_size=100 k=100,22
# learn_rate=1e-4 epsilon=1e-8 accuracy=92% loss=0.6 downsample_times=2 batch_size=100 k=100,44
# learn_rate=1e-4 epsilon=1e-8 accuracy=79% loss=1.2 downsample_times=2 batch_size=100 k=100,88
# learn_rate=1e-4 epsilon=1e-8 accuracy=80% loss=0.95 downsample_times=3 batch_size=100 k=100,55
# learn_rate=1e-4 epsilon=1e-8 accuracy=80% loss=1.2 downsample_times=3 batch_size=100 k=100,11
# learn_rate=1e-4 epsilon=1e-8 accuracy=79% loss=1.2 downsample_times=3 batch_size=100 k=100,66
# learn_rate=1e-4 epsilon=1e-8 accuracy=60% loss=1.6 downsample_times=2 batch_size=100 k=100,66
# learn_rate=1e-4 epsilon=1e-8 accuracy=0.08% loss=21 downsample_times=3 batch_size=10 k=100,66
# learn_rate=5e-5 epsilon=1e-8 accuracy=76% loss=1.3 downsample_times=2 batch_size=100 k=100,44
# learn_rate=5e-5 epsilon=1e-8 accuracy=68% loss=1.6 downsample_times=2 batch_size=100 k=100,44
# learn_rate=5e-4 epsilon=1e-8 accuracy=69% loss=1.6 downsample_times=2 batch_size=100 k=100,44
# learn_rate=2e-4 epsilon=1e-8 accuracy=99% loss=0.16 downsample_times=2 batch_size=100 k=100,44
# learn_rate=2e-4 epsilon=1e-8 accuracy=BAD loss=20+ downsample_times=2 batch_size=100 k=100,44
# learn_rate=1e-3 epsilon=1e-10 accuracy=95% loss=0.3 downsample_times=3 batch_size=100 k=100,44
# learn_rate=1e-3 epsilon=1e-10 accuracy=87% loss=0.6 downsample_times=3 batch_size=100 k=100,44
# learn_rate=1e-3 epsilon=1e-10 accuracy=87% loss=0.4 downsample_times=3 batch_size=100 k=100,33
# learn_rate=1e-4 epsilon=1e-10 accuracy=76% loss=1.1 downsample_times=3 batch_size=100 k=100,22
# learn_rate=5e-4 epsilon=1e-10 accuracy=92% loss=.37 downsample_times=3 batch_size=100 k=100,22
# learn_rate=5e-4 epsilon=1e-10 accuracy=97% loss=.23 downsample_times=3 batch_size=100 k=100,77
# learn_rate=5e-4 epsilon=1e-10 accuracy=96% loss=.28 downsample_times=3 batch_size=100 k=100,55
# learn_rate=5e-4 epsilon=1e-10 accuracy=96% loss=.18 downsample_times=3 batch_size=100 k=100,99
# learn_rate=5e-4 epsilon=1e-10 accuracy=94% loss=.19 downsample_times=3 batch_size=100 k=100,33
# learn_rate=5e-4 epsilon=1e-10 accuracy=96% loss=.3 downsample_times=3 batch_size=100 k=100,66
# learn_rate=5e-4 epsilon=1e-10 accuracy=96% loss=.23 downsample_times=3 batch_size=100 k=100,11
# learn_rate=5e-4 epsilon=1e-10 accuracy=96% loss=.29 downsample_times=3 batch_size=100 k=100,88
# learn_rate=5e-4 epsilon=1e-10 accuracy=96% loss=.21 downsample_times=3 batch_size=100 k=100,44


correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())

if mode == 'validate':
    train_classname, train_filename, test_classname, test_filename = \
        train_test_sets_k_fold('input/driver_imgs_list.csv', k, kth)
elif mode == 'test':
    train_classname, train_filename, test_filename = \
        train_and_test('input/driver_imgs_list.csv')
else:
    print('run mode error')
    
train_size = len(train_classname)
test_size = len(test_filename)
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
            image_array = np.float32(read_image('input/train/'+train_class+'/'+train_file))
            image_batch[j] = down_sample_by(image_array, downsample_times).flatten()
            label_batch[j][int(train_class.split('c')[1])] = 1
    if i%10 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:image_batch, y_: label_batch, keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
        sys.stdout.flush()
    train_step.run(feed_dict={x: image_batch, y_: label_batch, keep_prob: 0.5})
    print('processed batch %i now at %s' % (i,train_filename[i]))
    sys.stdout.flush()
    
train_duration = time.time() - train_start
print('train time %f s' % train_duration)
sys.stdout.flush()

if mode == 'validate':
    test_image_batch = np.zeros([test_size,pixel_count])
    test_label_batch = np.zeros([test_size,10])

    for i in range(test_size):
        test_class = test_classname[i]
        test_file = test_filename[i]
        image_array = np.float32(read_image('input/train/'+test_class+'/'+test_file))
        test_image_batch[i] = down_sample_by(image_array, downsample_times).flatten()
        test_label_batch[i][int(test_class.split('c')[1])] = 1
        if i%100 == 0:
            print('read in %i tests now at %s' % (i,test_filename[i]))
            sys.stdout.flush()

    
model_name = '%s_lr%1.0e%1.0eds%ibs%ik%ikth%i.ckpt' % \
    (mode, lr, ep, downsample_times, batch_size, k, kth)

eval_start = time.time()
print('starting eval')
sys.stdout.flush()

if mode == 'validate':
    print("test accuracy %g"%accuracy.eval(feed_dict={
        x: test_image_batch, y_: test_label_batch, keep_prob: 1.0}))

    prediction = y_conv
    prediction_array = prediction.eval(feed_dict={x: test_image_batch, keep_prob: 1.0}, session=sess)
    np.save(model_name, prediction_array)
    eval_duration = time.time()-eval_start
    print('eval test %f s' % eval_duration)

    print('saving predictions to file')

    batch_number, _ = divmod(train_size, batch_size)
    batch_number += 1
    
elif mode == 'test':
    total_prediction_array = np.array([])
    test_batch_number, _ = divmod(test_size, test_batch_size)
    test_batch_number += 1
    for i in range(test_batch_number):
        test_image_batch = np.zeros([test_batch_size,pixel_count])
        for j in range(test_batch_size):
            index = i*test_batch_size+j
            if index < test_size:
                test_file = test_filename[index].split('/')[-1]
                test_filename[index] = test_file
                test_image_array = np.float32(read_image(r'input/test/'+test_file))
                test_image_batch[j] = down_sample_by(test_image_array, downsample_times).flatten()
        print('starting prediction for batch %i' % i)
        sys.stdout.flush()
        prediction = y_conv
        prediction_array = prediction.eval(feed_dict={x: test_image_batch, keep_prob: 1.0},
                                               session=sess)
        if i == 0:
            total_prediction_array = prediction_array
        else:
            total_prediction_array = np.append(total_prediction_array, prediction_array, 0)

    total_prediction_array=total_prediction_array[:test_size]
    df = pd.DataFrame(data=total_prediction_array,
                          columns=['c0','c1','c2','c3','c4','c5','c6','c7','c8','c9'])
    df.insert(0,'img',test_filename)
    df.to_csv('submit.csv', index=False)

print('saving model as %s' % model_name)
sys.stdout.flush()
save_path = saver.save(sess, model_name)

#print('restoring model')
#sys.stdout.flush()
#saver.restore(sess, model_name)

print('total time %f s' % (time.time() - start_time))
sys.stdout.flush()
