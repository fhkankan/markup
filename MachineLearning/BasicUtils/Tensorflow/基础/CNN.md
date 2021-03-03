# CNN

虽然创建自定义的CNN来图像识别，但是强烈推荐使用现有的架构方案。

## 简单的CNN

一个四层卷积神经网络，提升预测MNIST数字的准确度。前两个卷积层由Convolution-ReLU-maxpool操作组成，后两层是全联接层。

```python
# Introductory CNN Model: MNIST Digits
# ---------------------------------------
#
# In this example, we will download the MNIST handwritten
# digits and create a simple CNN network to predict the
# digit category (0-9)

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Start a graph session
sess = tf.Session()

# Load data
data_dir = 'temp'
mnist = input_data.read_data_sets(data_dir, one_hot=False)

# Convert images into 28x28 (they are downloaded as 1x784)
train_xdata = np.array([np.reshape(x, (28, 28)) for x in mnist.train.images])
test_xdata = np.array([np.reshape(x, (28, 28)) for x in mnist.test.images])

# Convert labels into one-hot encoded vectors
train_labels = mnist.train.labels
test_labels = mnist.test.labels

# Set model parameters
batch_size = 100
learning_rate = 0.005
evaluation_size = 500
image_width = train_xdata[0].shape[0]
image_height = train_xdata[0].shape[1]
target_size = np.max(train_labels) + 1
num_channels = 1  # greyscale = 1 channel
generations = 500
eval_every = 5
conv1_features = 25
conv2_features = 50
max_pool_size1 = 2  # NxN window for 1st max pool layer
max_pool_size2 = 2  # NxN window for 2nd max pool layer
fully_connected_size1 = 100

# Declare model placeholders
x_input_shape = (batch_size, image_width, image_height, num_channels)
x_input = tf.placeholder(tf.float32, shape=x_input_shape)
y_target = tf.placeholder(tf.int32, shape=(batch_size))
eval_input_shape = (evaluation_size, image_width, image_height, num_channels)
eval_input = tf.placeholder(tf.float32, shape=eval_input_shape)
eval_target = tf.placeholder(tf.int32, shape=(evaluation_size))

# Declare model parameters
# 卷积层的权重和偏置
conv1_weight = tf.Variable(tf.truncated_normal([4, 4, num_channels, conv1_features],
                                               stddev=0.1, dtype=tf.float32))
conv1_bias = tf.Variable(tf.zeros([conv1_features], dtype=tf.float32))

conv2_weight = tf.Variable(tf.truncated_normal([4, 4, conv1_features, conv2_features],
                                               stddev=0.1, dtype=tf.float32))
conv2_bias = tf.Variable(tf.zeros([conv2_features], dtype=tf.float32))

# fully connected variables
# 全联接层的权重和偏置
resulting_width = image_width // (max_pool_size1 * max_pool_size2)
resulting_height = image_height // (max_pool_size1 * max_pool_size2)
full1_input_size = resulting_width * resulting_height * conv2_features
full1_weight = tf.Variable(tf.truncated_normal([full1_input_size, fully_connected_size1],
                           stddev=0.1, dtype=tf.float32))
full1_bias = tf.Variable(tf.truncated_normal([fully_connected_size1], stddev=0.1, dtype=tf.float32))
full2_weight = tf.Variable(tf.truncated_normal([fully_connected_size1, target_size],
                                               stddev=0.1, dtype=tf.float32))
full2_bias = tf.Variable(tf.truncated_normal([target_size], stddev=0.1, dtype=tf.float32))


# Initialize Model Operations
def my_conv_net(conv_input_data):
    # First Conv-ReLU-MaxPool Layer
    conv1 = tf.nn.conv2d(conv_input_data, conv1_weight, strides=[1, 1, 1, 1], padding='SAME')
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_bias))
    max_pool1 = tf.nn.max_pool(relu1, ksize=[1, max_pool_size1, max_pool_size1, 1],
                               strides=[1, max_pool_size1, max_pool_size1, 1], padding='SAME')

    # Second Conv-ReLU-MaxPool Layer
    conv2 = tf.nn.conv2d(max_pool1, conv2_weight, strides=[1, 1, 1, 1], padding='SAME')
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_bias))
    max_pool2 = tf.nn.max_pool(relu2, ksize=[1, max_pool_size2, max_pool_size2, 1],
                               strides=[1, max_pool_size2, max_pool_size2, 1], padding='SAME')

    # Transform Output into a 1xN layer for next fully connected layer
    final_conv_shape = max_pool2.get_shape().as_list()
    final_shape = final_conv_shape[1] * final_conv_shape[2] * final_conv_shape[3]
    flat_output = tf.reshape(max_pool2, [final_conv_shape[0], final_shape])

    # First Fully Connected Layer
    fully_connected1 = tf.nn.relu(tf.add(tf.matmul(flat_output, full1_weight), full1_bias))

    # Second Fully Connected Layer
    final_model_output = tf.add(tf.matmul(fully_connected1, full2_weight), full2_bias)
    
    return final_model_output

model_output = my_conv_net(x_input)
test_model_output = my_conv_net(eval_input)

# Declare Loss Function (softmax cross entropy)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model_output, labels=y_target))  # 预测结果不是多分类，而仅仅是一类

# Create a prediction function
prediction = tf.nn.softmax(model_output)
test_prediction = tf.nn.softmax(test_model_output)


# Create accuracy function
def get_accuracy(logits, targets):
    batch_predictions = np.argmax(logits, axis=1)
    num_correct = np.sum(np.equal(batch_predictions, targets))
    return 100. * num_correct/batch_predictions.shape[0]

# Create an optimizer
my_optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
train_step = my_optimizer.minimize(loss)

# Initialize Variables
init = tf.global_variables_initializer()
sess.run(init)

# Start training loop
train_loss = []
train_acc = []
test_acc = []
for i in range(generations):
    rand_index = np.random.choice(len(train_xdata), size=batch_size)
    rand_x = train_xdata[rand_index]
    rand_x = np.expand_dims(rand_x, 3)
    rand_y = train_labels[rand_index]
    train_dict = {x_input: rand_x, y_target: rand_y}
    
    sess.run(train_step, feed_dict=train_dict)
    temp_train_loss, temp_train_preds = sess.run([loss, prediction], feed_dict=train_dict)
    temp_train_acc = get_accuracy(temp_train_preds, rand_y)
    
    if (i+1) % eval_every == 0:
        eval_index = np.random.choice(len(test_xdata), size=evaluation_size)
        eval_x = test_xdata[eval_index]
        eval_x = np.expand_dims(eval_x, 3)
        eval_y = test_labels[eval_index]
        test_dict = {eval_input: eval_x, eval_target: eval_y}
        test_preds = sess.run(test_prediction, feed_dict=test_dict)
        temp_test_acc = get_accuracy(test_preds, eval_y)
        
        # Record and print results
        train_loss.append(temp_train_loss)
        train_acc.append(temp_train_acc)
        test_acc.append(temp_test_acc)
        acc_and_loss = [(i+1), temp_train_loss, temp_train_acc, temp_test_acc]
        acc_and_loss = [np.round(x, 2) for x in acc_and_loss]
        print('Generation # {}. Train Loss: {:.2f}. Train Acc (Test Acc): {:.2f} ({:.2f})'.format(*acc_and_loss))
    
    
# Matlotlib code to plot the loss and accuracies
eval_indices = range(0, generations, eval_every)
# Plot loss over time
plt.plot(eval_indices, train_loss, 'k-')
plt.title('Softmax Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Softmax Loss')
plt.show()

# Plot train and test accuracy
plt.plot(eval_indices, train_acc, 'k-', label='Train Set Accuracy')
plt.plot(eval_indices, test_acc, 'r--', label='Test Set Accuracy')
plt.title('Train and Test Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# Plot some samples
# Plot the 6 of the last batch results:
actuals = rand_y[0:6]
predictions = np.argmax(temp_train_preds, axis=1)[0:6]
images = np.squeeze(rand_x[0:6])

Nrows = 2
Ncols = 3
for i in range(6):
    plt.subplot(Nrows, Ncols, i+1)
    plt.imshow(np.reshape(images[i], [28, 28]), cmap='Greys_r')
    plt.title('Actual: ' + str(actuals[i]) + ' Pred: ' + str(predictions[i]),
              fontsize=10)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)

```

## 进阶的CNN

实现一个更高级的读取图像数据的方法，并使用更大的CNN模型进行CIFAR10数据集上的图像识别。大部分图片数据集都太大，不能全部放入内存。Tensorflow的做法是建立一个图像管道从文件中一次批量读取。

```python
# More Advanced CNN Model: CIFAR-10
#---------------------------------------
#
# In this example, we will download the CIFAR-10 images
# and build a CNN model with dropout and regularization
#
# CIFAR is composed ot 50k train and 10k test
# images that are 32x32.

import os
import sys
import tarfile
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from six.moves import urllib
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Change Directory
try:
    abspath = os.path.abspath(__file__)
except NameError:
    abspath = os.getcwd()
dname = os.path.dirname(abspath)
os.chdir(dname)

# Start a graph session
sess = tf.Session()

# Set model parameters
batch_size = 128  # 训练集和测试集的批量大小
data_dir = 'temp'
output_every = 50  # 每迭代50次打印出状态值
generations = 20000  # 共迭代20000次
eval_every = 500  # 每迭代500次将在测试集的批量数据上进行模型评估
image_height = 32  # 图片的高
image_width = 32  # 图片的宽
crop_height = 24  # 裁剪图的高
crop_width = 24  # 裁剪图的宽
num_channels = 3  # 颜色通道
num_targets = 10  # 目标分类10类
extract_folder = 'cifar-10-batches-bin'

# Exponential Learning Rate Decay Params
# 推荐降低学习率来训练更好的模型，采用指数级减小学习率：学习率初始值设为0.1，每迭代250次指数级减少学习率，因子为10%。公式为0.1*0.9^(x/250),x为当前迭代的次数
# Tensorflow默认是连续减小学习率，但是也接收阶梯式更新学习率
learning_rate = 0.1
lr_decay = 0.1
num_gens_to_wait = 250.

# Extract model parameters
image_vec_length = image_height * image_width * num_channels
record_length = 1 + image_vec_length # ( + 1 for the 0-9 label)

# Load data
data_dir = 'temp'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
cifar10_url = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

# Check if file exists, otherwise download it
data_file = os.path.join(data_dir, 'cifar-10-binary.tar.gz')
if os.path.isfile(data_file):
    pass
else:
    # Download file
    def progress(block_num, block_size, total_size):
        progress_info = [cifar10_url, float(block_num * block_size) / float(total_size) * 100.0]
        print('\r Downloading {} - {:.2f}%'.format(*progress_info), end="")
    filepath, _ = urllib.request.urlretrieve(cifar10_url, data_file, progress)
    # Extract file
    tarfile.open(filepath, 'r:gz').extractall(data_dir)
    

# Define CIFAR reader
def read_cifar_files(filename_queue, distort_images = True):
    """
    建立图片读取器，返回一个随机打乱的图片
    首先，声明一个读取固定字节长度的读取器
    然后，从图像队列中读取图片，抽取图片并标记
    最后，使用tensorflow内建的图像修改函数随机打乱图片
    """
    reader = tf.FixedLengthRecordReader(record_bytes=record_length)
    key, record_string = reader.read(filename_queue)
    record_bytes = tf.decode_raw(record_string, tf.uint8)
    image_label = tf.cast(tf.slice(record_bytes, [0], [1]), tf.int32)
  
    # Extract image
    image_extracted = tf.reshape(tf.slice(record_bytes, [1], [image_vec_length]),
                                 [num_channels, image_height, image_width])
    
    # Reshape image
    image_uint8image = tf.transpose(image_extracted, [1, 2, 0])
    reshaped_image = tf.cast(image_uint8image, tf.float32)
    # Randomly Crop image
    final_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, crop_width, crop_height)
    
    if distort_images:
        # Randomly flip the image horizontally, change the brightness and contrast
        final_image = tf.image.random_flip_left_right(final_image)
        final_image = tf.image.random_brightness(final_image,max_delta=63)
        final_image = tf.image.random_contrast(final_image,lower=0.2, upper=1.8)

    # Normalize whitening
    final_image = tf.image.per_image_standardization(final_image)
    return final_image, image_label


# Create a CIFAR image pipeline from reader
def input_pipeline(batch_size, train_logical=True):
    """
    批处理使用的图像管道填充函数
    首先，需要建立读取图片的列表，定义如何用tensorflow内建函数创建的input producer对象读取这些图片列表。把input producer传入上一步创建的图片读取函数read_cifar_files()中
    然后，创建图像队列的批量读取器shuffle_batch()
    """
    if train_logical:
        files = [os.path.join(data_dir, extract_folder, 'data_batch_{}.bin'.format(i)) for i in range(1,6)]
    else:
        files = [os.path.join(data_dir, extract_folder, 'test_batch.bin')]
    filename_queue = tf.train.string_input_producer(files)
    image, label = read_cifar_files(filename_queue)
    
    # min_after_dequeue defines how big a buffer we will randomly sample
    #   from -- bigger means better shuffling but slower start up and more
    #   memory used.
    # capacity must be larger than min_after_dequeue and the amount larger
    #   determines the maximum we will prefetch.  Recommendation:
    #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
    min_after_dequeue = 5000
    capacity = min_after_dequeue + 3 * batch_size
    example_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                        batch_size=batch_size,
                                                        capacity=capacity,
                                                        min_after_dequeue=min_after_dequeue)

    return example_batch, label_batch

    
# Define the model architecture, this will return logits from images
def cifar_cnn_model(input_images, batch_size, train_logical=True):
    """
    声明模型函数，使用两个卷积层，接着是三个全联接层
    两个卷积操作各创建64个特征，第一个全联接层联接第二个卷积层，有384个隐藏节点。第2个全联接层操作联接刚才的384个隐藏节点到192个隐藏节点。最后的隐藏层操作联接192个隐藏节点到我们预测的10个输出分类。
    """
    def truncated_normal_var(name, shape, dtype):
        return(tf.get_variable(name=name, shape=shape, dtype=dtype, initializer=tf.truncated_normal_initializer(stddev=0.05)))
    def zero_var(name, shape, dtype):
        return(tf.get_variable(name=name, shape=shape, dtype=dtype, initializer=tf.constant_initializer(0.0)))
    
    # First Convolutional Layer
    with tf.variable_scope('conv1') as scope:
        # Conv_kernel is 5x5 for all 3 colors and we will create 64 features
        conv1_kernel = truncated_normal_var(name='conv_kernel1', shape=[5, 5, 3, 64], dtype=tf.float32)
        # We convolve across the image with a stride size of 1
        conv1 = tf.nn.conv2d(input_images, conv1_kernel, [1, 1, 1, 1], padding='SAME')
        # Initialize and add the bias term
        conv1_bias = zero_var(name='conv_bias1', shape=[64], dtype=tf.float32)
        conv1_add_bias = tf.nn.bias_add(conv1, conv1_bias)
        # ReLU element wise
        relu_conv1 = tf.nn.relu(conv1_add_bias)
    
    # Max Pooling
    pool1 = tf.nn.max_pool(relu_conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],padding='SAME', name='pool_layer1')
    
    # Local Response Normalization (parameters from paper)
    # paper: http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks
    norm1 = tf.nn.lrn(pool1, depth_radius=5, bias=2.0, alpha=1e-3, beta=0.75, name='norm1')

    # Second Convolutional Layer
    with tf.variable_scope('conv2') as scope:
        # Conv kernel is 5x5, across all prior 64 features and we create 64 more features
        conv2_kernel = truncated_normal_var(name='conv_kernel2', shape=[5, 5, 64, 64], dtype=tf.float32)
        # Convolve filter across prior output with stride size of 1
        conv2 = tf.nn.conv2d(norm1, conv2_kernel, [1, 1, 1, 1], padding='SAME')
        # Initialize and add the bias
        conv2_bias = zero_var(name='conv_bias2', shape=[64], dtype=tf.float32)
        conv2_add_bias = tf.nn.bias_add(conv2, conv2_bias)
        # ReLU element wise
        relu_conv2 = tf.nn.relu(conv2_add_bias)
    
    # Max Pooling
    pool2 = tf.nn.max_pool(relu_conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_layer2')    
    
     # Local Response Normalization (parameters from paper)
    norm2 = tf.nn.lrn(pool2, depth_radius=5, bias=2.0, alpha=1e-3, beta=0.75, name='norm2')
    
    # Reshape output into a single matrix for multiplication for the fully connected layers
    reshaped_output = tf.reshape(norm2, [batch_size, -1])
    reshaped_dim = reshaped_output.get_shape()[1].value
    
    # First Fully Connected Layer
    with tf.variable_scope('full1') as scope:
        # Fully connected layer will have 384 outputs.
        full_weight1 = truncated_normal_var(name='full_mult1', shape=[reshaped_dim, 384], dtype=tf.float32)
        full_bias1 = zero_var(name='full_bias1', shape=[384], dtype=tf.float32)
        full_layer1 = tf.nn.relu(tf.add(tf.matmul(reshaped_output, full_weight1), full_bias1))

    # Second Fully Connected Layer
    with tf.variable_scope('full2') as scope:
        # Second fully connected layer has 192 outputs.
        full_weight2 = truncated_normal_var(name='full_mult2', shape=[384, 192], dtype=tf.float32)
        full_bias2 = zero_var(name='full_bias2', shape=[192], dtype=tf.float32)
        full_layer2 = tf.nn.relu(tf.add(tf.matmul(full_layer1, full_weight2), full_bias2))

    # Final Fully Connected Layer -> 10 categories for output (num_targets)
    with tf.variable_scope('full3') as scope:
        # Final fully connected layer has 10 (num_targets) outputs.
        full_weight3 = truncated_normal_var(name='full_mult3', shape=[192, num_targets], dtype=tf.float32)
        full_bias3 =  zero_var(name='full_bias3', shape=[num_targets], dtype=tf.float32)
        final_output = tf.add(tf.matmul(full_layer2, full_weight3), full_bias3)
        
    return final_output


# Loss function
def cifar_loss(logits, targets):
    # Get rid of extra dimensions and cast targets into integers
    targets = tf.squeeze(tf.cast(targets, tf.int32))
    # Calculate cross entropy from logits and targets
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)
    # Take the average loss across batch size
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    return cross_entropy_mean


# Train step
def train_step(loss_value, generation_num):
    # Our learning rate is an exponential decay after we wait a fair number of generations
    model_learning_rate = tf.train.exponential_decay(learning_rate, generation_num,
                                                     num_gens_to_wait, lr_decay, staircase=True)
    # Create optimizer
    my_optimizer = tf.train.GradientDescentOptimizer(model_learning_rate)
    # Initialize train step
    train_step = my_optimizer.minimize(loss_value)
    return train_step


# Accuracy function
def accuracy_of_batch(logits, targets):
    # Make sure targets are integers and drop extra dimensions
    targets = tf.squeeze(tf.cast(targets, tf.int32))
    # Get predicted values by finding which logit is the greatest
    batch_predictions = tf.cast(tf.argmax(logits, 1), tf.int32)
    # Check if they are equal across the batch
    predicted_correctly = tf.equal(batch_predictions, targets)
    # Average the 1's and 0's (True's and False's) across the batch size
    accuracy = tf.reduce_mean(tf.cast(predicted_correctly, tf.float32))
    return accuracy

# Get data
print('Getting/Transforming Data.')
# Initialize the data pipeline
images, targets = input_pipeline(batch_size, train_logical=True)
# Get batch test images and targets from pipline
test_images, test_targets = input_pipeline(batch_size, train_logical=False)

# Declare Model
print('Creating the CIFAR10 Model.')
with tf.variable_scope('model_definition') as scope:
    # Declare the training network model
    model_output = cifar_cnn_model(images, batch_size)
    # This is very important!!!  We must set the scope to REUSE the variables,
    #  otherwise, when we set the test network model, it will create new random
    #  variables.  Otherwise we get random evaluations on the test batches.
    scope.reuse_variables()  # 创建训练模型后声明，这样可以在创建测试模型时重用训练模型相同的模型参数
    test_output = cifar_cnn_model(test_images, batch_size)

# Declare loss function
print('Declare Loss Function.')
loss = cifar_loss(model_output, targets)

# Create accuracy function
accuracy = accuracy_of_batch(test_output, test_targets)

# Create training operations
# 迭代变量需要声明为非训练型变量，并传入训练函数，用于计算学习率的指数级衰减值
print('Creating the Training Operation.')
generation_num = tf.Variable(0, trainable=False)
train_op = train_step(loss, generation_num)

# Initialize Variables
print('Initializing the Variables.')
init = tf.global_variables_initializer()
sess.run(init)

# Initialize queue (This queue will feed into the model, so no placeholders necessary)
tf.train.start_queue_runners(sess=sess)

# Train CIFAR Model
print('Starting Training')
train_loss = []
test_accuracy = []
for i in range(generations):
    _, loss_value = sess.run([train_op, loss])
    
    if (i+1) % output_every == 0:
        train_loss.append(loss_value)
        output = 'Generation {}: Loss = {:.5f}'.format((i+1), loss_value)
        print(output)
    
    if (i+1) % eval_every == 0:
        [temp_accuracy] = sess.run([accuracy])
        test_accuracy.append(temp_accuracy)
        acc_output = ' --- Test Accuracy = {:.2f}%.'.format(100.*temp_accuracy)
        print(acc_output)

# Print loss and accuracy
# Matlotlib code to plot the loss and accuracies
eval_indices = range(0, generations, eval_every)
output_indices = range(0, generations, output_every)

# Plot loss over time
plt.plot(output_indices, train_loss, 'k-')
plt.title('Softmax Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Softmax Loss')
plt.show()

# Plot accuracy over time
plt.plot(eval_indices, test_accuracy, 'k-')
plt.title('Test Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.show()
```

## 训练已有的CNN

使用预训练好的图像模型，微调后训练其他图片数据集。

基本思路：重用预训练模型的卷积层的权重和结构，然后再训练全联接层。

使用Inception架构

```python
# Download/Saving CIFAR-10 images in Inception format
#---------------------------------------
#
# In this script, we download the CIFAR-10 images and
# transform/save them in the Inception Retraining Format
#
# The end purpose of the files is for re-training the
# Google Inception tensorflow model to work on the CIFAR-10.

import os
import tarfile
import _pickle as cPickle
import numpy as np
import urllib.request
import scipy.misc
from tensorflow.python.framework import ops
ops.reset_default_graph()

cifar_link = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
data_dir = 'temp'
if not os.path.isdir(data_dir):
    os.makedirs(data_dir)

# Download tar file
target_file = os.path.join(data_dir, 'cifar-10-python.tar.gz')
if not os.path.isfile(target_file):
    print('CIFAR-10 file not found. Downloading CIFAR data (Size = 163MB)')
    print('This may take a few minutes, please wait.')
    filename, headers = urllib.request.urlretrieve(cifar_link, target_file)

# Extract into memory
tar = tarfile.open(target_file)
tar.extractall(path=data_dir)
tar.close()
objects = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 创建训练所需的文件夹结构。每个文件夹下有10个子文件，分别存储10个目标分类
# Create train image folders
train_folder = 'train_dir'
if not os.path.isdir(os.path.join(data_dir, train_folder)):
    for i in range(10):
        folder = os.path.join(data_dir, train_folder, objects[i])
        os.makedirs(folder)
# Create test image folders
test_folder = 'validation_dir'
if not os.path.isdir(os.path.join(data_dir, test_folder)):
    for i in range(10):
        folder = os.path.join(data_dir, test_folder, objects[i])
        os.makedirs(folder)

# Extract images accordingly
data_location = os.path.join(data_dir, 'cifar-10-batches-py')
train_names = ['data_batch_' + str(x) for x in range(1,6)]
test_names = ['test_batch']


def load_batch_from_file(file):
    """
    创建函数从内存中加载图片并存入到图像字典中
    """
    file_conn = open(file, 'rb')
    image_dictionary = cPickle.load(file_conn, encoding='latin1')
    file_conn.close()
    return image_dictionary


def save_images_from_dict(image_dict, folder='data_dir'):
    """
    将每个文件保存到正确位置
    """
    # image_dict.keys() = 'labels', 'filenames', 'data', 'batch_label'
    for ix, label in enumerate(image_dict['labels']):
        folder_path = os.path.join(data_dir, folder, objects[label])
        filename = image_dict['filenames'][ix]
        #Transform image data
        image_array = image_dict['data'][ix]
        image_array.resize([3, 32, 32])
        # Save image
        output_location = os.path.join(folder_path, filename)
        scipy.misc.imsave(output_location,image_array.transpose())

# Sort train images
for file in train_names:
    print('Saving images from file: {}'.format(file))
    file_location = os.path.join(data_dir, 'cifar-10-batches-py', file)
    image_dict = load_batch_from_file(file_location)
    save_images_from_dict(image_dict, folder=train_folder)

# Sort test images
for file in test_names:
    print('Saving images from file: {}'.format(file))
    file_location = os.path.join(data_dir, 'cifar-10-batches-py', file)
    image_dict = load_batch_from_file(file_location)
    save_images_from_dict(image_dict, folder=test_folder)
    
# Create labels file
cifar_labels_file = os.path.join(data_dir,'cifar10_labels.txt')
print('Writing labels file, {}'.format(cifar_labels_file))
with open(cifar_labels_file, 'w') as labels_file:
    for item in objects:
        labels_file.write("{}\n".format(item))

# After this is done, we proceed with the TensorFlow fine-tuning tutorial.

# https://www.tensorflow.org/tutorials/image_retraining

```

## 图像风格迁移

Stylenet程序需输入两幅图片，将一幅图片的图像风格应用于另外一幅图的内容上。一些CNN

模型的中间层存在某些属性可以编码图片风格和图片内容，最后，从风格图片中训练图片风格层，从原始图片中训练图片内容层，并且反向传播这些计算损失函数，从而让原始图片更像风格图片。

使用imagenet-vgg-19网络

```python
# Using TensorFlow for Stylenet/NeuralStyle
#---------------------------------------
#
# We use two images, an original image and a style image
# and try to make the original image in the style of the style image.
#
# Reference paper:
# https://arxiv.org/abs/1508.06576
#
# Need to download the model 'imagenet-vgg-verydee-19.mat' from:
#   http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat

import os
import scipy.io
import scipy.misc
import imageio
from skimage.transform import resize
from operator import mul
from functools import reduce
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Image Files
original_image_file = 'images/book_cover.jpg'  # 原始图片
style_image_file = 'images/starry_night.jpg'  # 风格图片

# Saved VGG Network path under the current project dir.
vgg_path = 'imagenet-vgg-verydeep-19.mat'

# Default Arguments
original_image_weight = 5.0
style_image_weight = 500.0
regularization_weight = 100
learning_rate = 10
generations = 100
output_generations = 25
beta1 = 0.9
beta2 = 0.999

# Read in images
original_image = imageio.imread(original_image_file)
style_image = imageio.imread(style_image_file)

# Get shape of target and make the style image the same
target_shape = original_image.shape
style_image = resize(style_image, target_shape)

# VGG-19 Layer Setup
# From paper
vgg_layers = ['conv1_1', 'relu1_1',
              'conv1_2', 'relu1_2', 'pool1',
              'conv2_1', 'relu2_1',
              'conv2_2', 'relu2_2', 'pool2',
              'conv3_1', 'relu3_1',
              'conv3_2', 'relu3_2',
              'conv3_3', 'relu3_3',
              'conv3_4', 'relu3_4', 'pool3',
              'conv4_1', 'relu4_1',
              'conv4_2', 'relu4_2',
              'conv4_3', 'relu4_3',
              'conv4_4', 'relu4_4', 'pool4',
              'conv5_1', 'relu5_1',
              'conv5_2', 'relu5_2',
              'conv5_3', 'relu5_3',
              'conv5_4', 'relu5_4']


# Extract weights and matrix means
def extract_net_info(path_to_params):
    vgg_data = scipy.io.loadmat(path_to_params)
    normalization_matrix = vgg_data['normalization'][0][0][0]
    mat_mean = np.mean(normalization_matrix, axis=(0,1))
    network_weights = vgg_data['layers'][0]
    return mat_mean, network_weights
    

# Create the VGG-19 Network
def vgg_network(network_weights, init_image):
    network = {}
    image = init_image

    for i, layer in enumerate(vgg_layers):
        if layer[0] == 'c':
            weights, bias = network_weights[i][0][0][0][0]
            weights = np.transpose(weights, (1, 0, 2, 3))
            bias = bias.reshape(-1)
            conv_layer = tf.nn.conv2d(image, tf.constant(weights), (1, 1, 1, 1), 'SAME')
            image = tf.nn.bias_add(conv_layer, bias)
        elif layer[0] == 'r':
            image = tf.nn.relu(image)
        else:  # pooling
            image = tf.nn.max_pool(image, (1, 2, 2, 1), (1, 2, 2, 1), 'SAME')
        network[layer] = image
    return network

# Here we define which layers apply to the original or style image
# 文章中推荐了为原始图片和风格图片分配中间层的一些策略
original_layers = ['relu4_2', 'relu5_2']
style_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']

# Get network parameters
normalization_mean, network_weights = extract_net_info(vgg_path)

shape = (1,) + original_image.shape
style_shape = (1,) + style_image.shape
original_features = {}
style_features = {}

# Set style weights
style_weights = {l: 1./(len(style_layers)) for l in style_layers}

# Computer feature layers with original image
g_original = tf.Graph()
with g_original.as_default(), tf.Session() as sess1:
    image = tf.placeholder('float', shape=shape)
    vgg_net = vgg_network(network_weights, image)
    original_minus_mean = original_image - normalization_mean
    original_norm = np.array([original_minus_mean])
    for layer in original_layers:
        original_features[layer] = vgg_net[layer].eval(feed_dict={image: original_norm})

# Get style image network
g_style = tf.Graph()
with g_style.as_default(), tf.Session() as sess2:
    image = tf.placeholder('float', shape=style_shape)
    vgg_net = vgg_network(network_weights, image)
    style_minus_mean = style_image - normalization_mean
    style_norm = np.array([style_minus_mean])
    for layer in style_layers:
        features = vgg_net[layer].eval(feed_dict={image: style_norm})
        features = np.reshape(features, (-1, features.shape[3]))
        gram = np.matmul(features.T, features) / features.size
        style_features[layer] = gram

# Make Combined Image via loss function
with tf.Graph().as_default():
    # Get network parameters
    initial = tf.random_normal(shape) * 0.256
    init_image = tf.Variable(initial)
    vgg_net = vgg_network(network_weights, init_image)

    # Loss from Original Image
    # 内容损失
    original_layers_w = {'relu4_2': 0.5, 'relu5_2': 0.5}
    original_loss = 0
    for o_layer in original_layers:
        temp_original_loss = original_layers_w[o_layer] * original_image_weight *\
                             (2 * tf.nn.l2_loss(vgg_net[o_layer] - original_features[o_layer]))
        original_loss += (temp_original_loss / original_features[o_layer].size)

    # Loss from Style Image
    # 风格损失
    style_loss = 0
    style_losses = []
    for style_layer in style_layers:
        layer = vgg_net[style_layer]
        feats, height, width, channels = [x.value for x in layer.get_shape()]
        size = height * width * channels
        features = tf.reshape(layer, (-1, channels))
        style_gram_matrix = tf.matmul(tf.transpose(features), features) / size
        style_expected = style_features[style_layer]
        style_losses.append(style_weights[style_layer] * 2 *
                            tf.nn.l2_loss(style_gram_matrix - style_expected) /
                            style_expected.size)
    style_loss += style_image_weight * tf.reduce_sum(style_losses)

    # To Smooth the results, we add in total variation loss
    # 总变分损失，对变化显著的邻域像素进行抑制
    total_var_x = reduce(mul, init_image[:, 1:, :, :].get_shape().as_list(), 1)
    total_var_y = reduce(mul, init_image[:, :, 1:, :].get_shape().as_list(), 1)
    first_term = regularization_weight * 2
    second_term_numerator = tf.nn.l2_loss(init_image[:, 1:, :, :] - init_image[:, :shape[1]-1, :, :])
    second_term = second_term_numerator / total_var_y
    third_term = (tf.nn.l2_loss(init_image[:, :, 1:, :] - init_image[:, :, :shape[2]-1, :]) / total_var_x)
    total_variation_loss = first_term * (second_term + third_term)

    # Combined Loss
    loss = original_loss + style_loss + total_variation_loss

    # Declare Optimization Algorithm
    optimizer = tf.train.AdamOptimizer(learning_rate, beta1, beta2)
    train_step = optimizer.minimize(loss)

    # Initialize variables and start training
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(generations):

            train_step.run()

            # Print update and save temporary output
            if (i+1) % output_generations == 0:
                print('Generation {} out of {}, loss: {}'.format(i + 1, generations,sess.run(loss)))
                image_eval = init_image.eval()
                best_image_add_mean = image_eval.reshape(shape[1:]) + normalization_mean
                output_file = 'temp_output_{}.jpg'.format(i)
                imageio.imwrite(output_file, best_image_add_mean)
        
        
        # Save final image
        image_eval = init_image.eval()
        best_image_add_mean = image_eval.reshape(shape[1:]) + normalization_mean
        output_file = 'final_output.jpg'
        scipy.misc.imsave(output_file, best_image_add_mean)
```

## DeepDream

重用已训练好的CNN模型，利用一些中间节点检测标签特征的实质。利用这个实质，可以找到转换任何图像的方法，以反映选择的任何节点的特征。

```python
# Using TensorFlow for Deep Dream
#---------------------------------------
# From: Alexander Mordvintsev
#      --https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/deepdream
#
# Make sure to download the deep dream model here:
#   https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
#
# Run:
#  me@computer:~$ wget https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip 
#  me@computer:~$ unzip inception5h.zip
#
#  More comments added inline.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import tensorflow as tf
from io import BytesIO
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Start a graph session
graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)

os.chdir('~/Documents/tensorflow/inception-v1-model/')

# Model filename
model_fn = 'tensorflow_inception_graph.pb'

# Load graph parameters
with tf.gfile.FastGFile(model_fn, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

# Create placeholder for input
t_input = tf.placeholder(np.float32, name='input')

# Imagenet average bias to subtract off images
imagenet_mean = 117.0
t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)
tf.import_graph_def(graph_def, {'input':t_preprocessed})

# Create a list of layers that we can refer to later
layers = [op.name for op in graph.get_operations() if op.type=='Conv2D' and 'import/' in op.name]

# Count how many outputs for each layer
feature_nums = [int(graph.get_tensor_by_name(name+':0').get_shape()[-1]) for name in layers]

# Print count of layers and outputs (features nodes)
print('Number of layers', len(layers))
print('Total number of feature channels:', sum(feature_nums))

# Picking some internal layer. Note that we use outputs before applying the ReLU nonlinearity
# to have non-zero gradients for features with negative initial activations.
layer = 'mixed4d_3x3_bottleneck_pre_relu'
channel = 30 # picking some feature channel to visualize

# start with a gray image with a little noise
img_noise = np.random.uniform(size=(224,224,3)) + 100.0

def showarray(a, fmt='jpeg'):
    # First make sure everything is between 0 and 255
    a = np.uint8(np.clip(a, 0, 1)*255)
    # Pick an in-memory format for image display
    f = BytesIO()
    # Create the in memory image
    PIL.Image.fromarray(a).save(f, fmt)
    # Show image
    plt.imshow(a)


def T(layer):
    '''Helper for getting layer output tensor'''
    return graph.get_tensor_by_name("import/%s:0"%layer)


# The following function returns a function wrapper that will create the placeholder
# inputs of a specified dtype
def tffunc(*argtypes):
    '''Helper that transforms TF-graph generating function into a regular one.
    See "resize" function below.
    '''
    placeholders = list(map(tf.placeholder, argtypes))
    def wrap(f):
        out = f(*placeholders)
        def wrapper(*args, **kw):
            return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
        return wrapper
    return wrap


# Helper function that uses TF to resize an image
def resize(img, size):
    img = tf.expand_dims(img, 0)
    # Change 'img' size by linear interpolation
    return tf.image.resize_bilinear(img, size)[0, :, :, :]


def calc_grad_tiled(img, t_grad, tile_size=512):
    '''Compute the value of tensor t_grad over the image in a tiled way.
    Random shifts are applied to the image to blur tile boundaries over 
    multiple iterations.
    需要一种方法更新源图片，让其更像选择的特征。通过指定图片的梯度如何计算来实现。
    定义函数计算图片上子区域的梯度，使得梯度计算更快。在图片的x和y轴方向上随机移动或滚动，这将平滑方格的影响
    '''
    # Pick a subregion square size
    sz = tile_size
    # Get the image height and width
    h, w = img.shape[:2]
    # Get a random shift amount in the x and y direction
    sx, sy = np.random.randint(sz, size=2)
    # Randomly shift the image (roll image) in the x and y directions
    img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
    # Initialize the while image gradient as zeros
    grad = np.zeros_like(img)
    # Now we loop through all the sub-tiles in the image
    for y in range(0, max(h-sz//2, sz),sz):
        for x in range(0, max(w-sz//2, sz),sz):
            # Select the sub image tile
            sub = img_shift[y:y+sz,x:x+sz]
            # Calculate the gradient for the tile
            g = sess.run(t_grad, {t_input:sub})
            # Apply the gradient of the tile to the whole image gradient
            grad[y:y+sz,x:x+sz] = g
    # Return the gradient, undoing the roll operation
    return np.roll(np.roll(grad, -sx, 1), -sy, 0)

def render_deepdream(t_obj, img0=img_noise,
                     iter_n=10, step=1.5, octave_n=4, octave_scale=1.4):
    """
    声明deepdream函数，deepdream算法的对象是选择特征的平均值。损失函数是基于梯度的，其依赖于输入图片和选取特征之间的距离。策略是分割图像为高频部分和低频部分，在低频部分上计算梯度。将高频部分的结果再分割为高频部分和低频部分，重复前面的过程。原始图片和低频图片称为octaves.对传入的每个对象，计算其梯度并应用到图片中
    """
    # defining the optimization objective, the objective is the mean of the feature
    t_score = tf.reduce_mean(t_obj)
    # Our gradients will be defined as changing the t_input to get closer to
    # the values of t_score.  Here, t_score is the mean of the feature we select,
    # and t_input will be the image octave (starting with the last)
    t_grad = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!

    # Store the image
    img = img0
    # Initialize the octave list
    octaves = []
    # Since we stored the image, we need to only calculate n-1 octaves
    for i in range(octave_n-1):
        # Extract the image shape
        hw = img.shape[:2]
        # Resize the image, scale by the octave_scale (resize by linear interpolation)
        lo = resize(img, np.int32(np.float32(hw)/octave_scale))
        # Residual is hi.  Where residual = image - (Resize lo to be hw-shape)
        hi = img-resize(lo, hw)
        # Save the lo image for re-iterating
        img = lo
        # Save the extracted hi-image
        octaves.append(hi)
    
    # generate details octave by octave
    for octave in range(octave_n):
        if octave>0:
            # Start with the last octave
            hi = octaves[-octave]
            #
            img = resize(img, hi.shape[:2])+hi
        for i in range(iter_n):
            # Calculate gradient of the image.
            g = calc_grad_tiled(img, t_grad)
            # Ideally, we would just add the gradient, g, but
            # we want do a forward step size of it ('step'),
            # and divide it by the avg. norm of the gradient, so
            # we are adding a gradient of a certain size each step.
            # Also, to make sure we aren't dividing by zero, we add 1e-7.
            img += g*(step / (np.abs(g).mean()+1e-7))
            print('.',end = ' ')
        showarray(img/255.0)

# Run Deep Dream
if __name__=="__main__":
    # Create resize function that has a wrapper that creates specified placeholder types
    resize = tffunc(np.float32, np.int32)(resize)
    
    # Open image
    img0 = PIL.Image.open('book_cover.jpg')
    img0 = np.float32(img0)
    # Show Original Image
    showarray(img0/255.0)

    # Create deep dream
    render_deepdream(T(layer)[:, :, :, channel], img0, iter_n=15)

    sess.close()

```



