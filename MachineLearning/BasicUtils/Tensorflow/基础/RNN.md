# RNN

## RNN机性能垃圾邮件预测

使用UCI大学的机器学习仓库的SMS垃圾邮件数据集，使用的预测架构是，嵌入文本中的输入RNN序列，取最后一个RNN输出作为是否为垃圾邮件(1或0)的预测。

```python
# Implementing an RNN in TensorFlow
#----------------------------------
#
# We implement an RNN in TensorFlow to predict spam/ham from texts
#

import os
import re
import io
import requests
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from zipfile import ZipFile
from tensorflow.python.framework import ops
ops.reset_default_graph()


"""
计算图绘画，设置RNN模型参数
训练数据20个epoch，批量大小为250，邮件最大长度为25个单词，超过部分会被截取掉，不够部分用0填充。
RNN模型由10个单元组成，仅仅处理词频超过10的单词，每个单词会嵌入在长度为50的词向量中。
dropout概率为占位符，训练模型时设为0.5，评估模型时设为1.0.
"""
# Start a graph
sess = tf.Session()

# Set RNN parameters
epochs = 20
batch_size = 250
max_sequence_length = 25
rnn_size = 10
embedding_size = 50
min_word_frequency = 10
learning_rate = 0.0005
dropout_keep_prob = tf.placeholder(tf.float32)


# Download or open data
data_dir = 'temp'
data_file = 'text_data.txt'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

if not os.path.isfile(os.path.join(data_dir, data_file)):
    zip_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
    r = requests.get(zip_url)
    z = ZipFile(io.BytesIO(r.content))
    file = z.read('SMSSpamCollection')
    # Format Data
    text_data = file.decode()
    text_data = text_data.encode('ascii', errors='ignore')
    text_data = text_data.decode().split('\n')

    # Save data to text file
    with open(os.path.join(data_dir, data_file), 'w') as file_conn:
        for text in text_data:
            file_conn.write("{}\n".format(text))
else:
    # Open data from text file
    text_data = []
    with open(os.path.join(data_dir, data_file), 'r') as file_conn:
        for row in file_conn:
            text_data.append(row)
    text_data = text_data[:-1]

text_data = [x.split('\t') for x in text_data if len(x) >= 1]
[text_data_target, text_data_train] = [list(x) for x in zip(*text_data)]


# Create a text cleaning function
# 为降低词汇量，移除特殊字符和空格，将所有文本转为小写
def clean_text(text_string):
    text_string = re.sub(r'([^\s\w]|_|[0-9])+', '', text_string)
    text_string = " ".join(text_string.split())
    text_string = text_string.lower()
    return text_string


# Clean texts
text_data_train = [clean_text(x) for x in text_data_train]

# Change texts into numeric vectors
# 使用词汇处理器处理文本，将文本转换为索引列表
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_sequence_length,
                                                                     min_frequency=min_word_frequency)
text_processed = np.array(list(vocab_processor.fit_transform(text_data_train)))

# Shuffle and split data
text_processed = np.array(text_processed)
text_data_target = np.array([1 if x == 'ham' else 0 for x in text_data_target])
shuffled_ix = np.random.permutation(np.arange(len(text_data_target)))
x_shuffled = text_processed[shuffled_ix]
y_shuffled = text_data_target[shuffled_ix]

# Split train/test set
ix_cutoff = int(len(y_shuffled)*0.80)
x_train, x_test = x_shuffled[:ix_cutoff], x_shuffled[ix_cutoff:]
y_train, y_test = y_shuffled[:ix_cutoff], y_shuffled[ix_cutoff:]
vocab_size = len(vocab_processor.vocabulary_)
print("Vocabulary Size: {:d}".format(vocab_size))
print("80-20 Train Test split: {:d} -- {:d}".format(len(y_train), len(y_test)))

# Create placeholders
x_data = tf.placeholder(tf.int32, [None, max_sequence_length])
y_output = tf.placeholder(tf.int32, [None])

# Create embedding
embedding_mat = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
embedding_output = tf.nn.embedding_lookup(embedding_mat, x_data)

# Define the RNN cell
# tensorflow change >= 1.0, rnn is put into tensorflow.contrib directory. Prior version not test.
if tf.__version__[0] >= '1':
    cell = tf.contrib.rnn.BasicRNNCell(num_units=rnn_size)
else:
    cell = tf.nn.rnn_cell.BasicRNNCell(num_units=rnn_size)

output, state = tf.nn.dynamic_rnn(cell, embedding_output, dtype=tf.float32)
output = tf.nn.dropout(output, dropout_keep_prob)

# Get output of RNN sequence
output = tf.transpose(output, [1, 0, 2])
last = tf.gather(output, int(output.get_shape()[0]) - 1)

# 通过全联接层将rnn_size大小的输出转换为二分类输出
weight = tf.Variable(tf.truncated_normal([rnn_size, 2], stddev=0.1))
bias = tf.Variable(tf.constant(0.1, shape=[2]))
logits_out = tf.matmul(last, weight) + bias

# Loss function
losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_out, labels=y_output)
loss = tf.reduce_mean(losses)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits_out, 1), tf.cast(y_output, tf.int64)), tf.float32))

optimizer = tf.train.RMSPropOptimizer(learning_rate)
train_step = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess.run(init)

train_loss = []
test_loss = []
train_accuracy = []
test_accuracy = []
# Start training
for epoch in range(epochs):

    # Shuffle training data
    shuffled_ix = np.random.permutation(np.arange(len(x_train)))
    x_train = x_train[shuffled_ix]
    y_train = y_train[shuffled_ix]
    num_batches = int(len(x_train)/batch_size) + 1
    # TO DO CALCULATE GENERATIONS ExACTLY
    for i in range(num_batches):
        # Select train data
        min_ix = i * batch_size
        max_ix = np.min([len(x_train), ((i+1) * batch_size)])
        x_train_batch = x_train[min_ix:max_ix]
        y_train_batch = y_train[min_ix:max_ix]
        
        # Run train step
        train_dict = {x_data: x_train_batch, y_output: y_train_batch, dropout_keep_prob:0.5}
        sess.run(train_step, feed_dict=train_dict)
        
    # Run loss and accuracy for training
    temp_train_loss, temp_train_acc = sess.run([loss, accuracy], feed_dict=train_dict)
    train_loss.append(temp_train_loss)
    train_accuracy.append(temp_train_acc)
    
    # Run Eval Step
    test_dict = {x_data: x_test, y_output: y_test, dropout_keep_prob:1.0}
    temp_test_loss, temp_test_acc = sess.run([loss, accuracy], feed_dict=test_dict)
    test_loss.append(temp_test_loss)
    test_accuracy.append(temp_test_acc)
    print('Epoch: {}, Test Loss: {:.2}, Test Acc: {:.2}'.format(epoch+1, temp_test_loss, temp_test_acc))
    
# Plot loss over time
epoch_seq = np.arange(1, epochs+1)
plt.plot(epoch_seq, train_loss, 'k--', label='Train Set')
plt.plot(epoch_seq, test_loss, 'r-', label='Test Set')
plt.title('Softmax Loss')
plt.xlabel('Epochs')
plt.ylabel('Softmax Loss')
plt.legend(loc='upper left')
plt.show()

# Plot accuracy over time
plt.plot(epoch_seq, train_accuracy, 'k--', label='Train Set')
plt.plot(epoch_seq, test_accuracy, 'r-', label='Test Set')
plt.title('Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.show()
```

## LSTM

使用带有LSTM结构的序列RNN模型在莎士比亚文本数据集上训练，预测下一个单词。

```python
# -*- coding: utf-8 -*-
#
# Implementing an LSTM RNN Model
#------------------------------
#  Here we implement an LSTM model on all a data set of Shakespeare works.
#
#
#

import os
import re
import string
import requests
import numpy as np
import collections
import random
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Start a session
sess = tf.Session()

# Set RNN Parameters
min_word_freq = 5  # Trim the less frequent words off
rnn_size = 128  # RNN Model size
epochs = 10  # Number of epochs to cycle through data
batch_size = 100  # Train on this many examples at once
learning_rate = 0.001  # Learning rate
training_seq_len = 50  # how long of a word group to consider
embedding_size = rnn_size  # Word embedding size
save_every = 500  # How often to save model checkpoints
eval_every = 50  # How often to evaluate the test sentences
prime_texts = ['thou art more', 'to be or not to', 'wherefore art thou']

# Download/store Shakespeare data
data_dir = 'temp'
data_file = 'shakespeare.txt'
model_path = 'shakespeare_model'
full_model_dir = os.path.join(data_dir, model_path)

# Declare punctuation to remove, everything except hyphens and apostrophes
punctuation = string.punctuation
punctuation = ''.join([x for x in punctuation if x not in ['-', "'"]])

# Make Model Directory
if not os.path.exists(full_model_dir):
    os.makedirs(full_model_dir)

# Make data directory
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

print('Loading Shakespeare Data')
# Check if file is downloaded.
if not os.path.isfile(os.path.join(data_dir, data_file)):
    print('Not found, downloading Shakespeare texts from www.gutenberg.org')
    shakespeare_url = 'http://www.gutenberg.org/cache/epub/100/pg100.txt'
    # Get Shakespeare text
    response = requests.get(shakespeare_url)
    shakespeare_file = response.content
    # Decode binary into string
    s_text = shakespeare_file.decode('utf-8')
    # Drop first few descriptive paragraphs.
    s_text = s_text[7675:]
    # Remove newlines
    s_text = s_text.replace('\r\n', '')
    s_text = s_text.replace('\n', '')
    
    # Write to file
    with open(os.path.join(data_dir, data_file), 'w') as out_conn:
        out_conn.write(s_text)
else:
    # If file has been saved, load from that file
    with open(os.path.join(data_dir, data_file), 'r') as file_conn:
        s_text = file_conn.read().replace('\n', '')

# Clean text
print('Cleaning Text')
s_text = re.sub(r'[{}]'.format(punctuation), ' ', s_text)
s_text = re.sub('\s+', ' ', s_text).strip().lower()


# Build word vocabulary function
# 创建词汇表，返回两个单词字典（单词到索引的映射、索引到单词的映射）
def build_vocab(text, min_freq):
    word_counts = collections.Counter(text.split(' '))
    # limit word counts to those more frequent than cutoff
    word_counts = {key: val for key, val in word_counts.items() if val > min_freq}
    # Create vocab --> index mapping
    words = word_counts.keys()
    vocab_to_ix_dict = {key: (i_x+1) for i_x, key in enumerate(words)}
    # Add unknown key --> 0 index
    vocab_to_ix_dict['unknown'] = 0
    # Create index --> vocab mapping
    ix_to_vocab_dict = {val: key for key, val in vocab_to_ix_dict.items()}
    
    return ix_to_vocab_dict, vocab_to_ix_dict


# Build Shakespeare vocabulary
print('Building Shakespeare Vocab')
ix2vocab, vocab2ix = build_vocab(s_text, min_word_freq)
vocab_size = len(ix2vocab) + 1
print('Vocabulary Length = {}'.format(vocab_size))
# Sanity Check
assert(len(ix2vocab) == len(vocab2ix))

# Convert text to word vectors
s_text_words = s_text.split(' ')
s_text_ix = []
for ix, x in enumerate(s_text_words):
    try:
        s_text_ix.append(vocab2ix[x])
    except KeyError:
        s_text_ix.append(0)
s_text_ix = np.array(s_text_ix)


# Define LSTM RNN Model
class LSTM_Model():
    def __init__(self, embedding_size, rnn_size, batch_size, learning_rate,
                 training_seq_len, vocab_size, infer_sample=False):
        self.embedding_size = embedding_size
        self.rnn_size = rnn_size
        self.vocab_size = vocab_size
        self.infer_sample = infer_sample
        self.learning_rate = learning_rate
        
        if infer_sample:
            self.batch_size = 1
            self.training_seq_len = 1
        else:
            self.batch_size = batch_size
            self.training_seq_len = training_seq_len
        
        self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.rnn_size)
        self.initial_state = self.lstm_cell.zero_state(self.batch_size, tf.float32)
        
        self.x_data = tf.placeholder(tf.int32, [self.batch_size, self.training_seq_len])
        self.y_output = tf.placeholder(tf.int32, [self.batch_size, self.training_seq_len])
        
        with tf.variable_scope('lstm_vars'):
            # Softmax Output Weights
            W = tf.get_variable('W', [self.rnn_size, self.vocab_size], tf.float32, tf.random_normal_initializer())
            b = tf.get_variable('b', [self.vocab_size], tf.float32, tf.constant_initializer(0.0))
        
            # Define Embedding
            embedding_mat = tf.get_variable('embedding_mat', [self.vocab_size, self.embedding_size],
                                            tf.float32, tf.random_normal_initializer())
                                            
            embedding_output = tf.nn.embedding_lookup(embedding_mat, self.x_data)
            rnn_inputs = tf.split(axis=1, num_or_size_splits=self.training_seq_len, value=embedding_output)
            rnn_inputs_trimmed = [tf.squeeze(x, [1]) for x in rnn_inputs]
        
        # If we are inferring (generating text), we add a 'loop' function
        # Define how to get the i+1 th input from the i th output
        def inferred_loop(prev):
            # Apply hidden layer
            prev_transformed = tf.matmul(prev, W) + b
            # Get the index of the output (also don't run the gradient)
            prev_symbol = tf.stop_gradient(tf.argmax(prev_transformed, 1))
            # Get embedded vector
            out = tf.nn.embedding_lookup(embedding_mat, prev_symbol)
            return out
        
        decoder = tf.contrib.legacy_seq2seq.rnn_decoder
        outputs, last_state = decoder(rnn_inputs_trimmed,
                                      self.initial_state,
                                      self.lstm_cell,
                                      loop_function=inferred_loop if infer_sample else None)
        # Non inferred outputs
        output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, self.rnn_size])
        # Logits and output
        self.logit_output = tf.matmul(output, W) + b
        self.model_output = tf.nn.softmax(self.logit_output)
        
        loss_fun = tf.contrib.legacy_seq2seq.sequence_loss_by_example
        loss = loss_fun([self.logit_output], [tf.reshape(self.y_output, [-1])],
                        [tf.ones([self.batch_size * self.training_seq_len])])
        self.cost = tf.reduce_sum(loss) / (self.batch_size * self.training_seq_len)
        self.final_state = last_state
        gradients, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tf.trainable_variables()), 4.5)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(gradients, tf.trainable_variables()))
        
    def sample(self, sess, words=ix2vocab, vocab=vocab2ix, num=10, prime_text='thou art'):
        state = sess.run(self.lstm_cell.zero_state(1, tf.float32))
        word_list = prime_text.split()
        for word in word_list[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = vocab[word]
            feed_dict = {self.x_data: x, self.initial_state: state}
            [state] = sess.run([self.final_state], feed_dict=feed_dict)

        out_sentence = prime_text
        word = word_list[-1]
        for n in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocab[word]
            feed_dict = {self.x_data: x, self.initial_state: state}
            [model_output, state] = sess.run([self.model_output, self.final_state], feed_dict=feed_dict)
            sample = np.argmax(model_output[0])
            if sample == 0:
                break
            word = words[sample]
            out_sentence = out_sentence + ' ' + word
        return out_sentence


# Define LSTM Model
lstm_model = LSTM_Model(embedding_size, rnn_size, batch_size, learning_rate,
                        training_seq_len, vocab_size)

# Tell TensorFlow we are reusing the scope for the testing
with tf.variable_scope(tf.get_variable_scope(), reuse=True):
    test_lstm_model = LSTM_Model(embedding_size, rnn_size, batch_size, learning_rate,
                                 training_seq_len, vocab_size, infer_sample=True)


# Create model saver
saver = tf.train.Saver(tf.global_variables())

# Create batches for each epoch
num_batches = int(len(s_text_ix)/(batch_size * training_seq_len)) + 1
# Split up text indices into subarrays, of equal size
batches = np.array_split(s_text_ix, num_batches)
# Reshape each split into [batch_size, training_seq_len]
batches = [np.resize(x, [batch_size, training_seq_len]) for x in batches]

# Initialize all variables
init = tf.global_variables_initializer()
sess.run(init)

# Train model
train_loss = []
iteration_count = 1
for epoch in range(epochs):
    # Shuffle word indices
    random.shuffle(batches)
    # Create targets from shuffled batches
    targets = [np.roll(x, -1, axis=1) for x in batches]
    # Run a through one epoch
    print('Starting Epoch #{} of {}.'.format(epoch+1, epochs))
    # Reset initial LSTM state every epoch
    state = sess.run(lstm_model.initial_state)
    for ix, batch in enumerate(batches):
        training_dict = {lstm_model.x_data: batch, lstm_model.y_output: targets[ix]}
        c, h = lstm_model.initial_state
        training_dict[c] = state.c
        training_dict[h] = state.h
        
        temp_loss, state, _ = sess.run([lstm_model.cost, lstm_model.final_state, lstm_model.train_op],
                                       feed_dict=training_dict)
        train_loss.append(temp_loss)
        
        # Print status every 10 gens
        if iteration_count % 10 == 0:
            summary_nums = (iteration_count, epoch+1, ix+1, num_batches+1, temp_loss)
            print('Iteration: {}, Epoch: {}, Batch: {} out of {}, Loss: {:.2f}'.format(*summary_nums))
        
        # Save the model and the vocab
        if iteration_count % save_every == 0:
            # Save model
            model_file_name = os.path.join(full_model_dir, 'model')
            saver.save(sess, model_file_name, global_step=iteration_count)
            print('Model Saved To: {}'.format(model_file_name))
            # Save vocabulary
            dictionary_file = os.path.join(full_model_dir, 'vocab.pkl')
            with open(dictionary_file, 'wb') as dict_file_conn:
                pickle.dump([vocab2ix, ix2vocab], dict_file_conn)
        
        if iteration_count % eval_every == 0:
            for sample in prime_texts:
                print(test_lstm_model.sample(sess, ix2vocab, vocab2ix, num=10, prime_text=sample))
                
        iteration_count += 1


# Plot loss over time
plt.plot(train_loss, 'k-')
plt.title('Sequence to Sequence Loss')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()

```

## 堆叠多层LSTM

堆叠多个LSTM层增加RNN模型的深度，必要时，可以将目标输出作为输入赋值给另外一个网络。

使用多层堆叠LSTM实例变化：采用三层LSTM代替原有的一层网络，字符级别的预测代替单词级别的预测。字符级别的预测将极大地减少词汇表，表中仅有40个字符（26个字母，10个数字，1个空格，3个特殊字符）

```python
# -*- coding: utf-8 -*-
#
# Stacking LSTM Layers
#---------------------
#  Here we implement an LSTM model on all a data set of Shakespeare works.
#  We will stack multiple LSTM models for a more accurate representation
#  of Shakespearean language.  We will also use characters instead of words.
#

import os
import re
import string
import requests
import numpy as np
import collections
import random
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Start a session
sess = tf.Session()

# Set RNN Parameters
num_layers = 3  # Number of RNN layers stacked
min_word_freq = 5  # Trim the less frequent words off
rnn_size = 128  # RNN Model size, has to equal embedding size
epochs = 10  # Number of epochs to cycle through data
batch_size = 100  # Train on this many examples at once
learning_rate = 0.0005  # Learning rate
training_seq_len = 50  # how long of a word group to consider
save_every = 500  # How often to save model checkpoints
eval_every = 50  # How often to evaluate the test sentences
prime_texts = ['thou art more', 'to be or not to', 'wherefore art thou']

# Download/store Shakespeare data
data_dir = 'temp'
data_file = 'shakespeare.txt'
model_path = 'shakespeare_model'
full_model_dir = os.path.join(data_dir, model_path)

# Declare punctuation to remove, everything except hyphens and apostrophes
punctuation = string.punctuation
punctuation = ''.join([x for x in punctuation if x not in ['-', "'"]])

# Make Model Directory
if not os.path.exists(full_model_dir):
    os.makedirs(full_model_dir)

# Make data directory
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

print('Loading Shakespeare Data')
# Check if file is downloaded.
if not os.path.isfile(os.path.join(data_dir, data_file)):
    print('Not found, downloading Shakespeare texts from www.gutenberg.org')
    shakespeare_url = 'http://www.gutenberg.org/cache/epub/100/pg100.txt'
    # Get Shakespeare text
    response = requests.get(shakespeare_url)
    shakespeare_file = response.content
    # Decode binary into string
    s_text = shakespeare_file.decode('utf-8')
    # Drop first few descriptive paragraphs.
    s_text = s_text[7675:]
    # Remove newlines
    s_text = s_text.replace('\r\n', '')
    s_text = s_text.replace('\n', '')
    
    # Write to file
    with open(os.path.join(data_dir, data_file), 'w') as out_conn:
        out_conn.write(s_text)
else:
    # If file has been saved, load from that file
    with open(os.path.join(data_dir, data_file), 'r') as file_conn:
        s_text = file_conn.read().replace('\n', '')

# Clean text
print('Cleaning Text')
s_text = re.sub(r'[{}]'.format(punctuation), ' ', s_text)
s_text = re.sub('\s+', ' ', s_text).strip().lower()

# Split up by characters
char_list = list(s_text)


# Build word vocabulary function
def build_vocab(characters):
    character_counts = collections.Counter(characters)
    # Create vocab --> index mapping
    chars = character_counts.keys()
    vocab_to_ix_dict = {key: (inx + 1) for inx, key in enumerate(chars)}
    # Add unknown key --> 0 index
    vocab_to_ix_dict['unknown'] = 0
    # Create index --> vocab mapping
    ix_to_vocab_dict = {val: key for key, val in vocab_to_ix_dict.items()}
    return ix_to_vocab_dict, vocab_to_ix_dict


# Build Shakespeare vocabulary
print('Building Shakespeare Vocab by Characters')
ix2vocab, vocab2ix = build_vocab(char_list)
vocab_size = len(ix2vocab)
print('Vocabulary Length = {}'.format(vocab_size))
# Sanity Check
assert(len(ix2vocab) == len(vocab2ix))

# Convert text to word vectors
s_text_ix = []
for x in char_list:
    try:
        s_text_ix.append(vocab2ix[x])
    except KeyError:
        s_text_ix.append(0)
s_text_ix = np.array(s_text_ix)


# Define LSTM RNN Model
class LSTM_Model():
    def __init__(self, rnn_size, num_layers, batch_size, learning_rate,
                 training_seq_len, vocab_size, infer_sample=False):
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.infer_sample = infer_sample
        self.learning_rate = learning_rate
        
        if infer_sample:
            self.batch_size = 1
            self.training_seq_len = 1
        else:
            self.batch_size = batch_size
            self.training_seq_len = training_seq_len
        
        self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        # 创建多层RNN模型
        self.lstm_cell = tf.contrib.rnn.MultiRNNCell([self.lstm_cell for _ in range(self.num_layers)])
        self.initial_state = self.lstm_cell.zero_state(self.batch_size, tf.float32)
        
        self.x_data = tf.placeholder(tf.int32, [self.batch_size, self.training_seq_len])
        self.y_output = tf.placeholder(tf.int32, [self.batch_size, self.training_seq_len])
        
        with tf.variable_scope('lstm_vars'):
            # Softmax Output Weights
            W = tf.get_variable('W', [self.rnn_size, self.vocab_size], tf.float32, tf.random_normal_initializer())
            b = tf.get_variable('b', [self.vocab_size], tf.float32, tf.constant_initializer(0.0))
        
            # Define Embedding
            embedding_mat = tf.get_variable('embedding_mat', [self.vocab_size, self.rnn_size],
                                            tf.float32, tf.random_normal_initializer())
                                            
            embedding_output = tf.nn.embedding_lookup(embedding_mat, self.x_data)
            rnn_inputs = tf.split(axis=1, num_or_size_splits=self.training_seq_len, value=embedding_output)
            rnn_inputs_trimmed = [tf.squeeze(x, [1]) for x in rnn_inputs]
        
        decoder = tf.contrib.legacy_seq2seq.rnn_decoder
        outputs, last_state = decoder(rnn_inputs_trimmed,
                                      self.initial_state,
                                      self.lstm_cell)
        
        # RNN outputs
        output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, rnn_size])
        # Logits and output
        self.logit_output = tf.matmul(output, W) + b
        self.model_output = tf.nn.softmax(self.logit_output)
        
        loss_fun = tf.contrib.legacy_seq2seq.sequence_loss_by_example
        loss = loss_fun([self.logit_output],[tf.reshape(self.y_output, [-1])],
                [tf.ones([self.batch_size * self.training_seq_len])],
                self.vocab_size)
        self.cost = tf.reduce_sum(loss) / (self.batch_size * self.training_seq_len)
        self.final_state = last_state
        gradients, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tf.trainable_variables()), 4.5)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(gradients, tf.trainable_variables()))
        
    def sample(self, sess, words=ix2vocab, vocab=vocab2ix, num=20, prime_text='thou art'):
        state = sess.run(self.lstm_cell.zero_state(1, tf.float32))
        char_list = list(prime_text)
        for char in char_list[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed_dict = {self.x_data: x, self.initial_state:state}
            [state] = sess.run([self.final_state], feed_dict=feed_dict)

        out_sentence = prime_text
        char = char_list[-1]
        for n in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed_dict = {self.x_data: x, self.initial_state:state}
            [model_output, state] = sess.run([self.model_output, self.final_state], feed_dict=feed_dict)
            sample = np.argmax(model_output[0])
            if sample == 0:
                break
            char = words[sample]
            out_sentence = out_sentence + char
        return out_sentence


# Define LSTM Model
lstm_model = LSTM_Model(rnn_size,
                        num_layers,
                        batch_size,
                        learning_rate,
                        training_seq_len,
                        vocab_size)

# Tell TensorFlow we are reusing the scope for the testing
with tf.variable_scope(tf.get_variable_scope(), reuse=True):
    test_lstm_model = LSTM_Model(rnn_size,
                                 num_layers,
                                 batch_size,
                                 learning_rate,
                                 training_seq_len,
                                 vocab_size,
                                 infer_sample=True)

# Create model saver
saver = tf.train.Saver(tf.global_variables())

# Create batches for each epoch
num_batches = int(len(s_text_ix)/(batch_size * training_seq_len)) + 1
# Split up text indices into subarrays, of equal size
batches = np.array_split(s_text_ix, num_batches)
# Reshape each split into [batch_size, training_seq_len]
batches = [np.resize(x, [batch_size, training_seq_len]) for x in batches]

# Initialize all variables
init = tf.global_variables_initializer()
sess.run(init)

# Train model
train_loss = []
iteration_count = 1
for epoch in range(epochs):
    # Shuffle word indices
    random.shuffle(batches)
    # Create targets from shuffled batches
    targets = [np.roll(x, -1, axis=1) for x in batches]
    # Run a through one epoch
    print('Starting Epoch #{} of {}.'.format(epoch+1, epochs))
    # Reset initial LSTM state every epoch
    state = sess.run(lstm_model.initial_state)
    for ix, batch in enumerate(batches):
        training_dict = {lstm_model.x_data: batch, lstm_model.y_output: targets[ix]}
        # We need to update initial state for each RNN cell:
        for i, (c, h) in enumerate(lstm_model.initial_state):
                    training_dict[c] = state[i].c
                    training_dict[h] = state[i].h
        
        temp_loss, state, _ = sess.run([lstm_model.cost, lstm_model.final_state, lstm_model.train_op],
                                       feed_dict=training_dict)
        train_loss.append(temp_loss)
        
        # Print status every 10 gens
        if iteration_count % 10 == 0:
            summary_nums = (iteration_count, epoch+1, ix+1, num_batches+1, temp_loss)
            print('Iteration: {}, Epoch: {}, Batch: {} out of {}, Loss: {:.2f}'.format(*summary_nums))
        
        # Save the model and the vocab
        if iteration_count % save_every == 0:
            # Save model
            model_file_name = os.path.join(full_model_dir, 'model')
            saver.save(sess, model_file_name, global_step=iteration_count)
            print('Model Saved To: {}'.format(model_file_name))
            # Save vocabulary
            dictionary_file = os.path.join(full_model_dir, 'vocab.pkl')
            with open(dictionary_file, 'wb') as dict_file_conn:
                pickle.dump([vocab2ix, ix2vocab], dict_file_conn)
        
        if iteration_count % eval_every == 0:
            for sample in prime_texts:
                print(test_lstm_model.sample(sess, ix2vocab, vocab2ix, num=10, prime_text=sample))
                
        iteration_count += 1


# Plot loss over time
plt.plot(train_loss, 'k-')
plt.title('Sequence to Sequence Loss')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()

```

## Seq2Seq翻译模型

构建英语到德语的翻译模型

```python
# -*- coding: utf-8 -*-
#
# Creating Sequence to Sequence Models
#-------------------------------------
#  Here we show how to implement sequence to sequence models.
#  Specifically, we will build an English to German translation model.
#

import os
import re
import sys
import json
import math
import time
import string
import requests
import io
import numpy as np
import collections
import random
import pickle
import string
import matplotlib.pyplot as plt
import tensorflow as tf
from zipfile import ZipFile
from collections import Counter
from tensorflow.python.ops import lookup_ops
from tensorflow.python.framework import ops
ops.reset_default_graph()

local_repository = 'temp/seq2seq'

# models can be retrieved from github: https://github.com/tensorflow/models.git
# put the models dir under python search lib path.
# 将整个MNT模型库导入到temp文件夹
if not os.path.exists(local_repository):
    from git import Repo
    tf_model_repository = 'https://github.com/tensorflow/nmt/'
    Repo.clone_from(tf_model_repository, local_repository)
    sys.path.insert(0, 'temp/seq2seq/nmt/')

# May also try to use 'attention model' by importing the attention model:
# from temp.seq2seq.nmt import attention_model as attention_model
from temp.seq2seq.nmt import model as model
from temp.seq2seq.nmt.utils import vocab_utils as vocab_utils
import temp.seq2seq.nmt.model_helper as model_helper
import temp.seq2seq.nmt.utils.iterator_utils as iterator_utils
import temp.seq2seq.nmt.utils.misc_utils as utils
import temp.seq2seq.nmt.train as train

# Start a session
sess = tf.Session()

# 设置关于词汇量大小的参数、删除哪些标点符号及数据存储位置
# Model Parameters
vocab_size = 10000
punct = string.punctuation

# Data Parameters
data_dir = 'temp'
data_file = 'eng_ger.txt'
model_path = 'seq2seq_model'
full_model_dir = os.path.join(data_dir, model_path)

# Load hyper-parameters for translation model. (Good defaults are provided in Repository).
hparams = tf.contrib.training.HParams()
param_file = 'temp/seq2seq/nmt/standard_hparams/wmt16.json'
# Can also try: (For different architectures)
# 'temp/seq2seq/nmt/standard_hparams/iwslt15.json'
# 'temp/seq2seq/nmt/standard_hparams/wmt16_gnmt_4_layer.json',
# 'temp/seq2seq/nmt/standard_hparams/wmt16_gnmt_8_layer.json',

with open(param_file, "r") as f:
    params_json = json.loads(f.read())

for key, value in params_json.items():
    hparams.add_hparam(key, value)
hparams.add_hparam('num_gpus', 0)
hparams.add_hparam('num_encoder_layers', hparams.num_layers)
hparams.add_hparam('num_decoder_layers', hparams.num_layers)
hparams.add_hparam('num_encoder_residual_layers', 0)
hparams.add_hparam('num_decoder_residual_layers', 0)
hparams.add_hparam('init_op', 'uniform')
hparams.add_hparam('random_seed', None)
hparams.add_hparam('num_embeddings_partitions', 0)
hparams.add_hparam('warmup_steps', 0)
hparams.add_hparam('length_penalty_weight', 0)
hparams.add_hparam('sampling_temperature', 0.0)
hparams.add_hparam('num_translations_per_input', 1)
hparams.add_hparam('warmup_scheme', 't2t')
hparams.add_hparam('epoch_step', 0)
hparams.num_train_steps = 5000

# Not use any pretrained embeddings
hparams.add_hparam('src_embed_file', '')
hparams.add_hparam('tgt_embed_file', '')
hparams.add_hparam('num_keep_ckpts', 5)
hparams.add_hparam('avg_ckpts', False)

# Remove attention
hparams.attention = None

# Make Model Directory
if not os.path.exists(full_model_dir):
    os.makedirs(full_model_dir)

# Make data directory
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

print('Loading English-German Data')
# Check for data, if it doesn't exist, download it and save it
if not os.path.isfile(os.path.join(data_dir, data_file)):
    print('Data not found, downloading Eng-Ger sentences from www.manythings.org')
    sentence_url = 'http://www.manythings.org/anki/deu-eng.zip'
    r = requests.get(sentence_url)
    z = ZipFile(io.BytesIO(r.content))
    file = z.read('deu.txt')
    # Format Data
    eng_ger_data = file.decode()
    eng_ger_data = eng_ger_data.encode('ascii', errors='ignore')
    eng_ger_data = eng_ger_data.decode().split('\n')
    # Write to file
    with open(os.path.join(data_dir, data_file), 'w') as out_conn:
        for sentence in eng_ger_data:
            out_conn.write(sentence + '\n')
else:
    eng_ger_data = []
    with open(os.path.join(data_dir, data_file), 'r') as in_conn:
        for row in in_conn:
            eng_ger_data.append(row[:-1])
print('Done!')

# Remove punctuation
eng_ger_data = [''.join(char for char in sent if char not in punct) for sent in eng_ger_data]
# Split each sentence by tabs    
eng_ger_data = [x.split('\t') for x in eng_ger_data if len(x) >= 1]
[english_sentence, german_sentence] = [list(x) for x in zip(*eng_ger_data)]
english_sentence = [x.lower().split() for x in english_sentence]
german_sentence = [x.lower().split() for x in german_sentence]

# We need to write them to separate text files for the text-line-dataset operations.
train_prefix = 'train'
src_suffix = 'en'  # English
tgt_suffix = 'de'  # Deutsch (German)
source_txt_file = train_prefix + '.' + src_suffix
hparams.add_hparam('src_file', source_txt_file)
target_txt_file = train_prefix + '.' + tgt_suffix
hparams.add_hparam('tgt_file', target_txt_file)
with open(source_txt_file, 'w') as f:
    for sent in english_sentence:
        f.write(' '.join(sent) + '\n')

with open(target_txt_file, 'w') as f:
    for sent in german_sentence:
        f.write(' '.join(sent) + '\n')


# Partition some sentences off for testing files
# 解析一些测试句子
test_prefix = 'test_sent'
hparams.add_hparam('dev_prefix', test_prefix)
hparams.add_hparam('train_prefix', train_prefix)
hparams.add_hparam('test_prefix', test_prefix)
hparams.add_hparam('src', src_suffix)
hparams.add_hparam('tgt', tgt_suffix)

num_sample = 100
total_samples = len(english_sentence)
# Get around 'num_sample's every so often in the src/tgt sentences
ix_sample = [x for x in range(total_samples) if x % (total_samples // num_sample) == 0]
test_src = [' '.join(english_sentence[x]) for x in ix_sample]
test_tgt = [' '.join(german_sentence[x]) for x in ix_sample]

# Write test sentences to file
with open(test_prefix + '.' + src_suffix, 'w') as f:
    for eng_test in test_src:
        f.write(eng_test + '\n')

with open(test_prefix + '.' + tgt_suffix, 'w') as f:
    for ger_test in test_src:
        f.write(ger_test + '\n')

print('Processing the vocabularies.')
# Process the English Vocabulary
# 处理英语和德语句子的词汇表，将词汇表保存在文件中
all_english_words = [word for sentence in english_sentence for word in sentence]
all_english_counts = Counter(all_english_words)
eng_word_keys = [x[0] for x in all_english_counts.most_common(vocab_size-3)]  # -3 because UNK, S, /S is also in there
eng_vocab2ix = dict(zip(eng_word_keys, range(1, vocab_size)))
eng_ix2vocab = {val: key for key, val in eng_vocab2ix.items()}
english_processed = []
for sent in english_sentence:
    temp_sentence = []
    for word in sent:
        try:
            temp_sentence.append(eng_vocab2ix[word])
        except KeyError:
            temp_sentence.append(0)
    english_processed.append(temp_sentence)


# Process the German Vocabulary
all_german_words = [word for sentence in german_sentence for word in sentence]
all_german_counts = Counter(all_german_words)
ger_word_keys = [x[0] for x in all_german_counts.most_common(vocab_size-3)]  # -3 because UNK, S, /S is also in there
ger_vocab2ix = dict(zip(ger_word_keys, range(1, vocab_size)))
ger_ix2vocab = {val: key for key, val in ger_vocab2ix.items()}
german_processed = []
for sent in german_sentence:
    temp_sentence = []
    for word in sent:
        try:
            temp_sentence.append(ger_vocab2ix[word])
        except KeyError:
            temp_sentence.append(0)
    german_processed.append(temp_sentence)


# Save vocab files for data processing
source_vocab_file = 'vocab' + '.' + src_suffix
hparams.add_hparam('src_vocab_file', source_vocab_file)
eng_word_keys = ['<unk>', '<s>', '</s>'] + eng_word_keys

target_vocab_file = 'vocab' + '.' + tgt_suffix
hparams.add_hparam('tgt_vocab_file', target_vocab_file)
ger_word_keys = ['<unk>', '<s>', '</s>'] + ger_word_keys

# Write out all unique english words
with open(source_vocab_file, 'w') as f:
    for eng_word in eng_word_keys:
        f.write(eng_word + '\n')

# Write out all unique german words
with open(target_vocab_file, 'w') as f:
    for ger_word in ger_word_keys:
        f.write(ger_word + '\n')

# Add vocab size to hyper parameters
hparams.add_hparam('src_vocab_size', vocab_size)
hparams.add_hparam('tgt_vocab_size', vocab_size)

# Add out-directory
out_dir = 'temp/seq2seq/nmt_out'
hparams.add_hparam('out_dir', out_dir)
if not tf.gfile.Exists(out_dir):
    tf.gfile.MakeDirs(out_dir)

# 分别创建训练、推理和评价计算图
# 创建训练图
class TrainGraph(collections.namedtuple("TrainGraph", ("graph", "model", "iterator", "skip_count_placeholder"))):
    pass


def create_train_graph(scope=None):
    graph = tf.Graph()
    with graph.as_default():
        src_vocab_table, tgt_vocab_table = vocab_utils.create_vocab_tables(hparams.src_vocab_file,
                                                                           hparams.tgt_vocab_file,
                                                                           share_vocab=False)

        src_dataset = tf.data.TextLineDataset(hparams.src_file)
        tgt_dataset = tf.data.TextLineDataset(hparams.tgt_file)
        skip_count_placeholder = tf.placeholder(shape=(), dtype=tf.int64)

        iterator = iterator_utils.get_iterator(src_dataset, tgt_dataset, src_vocab_table, tgt_vocab_table,
                                               batch_size=hparams.batch_size,
                                               sos=hparams.sos,
                                               eos=hparams.eos,
                                               random_seed=None,
                                               num_buckets=hparams.num_buckets,
                                               src_max_len=hparams.src_max_len,
                                               tgt_max_len=hparams.tgt_max_len,
                                               skip_count=skip_count_placeholder)
        final_model = model.Model(hparams,
                                  iterator=iterator,
                                  mode=tf.contrib.learn.ModeKeys.TRAIN,
                                  source_vocab_table=src_vocab_table,
                                  target_vocab_table=tgt_vocab_table,
                                  scope=scope)

    return TrainGraph(graph=graph, model=final_model, iterator=iterator, skip_count_placeholder=skip_count_placeholder)


train_graph = create_train_graph()


# Create the evaluation graph
# 创建评价图
class EvalGraph(collections.namedtuple("EvalGraph", ("graph", "model", "src_file_placeholder", "tgt_file_placeholder",
                                                     "iterator"))):
    pass


def create_eval_graph(scope=None):
    graph = tf.Graph()

    with graph.as_default():
        src_vocab_table, tgt_vocab_table = vocab_utils.create_vocab_tables(
            hparams.src_vocab_file, hparams.tgt_vocab_file, hparams.share_vocab)
        src_file_placeholder = tf.placeholder(shape=(), dtype=tf.string)
        tgt_file_placeholder = tf.placeholder(shape=(), dtype=tf.string)
        src_dataset = tf.data.TextLineDataset(src_file_placeholder)
        tgt_dataset = tf.data.TextLineDataset(tgt_file_placeholder)
        iterator = iterator_utils.get_iterator(
            src_dataset,
            tgt_dataset,
            src_vocab_table,
            tgt_vocab_table,
            hparams.batch_size,
            sos=hparams.sos,
            eos=hparams.eos,
            random_seed=hparams.random_seed,
            num_buckets=hparams.num_buckets,
            src_max_len=hparams.src_max_len_infer,
            tgt_max_len=hparams.tgt_max_len_infer)
        final_model = model.Model(hparams,
                                  iterator=iterator,
                                  mode=tf.contrib.learn.ModeKeys.EVAL,
                                  source_vocab_table=src_vocab_table,
                                  target_vocab_table=tgt_vocab_table,
                                  scope=scope)
    return EvalGraph(graph=graph,
                     model=final_model,
                     src_file_placeholder=src_file_placeholder,
                     tgt_file_placeholder=tgt_file_placeholder,
                     iterator=iterator)


eval_graph = create_eval_graph()


# Inference graph
# 创建推理图
class InferGraph(
    collections.namedtuple("InferGraph", ("graph", "model", "src_placeholder", "batch_size_placeholder", "iterator"))):
    pass


def create_infer_graph(scope=None):
    graph = tf.Graph()
    with graph.as_default():
        src_vocab_table, tgt_vocab_table = vocab_utils.create_vocab_tables(hparams.src_vocab_file,
                                                                           hparams.tgt_vocab_file,
                                                                           hparams.share_vocab)
        reverse_tgt_vocab_table = lookup_ops.index_to_string_table_from_file(hparams.tgt_vocab_file,
                                                                             default_value=vocab_utils.UNK)

        src_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
        batch_size_placeholder = tf.placeholder(shape=[], dtype=tf.int64)
        src_dataset = tf.data.Dataset.from_tensor_slices(src_placeholder)
        iterator = iterator_utils.get_infer_iterator(src_dataset,
                                                     src_vocab_table,
                                                     batch_size=batch_size_placeholder,
                                                     eos=hparams.eos,
                                                     src_max_len=hparams.src_max_len_infer)
        final_model = model.Model(hparams,
                                  iterator=iterator,
                                  mode=tf.contrib.learn.ModeKeys.INFER,
                                  source_vocab_table=src_vocab_table,
                                  target_vocab_table=tgt_vocab_table,
                                  reverse_target_vocab_table=reverse_tgt_vocab_table,
                                  scope=scope)
    return InferGraph(graph=graph,
                      model=final_model,
                      src_placeholder=src_placeholder,
                      batch_size_placeholder=batch_size_placeholder,
                      iterator=iterator)


infer_graph = create_infer_graph()


# Create sample data for evaluation
sample_ix = [25, 125, 240, 450]
sample_src_data = [' '.join(english_sentence[x]) for x in sample_ix]
sample_tgt_data = [' '.join(german_sentence[x]) for x in sample_ix]

config_proto = utils.get_config_proto()

train_sess = tf.Session(config=config_proto, graph=train_graph.graph)
eval_sess = tf.Session(config=config_proto, graph=eval_graph.graph)
infer_sess = tf.Session(config=config_proto, graph=infer_graph.graph)

# Load the training graph
with train_graph.graph.as_default():
    loaded_train_model, global_step = model_helper.create_or_load_model(train_graph.model,
                                                                        hparams.out_dir,
                                                                        train_sess,
                                                                        "train")


summary_writer = tf.summary.FileWriter(os.path.join(hparams.out_dir, 'Training'), train_graph.graph)

# 将评价操作添加到计算图中
for metric in hparams.metrics:
    hparams.add_hparam("best_" + metric, 0)
    best_metric_dir = os.path.join(hparams.out_dir, "best_" + metric)
    hparams.add_hparam("best_" + metric + "_dir", best_metric_dir)
    tf.gfile.MakeDirs(best_metric_dir)


eval_output = train.run_full_eval(hparams.out_dir, infer_graph, infer_sess, eval_graph, eval_sess,
                                  hparams, summary_writer, sample_src_data, sample_tgt_data)

eval_results, _, acc_blue_scores = eval_output

# Training Initialization
last_stats_step = global_step
last_eval_step = global_step
last_external_eval_step = global_step

steps_per_eval = 10 * hparams.steps_per_stats
steps_per_external_eval = 5 * steps_per_eval

avg_step_time = 0.0
step_time, checkpoint_loss, checkpoint_predict_count = 0.0, 0.0, 0.0
checkpoint_total_count = 0.0
speed, train_ppl = 0.0, 0.0

utils.print_out("# Start step %d, lr %g, %s" %
                (global_step, loaded_train_model.learning_rate.eval(session=train_sess),
                 time.ctime()))
skip_count = hparams.batch_size * hparams.epoch_step
utils.print_out("# Init train iterator, skipping %d elements" % skip_count)

train_sess.run(train_graph.iterator.initializer,
              feed_dict={train_graph.skip_count_placeholder: skip_count})


# Run training
while global_step < hparams.num_train_steps:
    start_time = time.time()
    try:
        step_result = loaded_train_model.train(train_sess)
        (_, step_loss, step_predict_count, step_summary, global_step, step_word_count,
         batch_size, __, ___) = step_result
        hparams.epoch_step += 1
    except tf.errors.OutOfRangeError:
        # Next Epoch
        hparams.epoch_step = 0
        utils.print_out("# Finished an epoch, step %d. Perform external evaluation" % global_step)
        train.run_sample_decode(infer_graph,
                                infer_sess,
                                hparams.out_dir,
                                hparams,
                                summary_writer,
                                sample_src_data,
                                sample_tgt_data)
        dev_scores, test_scores, _ = train.run_external_eval(infer_graph,
                                                             infer_sess,
                                                             hparams.out_dir,
                                                             hparams,
                                                             summary_writer)
        train_sess.run(train_graph.iterator.initializer, feed_dict={train_graph.skip_count_placeholder: 0})
        continue

    summary_writer.add_summary(step_summary, global_step)

    # Statistics
    step_time += (time.time() - start_time)
    checkpoint_loss += (step_loss * batch_size)
    checkpoint_predict_count += step_predict_count
    checkpoint_total_count += float(step_word_count)

    # print statistics
    if global_step - last_stats_step >= hparams.steps_per_stats:
        last_stats_step = global_step
        avg_step_time = step_time / hparams.steps_per_stats
        train_ppl = utils.safe_exp(checkpoint_loss / checkpoint_predict_count)
        speed = checkpoint_total_count / (1000 * step_time)

        utils.print_out("  global step %d lr %g "
                        "step-time %.2fs wps %.2fK ppl %.2f %s" %
                        (global_step,
                         loaded_train_model.learning_rate.eval(session=train_sess),
                         avg_step_time, speed, train_ppl, train._get_best_results(hparams)))

        if math.isnan(train_ppl):
            break

        # Reset timer and loss.
        step_time, checkpoint_loss, checkpoint_predict_count = 0.0, 0.0, 0.0
        checkpoint_total_count = 0.0

    if global_step - last_eval_step >= steps_per_eval:
        last_eval_step = global_step
        utils.print_out("# Save eval, global step %d" % global_step)
        utils.add_summary(summary_writer, global_step, "train_ppl", train_ppl)

        # Save checkpoint
        loaded_train_model.saver.save(train_sess, os.path.join(hparams.out_dir, "translate.ckpt"),
                                      global_step=global_step)

        # Evaluate on dev/test
        train.run_sample_decode(infer_graph,
                                infer_sess,
                                out_dir,
                                hparams,
                                summary_writer,
                                sample_src_data,
                                sample_tgt_data)
        dev_ppl, test_ppl = train.run_internal_eval(eval_graph,
                                                    eval_sess,
                                                    out_dir,
                                                    hparams,
                                                    summary_writer)

    if global_step - last_external_eval_step >= steps_per_external_eval:
        last_external_eval_step = global_step

        # Save checkpoint
        loaded_train_model.saver.save(train_sess, os.path.join(hparams.out_dir, "translate.ckpt"),
                                      global_step=global_step)

        train.run_sample_decode(infer_graph,
                                infer_sess,
                                out_dir,
                                hparams,
                                summary_writer,
                                sample_src_data,
                                sample_tgt_data)
        dev_scores, test_scores, _ = train.run_external_eval(infer_graph,
                                                             infer_sess,
                                                             out_dir,
                                                             hparams,
                                                             summary_writer)
```

translation_model_sample

```python
# -*- coding: utf-8 -*-
#
# Creating Sequence to Sequence Model Class
#-------------------------------------
#  Here we implement the Seq2Seq class for modeling language translation
#

import numpy as np
import tensorflow as tf


# Declare Seq2seq translation model
class Seq2Seq(object):
    def __init__(self, vocab_size, x_buckets, y_buckets, rnn_size,
                 num_layers, max_gradient, batch_size, learning_rate,
                 lr_decay_rate, forward_only=False):
        self.vocab_size = vocab_size
        self.x_buckets = x_buckets
        self.y_buckets = y_buckets
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
        self.lr_decay = self.learning_rate.assign(self.learning_rate * lr_decay_rate)
        self.global_step = tf.Variable(0, trainable=False)
        
        cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)
        
        # Decoding function
        def decode_wrapper(x, y, forward_only):
            decode_fun = tf.nn.seq2seq.embedding_attention_seq2seq(x,
                y,
                cell,
                num_encoder_symbols=vocab_size,
                num_decoder_symbols=vocab_size,
                embedding_size=rnn_size,
                feed_previous=forward_only,
                dtype=tf.float32)
            return(decode_fun)
        
        # Loss function
        loss_fun = tf.nn.softmax_cross_entropy_with_logits
        
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        for i in range(x_buckets[-1]):  # Last bucket is the biggest one.
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                      name="encoder{}".format(i)))
                                                      
        for i in range(x_buckets[-1] + 1):
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                      name="decoder{}".format(i)))
            self.target_weights.append(tf.placeholder(tf.float32, shape=[None],
                                                      name="weight{}".format(i)))
        
        
        targets = [self.decoder_inputs[i + 1] for i in range(len(self.decoder_inputs) - 1)]
        xy_buckets = [x for x in zip(x_buckets, y_buckets)]
        
        if forward_only:
            self.outputs, self.loss =  tf.nn.seq2seq.model_with_buckets(
                self.encoder_inputs, self.decoder_inputs, targets,
                self.target_weights, xy_buckets, lambda x, y: decode_wrapper(x, y, True),
                softmax_loss_function=loss_fun)
        else:
            self.outputs, self.loss =  tf.nn.seq2seq.model_with_buckets(
                self.encoder_inputs, self.decoder_inputs, targets,
                self.target_weights, xy_buckets, lambda x, y: decode_wrapper(x, y, False),
                softmax_loss_function=loss_fun)
        
        # Gradients and SGD update operation for training the model.
        if not forward_only:
            # Initialize gradient and update functions
            self.gradient_norms = []
            self.update_funs = []
            # Create a optimizer to use for each data bucket
            my_optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            for b in range(len(xy_buckets)):
                # For each bucket, get the gradients
                gradients = tf.gradients(self.losses[b], tf.trainable_variables())
                # Clip the gradients
                clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient)
                self.gradient_norms.append(norm)
                # Get the gradient update step for each variable
                temp_optimizer = my_optimizer.apply_gradients(zip(clipped_gradients, tf.trainable_variables()),
                                                              global_step=self.global_step)
                self.update_funs.append(temp_optimizer)

        self.saver = tf.train.Saver(tf.global_variables())
        
    # Define how to step forward (or backward) in the model
    def step(self, session, encoder_inputs, decoder_inputs, target_weights,
             bucket_id, forward_only):
        # Check if the sizes match.
        encoder_size, decoder_size = self.buckets[bucket_id]
        if len(encoder_inputs) != encoder_size:
            raise ValueError("Encoder length must be equal to the one in bucket,"
                           " %d != %d." % (len(encoder_inputs), encoder_size))
        if len(decoder_inputs) != decoder_size:
            raise ValueError("Decoder length must be equal to the one in bucket,"
                           " %d != %d." % (len(decoder_inputs), decoder_size))
        if len(target_weights) != decoder_size:
            raise ValueError("Weights length must be equal to the one in bucket,"
                           " %d != %d." % (len(target_weights), decoder_size))

        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        for l in range(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in range(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]

        # Since our targets are decoder inputs shifted by one, we need one more.
        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        # Output feed: depends on whether we do a backward step or not.
        if not forward_only:
            output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                           self.gradient_norms[bucket_id],  # Gradient norm.
                           self.losses[bucket_id]]  # Loss for this batch.
        else:
            output_feed = [self.losses[bucket_id]]  # Loss for this batch.
        for l in range(decoder_size):  # Output logits.
            output_feed.append(self.outputs[bucket_id][l])
        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
        else:
            return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.
    
    # Define batch iteration function
    def batch_iter(self, data, bucket_num):
        decoder_len = self.y_buckets[bucket_num]
        encoder_len = self.x_buckets[bucket_num]
        batch_ix = np.random.choice(range(len(data[bucket_num])),
                                    size = self.batch_size)
        batch_data = [data[bucket_num][ix] for ix in batch_ix]
        encoder_inputs = [x[0] for x in batch_data]
        decoder_inputs = [x[1] for x in batch_data]
        
        # Pad encoder inputs with zeros
        encoder_inputs = [(x + [0]*encoder_len)[:encoder_len] for x in encoder_inputs]
        
        # Put a '1' at the start of decoder, and pad end with zeros
        decoder_inputs = [([1] + x + [0]*decoder_len)[:(decoder_len)] for x in decoder_inputs]
        
        # Transpose the inputs/outputs into list of arrays, each array is the i-th element
        encoder_inputs_t = [np.array(x) for x in zip(*encoder_inputs)]
        decoder_inputs_t = [np.array(x) for x in zip(*decoder_inputs)]
        
        # Create batch weights (0 for padding, 1 otherwise)
        target_weights = np.ones(shape=np.array(decoder_inputs_t).shape)
        zero_ix = [[(row, col) for col, c_val in enumerate(rows) if c_val==0] for row, rows in enumerate(decoder_inputs_t)]
        zero_ix = [val for sublist in zero_ix for val in sublist if sublist]
        # Set batch weights to zero
        for row, col in zero_ix:
            target_weights[row,col]=0
        
        # Need to roll the target weights (weights point to the next case)
        batch_weights = np.roll(batch_weights, -1, axis=0)
        # Last row should be all zeros
        target_weights[-1,:] = [0]*self.batch_size
        # Make weights a list of arrays
        target_weights = [np.array(r, dtype=np.float32) for r in target_weights]
        
        return(encoder_inputs_t, decoder_inputs_t, target_weights)
```

## 孪生RNN度量相似度

相对于其他模型来说，RNN模型能处理变长的序列数据。利用该特性以及其可以生成全新的序列数据，我们创建方法度量输入序列和其他序列之间的相似度。

创建一个双向RNN模型，输入地址，将输出传入一个全联接层，全联接层输出固定长度的数值型向量。然后比较两个向量输出的余弦相似度，其值在-1和1之间。输入数据与目标相似则为1，否则为-1。余弦距离的预测仅仅为输出结果的符号值。使用该网络进行记录匹配，取基准地址与查询地址之间余弦距离最高的。

优点是可以接受从未见过的输入并进行比较，输出-1到1.

```python
# -*- coding: utf-8 -*-
# Siamese Address Similarity with TensorFlow (Driver File)
#------------------------------------------
#
# Here, we show how to perform address matching
#   with a Siamese RNN model

import random
import string
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

import siamese_similarity_model as model

# Start a graph session
sess = tf.Session()

# Model parameters
batch_size = 200
n_batches = 300
max_address_len = 20
margin = 0.25
num_features = 50
dropout_keep_prob = 0.8


# Function to randomly create one typo in a string w/ a probability
# 使用基准地址创建有"打印错误"的相似地址
def create_typo(s):
    rand_ind = random.choice(range(len(s)))
    s_list = list(s)
    s_list[rand_ind]=random.choice(string.ascii_lowercase + '0123456789')
    s = ''.join(s_list)
    return s

# Generate data
street_names = ['abbey', 'baker', 'canal', 'donner', 'elm', 'fifth',
                'grandvia', 'hollywood', 'interstate', 'jay', 'kings']
street_types = ['rd', 'st', 'ln', 'pass', 'ave', 'hwy', 'cir', 'dr', 'jct']

# Define test addresses
test_queries = ['111 abbey ln', '271 doner cicle',
                '314 king avenue', 'tensorflow is fun']
test_references = ['123 abbey ln', '217 donner cir', '314 kings ave',
                   '404 hollywood st', 'tensorflow is so fun']

# Get a batch of size n, half of which is similar addresses, half are not
# 定义如何生成批量数据。本例的批量数据是一半相似的地址（基准地址和打印错误地址 ）和一半不相似地址。
# 不相似的地址是通过读取地址列表的后半部分，并使用
def get_batch(n):
    # Generate a list of reference addresses with similar addresses that have
    # a typo.
    numbers = [random.randint(1, 9999) for i in range(n)]
    streets = [random.choice(street_names) for i in range(n)]
    street_suffs = [random.choice(street_types) for i in range(n)]
    full_streets = [str(w) + ' ' + x + ' ' + y for w,x,y in zip(numbers, streets, street_suffs)]
    typo_streets = [create_typo(x) for x in full_streets]
    reference = [list(x) for x in zip(full_streets, typo_streets)]
    
    # Shuffle last half of them for training on dissimilar addresses
    half_ix = int(n/2)
    bottom_half = reference[half_ix:]
    true_address = [x[0] for x in bottom_half]
    typo_address = [x[1] for x in bottom_half]
    typo_address = list(np.roll(typo_address, 1))
    bottom_half = [[x,y] for x,y in zip(true_address, typo_address)]
    reference[half_ix:] = bottom_half
    
    # Get target similarities (1's for similar, -1's for non-similar)
    target = [1]*(n-half_ix) + [-1]*half_ix
    reference = [[x,y] for x,y in zip(reference, target)]
    return reference
    

# Define vocabulary dictionary (remember to save '0' for padding)
vocab_chars = string.ascii_lowercase + '0123456789 '
vocab2ix_dict = {char:(ix+1) for ix, char in enumerate(vocab_chars)}
vocab_length = len(vocab_chars) + 1

# Define vocab one-hot encoding
def address2onehot(address,
                   vocab2ix_dict = vocab2ix_dict,
                   max_address_len = max_address_len):
    # translate address string into indices
    address_ix = [vocab2ix_dict[x] for x in list(address)]

    # Pad or crop to max_address_len
    address_ix = (address_ix + [0]*max_address_len)[0:max_address_len]
    return address_ix


# Define placeholders
address1_ph = tf.placeholder(tf.int32, [None, max_address_len], name="address1_ph")
address2_ph = tf.placeholder(tf.int32, [None, max_address_len], name="address2_ph")
    
y_target_ph = tf.placeholder(tf.int32, [None], name="y_target_ph")
dropout_keep_prob_ph = tf.placeholder(tf.float32, name="dropout_keep_prob")

# Create embedding lookup
identity_mat = tf.diag(tf.ones(shape=[vocab_length]))
address1_embed = tf.nn.embedding_lookup(identity_mat, address1_ph)
address2_embed = tf.nn.embedding_lookup(identity_mat, address2_ph)

# Define Model
text_snn = model.snn(address1_embed, address2_embed, dropout_keep_prob_ph,
                     vocab_length, num_features, max_address_len)

# Define Accuracy
batch_accuracy = model.accuracy(text_snn, y_target_ph)
# Define Loss
batch_loss = model.loss(text_snn, y_target_ph, margin)
# Define Predictions
predictions = model.get_predictions(text_snn)

# Declare optimizer
optimizer = tf.train.AdamOptimizer(0.01)
# Apply gradients
train_op = optimizer.minimize(batch_loss)

# Initialize Variables
init = tf.global_variables_initializer()
sess.run(init)

# Train loop
train_loss_vec = []
train_acc_vec = []
for b in range(n_batches):
    # Get a batch of data
    batch_data = get_batch(batch_size)
    # Shuffle data
    np.random.shuffle(batch_data)
    # Parse addresses and targets
    input_addresses = [x[0] for x in batch_data]
    target_similarity = np.array([x[1] for x in batch_data])
    address1 = np.array([address2onehot(x[0]) for x in input_addresses])
    address2 = np.array([address2onehot(x[1]) for x in input_addresses])
    
    train_feed_dict = {address1_ph: address1,
                       address2_ph: address2,
                       y_target_ph: target_similarity,
                       dropout_keep_prob_ph: dropout_keep_prob}

    _, train_loss, train_acc = sess.run([train_op, batch_loss, batch_accuracy],
                                        feed_dict=train_feed_dict)
    # Save train loss and accuracy
    train_loss_vec.append(train_loss)
    train_acc_vec.append(train_acc)
    # Print out statistics
    if b%10==0:
        print('Training Metrics, Batch {0}: Loss={1:.3f}, Accuracy={2:.3f}.'.format(b, train_loss, train_acc))


# Calculate the nearest addresses for test inputs
# First process the test_queries and test_references
test_queries_ix = np.array([address2onehot(x) for x in test_queries])
test_references_ix = np.array([address2onehot(x) for x in test_references])
num_refs = test_references_ix.shape[0]
best_fit_refs = []
for query in test_queries_ix:
    test_query = np.repeat(np.array([query]), num_refs, axis=0)
    test_feed_dict = {address1_ph: test_query,
                      address2_ph: test_references_ix,
                      y_target_ph: target_similarity,
                      dropout_keep_prob_ph: 1.0}
    test_out = sess.run(text_snn, feed_dict=test_feed_dict)
    best_fit = test_references[np.argmax(test_out)]
    best_fit_refs.append(best_fit)

print('Query Addresses: {}'.format(test_queries))
print('Model Found Matches: {}'.format(best_fit_refs))

# Plot the loss and accuracy
plt.plot(train_loss_vec, 'k-', lw=2, label='Batch Loss')
plt.plot(train_acc_vec, 'r:', label='Batch Accuracy')
plt.xlabel('Iterations')
plt.ylabel('Accuracy and Loss')
plt.title('Accuracy and Loss of Siamese RNN')
plt.grid()
plt.legend(loc='lower right')
plt.show()

```

siamese_similarity_model

```python
# -*- coding: utf-8 -*-
# Siamese Address Similarity with TensorFlow (Model File)
#------------------------------------------
#
# Here, we show how to perform address matching
#   with a Siamese RNN model

import tensorflow as tf


# 创建一个孪生RNN相似度模型类
def snn(address1, address2, dropout_keep_prob,
        vocab_size, num_features, input_length):
    
    # Define the siamese double RNN with a fully connected layer at the end
    def siamese_nn(input_vector, num_hidden):
        cell_unit = tf.contrib.rnn.BasicLSTMCell#tf.nn.rnn_cell.BasicLSTMCell
        
        # Forward direction cell
        lstm_forward_cell = cell_unit(num_hidden, forget_bias=1.0)
        lstm_forward_cell = tf.contrib.rnn.DropoutWrapper(lstm_forward_cell, output_keep_prob=dropout_keep_prob)
        
        # Backward direction cell
        lstm_backward_cell = cell_unit(num_hidden, forget_bias=1.0)
        lstm_backward_cell = tf.contrib.rnn.DropoutWrapper(lstm_backward_cell, output_keep_prob=dropout_keep_prob)
    
        # Split title into a character sequence
        input_embed_split = tf.split(axis=1, num_or_size_splits=input_length, value=input_vector)
        input_embed_split = [tf.squeeze(x, axis=[1]) for x in input_embed_split]
        
        # Create bidirectional layer
        try:
            outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_forward_cell,
                                                                    lstm_backward_cell,
                                                                    input_embed_split,
                                                                    dtype=tf.float32)
        except Exception:
            outputs = tf.contrib.rnn.static_bidirectional_rnn(lstm_forward_cell,
                                                              lstm_backward_cell,
                                                              input_embed_split,
                                                              dtype=tf.float32)
        # Average The output over the sequence
        temporal_mean = tf.add_n(outputs) / input_length
        
        # Fully connected layer
        output_size = 10
        A = tf.get_variable(name="A", shape=[2*num_hidden, output_size],
                            dtype=tf.float32,
                            initializer=tf.random_normal_initializer(stddev=0.1))
        b = tf.get_variable(name="b", shape=[output_size], dtype=tf.float32,
                            initializer=tf.random_normal_initializer(stddev=0.1))
        
        final_output = tf.matmul(temporal_mean, A) + b
        final_output = tf.nn.dropout(final_output, dropout_keep_prob)
        
        return(final_output)
        
    output1 = siamese_nn(address1, num_features)
    # Declare that we will use the same variables on the second string
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        output2 = siamese_nn(address2, num_features)
    
    # Unit normalize the outputs
    output1 = tf.nn.l2_normalize(output1, 1)
    output2 = tf.nn.l2_normalize(output2, 1)
    # Return cosine distance
    #   in this case, the dot product of the norms is the same.
    dot_prod = tf.reduce_sum(tf.multiply(output1, output2), 1)
    
    return dot_prod

# 声明预测函数，该函数是余弦距离的符号值
def get_predictions(scores):
    predictions = tf.sign(scores, name="predictions")
    return predictions

# 声明损失函数
def loss(scores, y_target, margin):
    # Calculate the positive losses
    pos_loss_term = 0.25 * tf.square(tf.subtract(1., scores))
    
    # If y-target is -1 to 1, then do the following
    pos_mult = tf.add(tf.multiply(0.5, tf.cast(y_target, tf.float32)), 0.5)
    # Else if y-target is 0 to 1, then do the following
    pos_mult = tf.cast(y_target, tf.float32)
    
    # Make sure positive losses are on similar strings
    positive_loss = tf.multiply(pos_mult, pos_loss_term)
    
    # Calculate negative losses, then make sure on dissimilar strings
    
    # If y-target is -1 to 1, then do the following:
    neg_mult = tf.add(tf.multiply(-0.5, tf.cast(y_target, tf.float32)), 0.5)
    # Else if y-target is 0 to 1, then do the following
    neg_mult = tf.subtract(1., tf.cast(y_target, tf.float32))
    
    negative_loss = neg_mult*tf.square(scores)
    
    # Combine similar and dissimilar losses
    loss = tf.add(positive_loss, negative_loss)
    
    # Create the margin term.  This is when the targets are 0.,
    #  and the scores are less than m, return 0.
    
    # Check if target is zero (dissimilar strings)
    target_zero = tf.equal(tf.cast(y_target, tf.float32), 0.)
    # Check if cosine outputs is smaller than margin
    less_than_margin = tf.less(scores, margin)
    # Check if both are true
    both_logical = tf.logical_and(target_zero, less_than_margin)
    both_logical = tf.cast(both_logical, tf.float32)
    # If both are true, then multiply by (1-1)=0.
    multiplicative_factor = tf.cast(1. - both_logical, tf.float32)
    total_loss = tf.multiply(loss, multiplicative_factor)
    
    # Average loss over batch
    avg_loss = tf.reduce_mean(total_loss)
    return avg_loss

# 声明准确度函数
def accuracy(scores, y_target):
    predictions = get_predictions(scores)
    # Cast into integers (outputs can only be -1 or +1)
    y_target_int = tf.cast(y_target, tf.int32)
    # Change targets from (0,1) --> (-1, 1)
    #    via (2 * x - 1)
    #y_target_int = tf.sub(tf.mul(y_target_int, 2), 1)
    predictions_int = tf.cast(tf.sign(predictions), tf.int32)
    correct_predictions = tf.equal(predictions_int, y_target_int)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return accuracy
```

