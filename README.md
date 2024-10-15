# google-colab平台訓練模型案例
南華大學_跨領域-人工智慧_MidReport

11124208王品雯、11124209蔡岱伶
## 數據集介紹
![截圖01](https://github.com/aY-Dling/REPORT_Midterm/blob/main/%E6%88%AA%E5%9C%9601.png?raw=true)
数据集来自Kaggle，质量很高，由知名医院的专业人员严格审核标注，如图所示数据有4种类别：
  CNV：具有新生血管膜和相关视网膜下液的脉络膜新血管形成
  DME：糖尿病性黄斑水肿与视网膜增厚相关的视网膜内液
  DRUSEN:早期AMD中存在多个玻璃疣
  NORMAL：视网膜正常，没有任何视网膜液或水肿
  ![截圖02](https://github.com/aY-Dling/REPORT_Midterm/blob/main/%E6%88%AA%E5%9C%9602.png?raw=true)
文件大小约为5GB，8万多张图像，分为训练，测试，验证三个文件夹，每个文件夹按照种类不同分成4个子文件夹，其次是具体图像文件。

##數據集下載
挂载文件夹：

from google.colab import drive
drive.mount('/content/gdrive/')
按照提示进行验证，结果如下：
![截圖03](https://github.com/aY-Dling/REPORT_Midterm/blob/main/%E6%88%AA%E5%9C%9603.png?raw=true)
kaggle数据下载：

创建kaggle账户并下载kaggle.json文件。创建账户这里就不介绍了，创建完账户后在“我的账户”-“API”中选择“CREATE NEW API TOKEN”，然后下载kaggle.json文件。

创建kaggle文件夹：

!mkdir -p ~/.kaggle
将kaggle.json文件夹复制到指定文件夹：

!cp /content/gdrive/My\ Drive/kaggle.json ~/.kaggle/
测试是否成功：

!kaggle competitions list
![截圖04](https://github.com/aY-Dling/REPORT_Midterm/blob/main/%E6%88%AA%E5%9C%9604.png?raw=true)
下载数据集：

!kaggle datasets download -d paultimothymooney/kermany2018
![截圖05](https://github.com/aY-Dling/REPORT_Midterm/blob/main/%E6%88%AA%E5%9C%9605.png?raw=true)
解压文件：

!unzip "/content/kermany2018.zip"
![截圖06](https://github.com/aY-Dling/REPORT_Midterm/blob/main/%E6%88%AA%E5%9C%9606.png?raw=true)
将文件解压至google云盘：

!unzip "/content/OCT2017.zip" -d "/content/gdrive/My Drive"
![截圖07](https://github.com/aY-Dling/REPORT_Midterm/blob/main/%E6%88%AA%E5%9C%9607.png?raw=true)
数据读取
训练，测试文件夹：

import os

train_folder = os.path.join('/','content','gdrive','My Drive','OCT', 'train', '**', '*.jpeg')
test_folder = os.path.join('/','content','gdrive','My Drive','OCT', 'test', '**', '*.jpeg')
有人不知道这里的“ ** ”什么意思，我举例说明吧：
Example:
      If we had the following files on our filesystem:
        - /path/to/dir/a.txt
        - /path/to/dir/b.py
        - /path/to/dir/c.py
      If we pass "/path/to/dir/*.py" as the directory, the dataset would
      produce:
        - /path/to/dir/b.py
        - /path/to/dir/c.py
数据处理
def input_fn(file_pattern, labels,
             image_size=(224,224),
             shuffle=False,
             batch_size=64, 
             num_epochs=None, 
             buffer_size=4096,
             prefetch_buffer_size=None):

    table = tf.contrib.lookup.index_table_from_tensor(mapping=tf.constant(labels))
    num_classes = len(labels)

    def _map_func(filename):
        label = tf.string_split([filename], delimiter=os.sep).values[-2]
        image = tf.image.decode_jpeg(tf.read_file(filename), channels=3)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        
        # vgg16模型图像输入shape
        image = tf.image.resize_images(image, size=image_size)
        return (image, tf.one_hot(table.lookup(label), num_classes))
    
    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=shuffle)
    
    # tensorflow2.0以后tf.contrib模块就不再维护了
    if num_epochs is not None and shuffle:
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size, num_epochs))
    elif shuffle:
        dataset = dataset.shuffle(buffer_size)
    elif num_epochs is not None:
        dataset = dataset.repeat(num_epochs)
    
    # map默认是序列的处理数据，取消序列可加快数据处理
    dataset = dataset.apply(
        tf.contrib.data.map_and_batch(map_func=_map_func,
                                      batch_size=batch_size,
                                      num_parallel_calls=os.cpu_count()))
    
    # prefetch数据预读取，合理利用CPU和GPU的空闲时间
    dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)
    
    return dataset
模型训练
import tensorflow as tf
import os

# 设置log显示等级
tf.logging.set_verbosity(tf.logging.INFO)

# 数据集标签
labels = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

# include_top:不包含最后3个全连接层
keras_vgg16 = tf.keras.applications.VGG16(input_shape=(224,224,3),
                                          include_top=False)
output = keras_vgg16.output
output = tf.keras.layers.Flatten()(output)
predictions = tf.keras.layers.Dense(len(labels), activation=tf.nn.softmax)(output)

model = tf.keras.Model(inputs=keras_vgg16.input, outputs=predictions)

for layer in keras_vgg16.layers[:-4]:
    layer.trainable = False
    
optimizer = tf.train.AdamOptimizer()
model.compile(loss='categorical_crossentropy', 
              optimizer=optimizer,
              metrics=['accuracy'])
              
est_config=tf.estimator.RunConfig(log_step_count_steps=10)
estimator = tf.keras.estimator.model_to_estimator(model,model_dir='/content/gdrive/My Drive/estlogs',config=est_config)
BATCH_SIZE = 32
EPOCHS = 2

estimator.train(input_fn=lambda:input_fn(test_folder,
                                         labels,
                                         shuffle=True,
                                         batch_size=BATCH_SIZE,
                                         buffer_size=2048,
                                         num_epochs=EPOCHS,
                                         prefetch_buffer_size=4))
