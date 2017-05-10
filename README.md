# caffe
caffe with some things for my experiment(class  classification with clickture data)<br>
这是一个为了完成我的实验，而添加过一些loss层，小工具和代码的caffe，具体添加的东西我会一项一项加上来介绍：<br>
***

## Cosine Loss：
*the objective function is definede as:<br>
Cosine Loss的目标函数定义如下：*<br>
![Image text](https://github.com/GYxiaOH/caffe/blob/master/CosineLoss.png)
***

## tools/data：
1. HDF5 and LMDB<br>
to creat lmdb/hdf5 files：<br>
生成LMDB/HDF5的文件，其中HDF5为了保证每个文件小于2G，每3000张图片生成一个HDF5文件<br>
LMDB中，一个是train_val生成，一个只生成train<br>

2. random<br>
load filename.txt type.txt wordweight.txt for shuffling<br>
读取文件名 种类标签和点击特征文件，随机生成打乱的文件<br>
***

## tools/word：
1. tokenize<br>
load queryname.txt query_clickcout.txt to creat word.txt (word sorted by click times)<br>
读取词组名 词组点击次数文件，生成词文件，按照词的点击量排序

2. tf-idf<br>
to create tf-idf file<br>
读取每张图片的非空词组点击次数文件（行号是图片的索引，每行格式为：query索引 点击次数），词组名文件，选出的词组文件。 生成每张图片的词频文件，然后读取词频文件，生成每张图片的tf-idf文件。
***

