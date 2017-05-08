# caffe
caffe with some things for my experiment(class  classification with clickture data)<br>
这是一个为了完成我的实验，而添加过一些loss层，小工具和代码的caffe，具体添加的东西我会一项一项加上来介绍：<br>
***

## Cosine Loss：
*the objective function is definede as:<br>
Cosine Loss的目标函数定义如下：*<br>
![Image text](https://github.com/GYxiaOH/caffe/blob/master/CosineLoss.png)
—————

## tools/data：
to creat lmdb/hdf5 files：
生成LMDB/HDF5的文件，其中HDF5为了保证每个文件小于2G，每3000张图片生成一个HDF5文件
LMDB中，一个是train_val生成，一个只生成train


