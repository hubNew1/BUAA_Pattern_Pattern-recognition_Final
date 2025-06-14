# BUAA_Pattern_Pattern-recognition_Final
北航模式识别大作业，这里使用的是Pytorch神经网络进行花朵的识别（这个数据集的效果并不好，Test文件夹中有很多污染，但是可以尝试将这个程序移植到其他数据集上）<br />
<br />
**数据从老师给的文件里找花朵的那个数据集**<br />
**记得修改对应的文件地址**<br />
希望能帮到大家<br />

Final_CM.ipynb是用来画图的程序，提供了绘制混淆矩阵的方法<br />
main.py是主程序，模型的定义和训练都在里面，如果需要单纯的调用模型进行识别，可以把trainloop注释掉，然后先运行加载模型参数，在调用evaluate

<br />
环境需要用到pytorch、opencv、numpy，可以用conda管理
