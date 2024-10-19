# Falsetto-SVM

论文[SINGER-DEPENDENT FALSETTO DETECTION FOR LIVE VOCAL PROCESSING BASED ON SUPPORT VECTOR CLASSIFICATION](https://ieeexplore.ieee.org/document/4176742) 的PyTorch实现

使用数据集[Dataset Card for Chest voice and Falsetto Dataset](https://www.modelscope.cn/datasets/ccmusic-database/chest_falsetto)

因为PyTorch并不适合去实现支持向量机（SVM），故本实现并没有使用核优化等技巧，本身论文年代久远也很难保证其他条件相同，故本实现效果不佳，如果感兴趣可以使用SKlearn进行进一步研究

你可以使用`train.py`去训练一个SVM，在那之前使用`preprocess.py`进行预处理

可以使用'main.py'识别一些短音频是否为falsetto
