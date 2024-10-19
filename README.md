# Falsetto-SVM

Paper: [SINGER-DEPENDENT FALSETTO DETECTION FOR LIVE VOCAL PROCESSING BASED ON SUPPORT VECTOR CLASSIFICATION](https://ieeexplore.ieee.org/document/4176742) PyTorch Implementation

Using the dataset: [Dataset Card for Chest voice and Falsetto Dataset](https://www.modelscope.cn/datasets/ccmusic-database/chest_falsetto)

Since PyTorch is not particularly suited for implementing Support Vector Machines (SVMs), this implementation does not employ kernel optimization techniques. Given that the paper is quite old, it’s challenging to ensure that other conditions remain the same, so the performance of this implementation may not be optimal. If you are interested, you can use SKlearn for further research.

You can use train.py to train an SVM, but before doing so, you should use preprocess.py for preprocessing.

You can use main.py to identify whether some short audio clips are in falsetto
