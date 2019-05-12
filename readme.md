# facial keypoints detection

## 参考自https://github.com/ewrfcas/Machine-Learning-Toolbox/blob/master/CNN_facial_keypoint_detection_ex.ipynb
## 目录结构及含义
````
facial-keypoints-detection
    ckpts: 用于存放模型文件
            best_model.h5
    data: 用于存放kaggle人脸关键点检测数据集
            IdLookupTable.csv
            SampleSubmission.csv
            test.csv
            training.csv
    datasets.py: 用于加载训练数据集(进一步划分为8：2，训练集和验证集合)和测试数据集 \
    models.py: 网络结构文件
    train.py: 训练文件
    test.py: 测试文件
    readme.md
    requirements.txt
````


````
datasets download
https://www.kaggle.com/c/facial-keypoints-detection
````


