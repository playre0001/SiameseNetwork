# What's This?
Be able to run a CNN-based Siamese Network.
Depending on customization, it can be used for other data types.

# Requirement
* Python 3.X
* tensorflow 2.X

# QuickStart

1. Install Tensorflow on your device

Look [Tensorflow page](https://www.tensorflow.org/install), and install it.

2. Setup Dataset

Download Dogs-vs-Cats Dataset from [Kaggle page](https://www.kaggle.com/c/dogs-vs-cats).
Unzip the files as shown in the following structure.

```
-
|-SiameseNetwork.py
|-test_learning.py
|-dogs-vs-cats
   |-train
      |-train
        |-image A
        |-image B
        |-image C
        |-...
```

3. Do this program

Input the command bellow, and press Enter.

```
python test_learning.py
```

# Customize

To use classes in SiameseNetwork.py, please refer code in test_learning.py
