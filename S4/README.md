# Problem Statement:
Train network for mnist dataset for below targets:
1. 99.4% (this must be consistently shown in your last few epochs, and not a one-time achievement)
2. Less than or equal to 15 Epochs
3. Less than 10000 Parameters (additional points for doing this in less than 8000 pts)

## Step 1:
_Solution: [Notebook](https://github.com/Sudarshanpune/mnist_nn/blob/main/S4/EVA8_S4_Step1.ipynb) | [Colab link](https://colab.research.google.com/drive/1j3BSQhzV5lZ1OhvgsXUuwlYp6jZrWt2X?usp=sharing)_

### Targets:
1. Loading the dataset
2. Code set up
3. Dataset Inspection
4. Ploting the test, train accuracies and losses

### Results:
1. Train accuracy: 99.95%
2. Test accuracy: 99.31%
3. Parameters: 6.3M

### Analysis:
1. Extremely Heavy Model
2. Model is over-fitting

## Step 2:
_Solution: [Notebook](https://github.com/Sudarshanpune/mnist_nn/blob/main/S4/EVA8_S4_Step2.ipynb) | [Colab link](https://colab.research.google.com/drive/1afyrs0Z_hG5skQU4mKErDVyGz4Bi5Raa?usp=sharing)_

### Changes:
1. Improved the network with less parameters

### Results:
1. Train accuracy: 99.34%
2. Test accuracy: 98.39%
3. Parameters: 7930

### Analysis:
1. Test accuracy decreased with less parameters

## Step 3:
_Solution: [Notebook](https://github.com/Sudarshanpune/mnist_nn/blob/main/S4/EVA8_S4_Step3.ipynb) | [Colab link](https://colab.research.google.com/drive/1Xk8pw7_121PlPM4FoenOKqkUIKZRvVna?usp=sharing)_

### Changes:
1. Added BatchNormalization
2. Added Dropout of 0.05

### Results:
1. Train accuracy: 98.81%
2. Test accuracy: 99.32%
3. Parameters: 8,870

### Analysis:
1. Improved accuracy with less parameters

## Step 4:
_Solution: [Notebook](https://github.com/Sudarshanpune/mnist_nn/blob/main/S4/EVA8_S4_Step4.ipynb) | [Colab link](https://colab.research.google.com/drive/1ekp6hzEuH9GC6LCzAn1E9KF2SUr0c_DE?usp=sharing)_

### Changes:
1. Added RandomRotation to training dataset
2. Added StepLR of step_size = 5
3. Changed Dropout of 0.03

### Results:
1. Train accuracy: 98.87%
2. Test accuracy: 99.48%
3. Parameters: 8,870

### Analysis:
1. Achieved minimum of 99.4% accuracy for continous 5 epochs (Epoch6 - Epoch 14)
2. Model is under-fitting

**Network:**
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 28, 28]              80
              ReLU-2            [-1, 8, 28, 28]               0
       BatchNorm2d-3            [-1, 8, 28, 28]              16
           Dropout-4            [-1, 8, 28, 28]               0
            Conv2d-5           [-1, 16, 28, 28]           1,168
              ReLU-6           [-1, 16, 28, 28]               0
       BatchNorm2d-7           [-1, 16, 28, 28]              32
           Dropout-8           [-1, 16, 28, 28]               0
         MaxPool2d-9           [-1, 16, 14, 14]               0
           Conv2d-10           [-1, 16, 14, 14]           2,320
             ReLU-11           [-1, 16, 14, 14]               0
      BatchNorm2d-12           [-1, 16, 14, 14]              32
          Dropout-13           [-1, 16, 14, 14]               0
           Conv2d-14            [-1, 8, 14, 14]           1,160
             ReLU-15            [-1, 8, 14, 14]               0
      BatchNorm2d-16            [-1, 8, 14, 14]              16
          Dropout-17            [-1, 8, 14, 14]               0
        MaxPool2d-18              [-1, 8, 7, 7]               0
           Conv2d-19             [-1, 16, 5, 5]           1,168
             ReLU-20             [-1, 16, 5, 5]               0
      BatchNorm2d-21             [-1, 16, 5, 5]              32
          Dropout-22             [-1, 16, 5, 5]               0
           Conv2d-23             [-1, 16, 3, 3]           2,320
             ReLU-24             [-1, 16, 3, 3]               0
      BatchNorm2d-25             [-1, 16, 3, 3]              32
          Dropout-26             [-1, 16, 3, 3]               0
           Conv2d-27             [-1, 16, 3, 3]             272
             ReLU-28             [-1, 16, 3, 3]               0
      BatchNorm2d-29             [-1, 16, 3, 3]              32
          Dropout-30             [-1, 16, 3, 3]               0
           Conv2d-31             [-1, 10, 3, 3]             170
             ReLU-32             [-1, 10, 3, 3]               0
      BatchNorm2d-33             [-1, 10, 3, 3]              20
          Dropout-34             [-1, 10, 3, 3]               0
        AvgPool2d-35             [-1, 10, 1, 1]               0
================================================================
Total params: 8,870
Trainable params: 8,870
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.77
Params size (MB): 0.03
Estimated Total Size (MB): 0.81
----------------------------------------------------------------
```
**Logs of above network**
```
Epoch=0 Loss=0.15534280240535736 Batch_id=937 Accuracy=92.95: 100%|██████████| 938/938 [01:15<00:00, 12.45it/s]

Test set: Average loss: 0.0755, Accuracy: 9804/10000 (98.04%)

Epoch=1 Loss=0.1568378508090973 Batch_id=937 Accuracy=97.44: 100%|██████████| 938/938 [01:16<00:00, 12.27it/s]

Test set: Average loss: 0.0400, Accuracy: 9895/10000 (98.95%)

Epoch=2 Loss=0.06198321282863617 Batch_id=937 Accuracy=97.83: 100%|██████████| 938/938 [01:14<00:00, 12.53it/s]

Test set: Average loss: 0.0328, Accuracy: 9894/10000 (98.94%)

Epoch=3 Loss=0.019678112119436264 Batch_id=937 Accuracy=98.02: 100%|██████████| 938/938 [01:15<00:00, 12.46it/s]

Test set: Average loss: 0.0279, Accuracy: 9912/10000 (99.12%)

Epoch=4 Loss=0.049531400203704834 Batch_id=937 Accuracy=98.37: 100%|██████████| 938/938 [01:14<00:00, 12.60it/s]

Test set: Average loss: 0.0280, Accuracy: 9914/10000 (99.14%)

Epoch=5 Loss=0.2574065029621124 Batch_id=937 Accuracy=98.63: 100%|██████████| 938/938 [01:18<00:00, 12.00it/s]

Test set: Average loss: 0.0219, Accuracy: 9940/10000 (99.40%)

Epoch=6 Loss=0.011770752258598804 Batch_id=937 Accuracy=98.73: 100%|██████████| 938/938 [01:17<00:00, 12.04it/s]

Test set: Average loss: 0.0204, Accuracy: 9944/10000 (99.44%)

Epoch=7 Loss=0.03739934042096138 Batch_id=937 Accuracy=98.81: 100%|██████████| 938/938 [01:17<00:00, 12.09it/s]

Test set: Average loss: 0.0199, Accuracy: 9944/10000 (99.44%)

Epoch=8 Loss=0.016667358577251434 Batch_id=937 Accuracy=98.85: 100%|██████████| 938/938 [01:16<00:00, 12.24it/s]

Test set: Average loss: 0.0200, Accuracy: 9946/10000 (99.46%)

Epoch=9 Loss=0.004943458829075098 Batch_id=937 Accuracy=98.80: 100%|██████████| 938/938 [01:16<00:00, 12.27it/s]

Test set: Average loss: 0.0193, Accuracy: 9944/10000 (99.44%)

Epoch=10 Loss=0.10803059488534927 Batch_id=937 Accuracy=98.84: 100%|██████████| 938/938 [01:16<00:00, 12.24it/s]

Test set: Average loss: 0.0191, Accuracy: 9948/10000 (99.48%)

Epoch=11 Loss=0.027243567630648613 Batch_id=937 Accuracy=98.87: 100%|██████████| 938/938 [01:15<00:00, 12.37it/s]

Test set: Average loss: 0.0201, Accuracy: 9947/10000 (99.47%)

Epoch=12 Loss=0.030809050425887108 Batch_id=937 Accuracy=98.76: 100%|██████████| 938/938 [01:15<00:00, 12.42it/s]

Test set: Average loss: 0.0195, Accuracy: 9948/10000 (99.48%)

Epoch=13 Loss=0.06629624217748642 Batch_id=937 Accuracy=98.86: 100%|██████████| 938/938 [01:16<00:00, 12.21it/s]

Test set: Average loss: 0.0205, Accuracy: 9941/10000 (99.41%)

Epoch=14 Loss=0.05938711762428284 Batch_id=937 Accuracy=98.85: 100%|██████████| 938/938 [01:17<00:00, 12.04it/s]

Test set: Average loss: 0.0197, Accuracy: 9942/10000 (99.42%)
```

**Loss Plot**

<img width="898" alt="loss_graph" src="https://user-images.githubusercontent.com/32274516/213906038-56c7d941-c8c4-4684-ab20-afbbbac273f9.png">
