
# Problem Statement:
Run this [network](https://colab.research.google.com/drive/1qlewMtxcAJT6fIJdmMh8pSf2e-dh51Rw) for cifar10 dataset and fix the code for below targets:
1. change the architecture to C1C2C3C40 (No MaxPooling, but 3 3x3 layers with stride of 2 instead)
2. total RF must be more than 44
3. one of the layers must use Depthwise Separable Convolution
4. one of the layers must use Dilated Convolution
5. use GAP (compulsory):- add FC after GAP to target #of classes (optional)
6. use albumentation library and apply:
   - horizontal flip
   - shiftScaleRotate
   - coarseDropout (max_holes = 1, max_height=16px, max_width=1, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset),    mask_fill_value = None)
7. achieve 85% accuracy, as many epochs as you want. Total Params to be less than 200k.

**Network:**
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 32, 32]             864
              ReLU-2           [-1, 32, 32, 32]               0
       BatchNorm2d-3           [-1, 32, 32, 32]              64
           Dropout-4           [-1, 32, 32, 32]               0
            Conv2d-5           [-1, 64, 32, 32]          18,432
              ReLU-6           [-1, 64, 32, 32]               0
       BatchNorm2d-7           [-1, 64, 32, 32]             128
           Dropout-8           [-1, 64, 32, 32]               0
            Conv2d-9           [-1, 64, 32, 32]          36,864
             ReLU-10           [-1, 64, 32, 32]               0
      BatchNorm2d-11           [-1, 64, 32, 32]             128
          Dropout-12           [-1, 64, 32, 32]               0
           Conv2d-13           [-1, 64, 16, 16]          36,864
             ReLU-14           [-1, 64, 16, 16]               0
      BatchNorm2d-15           [-1, 64, 16, 16]             128
          Dropout-16           [-1, 64, 16, 16]               0
           Conv2d-17           [-1, 64, 16, 16]             576
           Conv2d-18           [-1, 64, 16, 16]           4,096
             ReLU-19           [-1, 64, 16, 16]               0
      BatchNorm2d-20           [-1, 64, 16, 16]             128
          Dropout-21           [-1, 64, 16, 16]               0
           Conv2d-22           [-1, 64, 16, 16]             576
           Conv2d-23           [-1, 64, 16, 16]           4,096
             ReLU-24           [-1, 64, 16, 16]               0
      BatchNorm2d-25           [-1, 64, 16, 16]             128
          Dropout-26           [-1, 64, 16, 16]               0
           Conv2d-27             [-1, 64, 8, 8]          36,864
             ReLU-28             [-1, 64, 8, 8]               0
      BatchNorm2d-29             [-1, 64, 8, 8]             128
          Dropout-30             [-1, 64, 8, 8]               0
           Conv2d-31             [-1, 64, 6, 6]          36,864
             ReLU-32             [-1, 64, 6, 6]               0
      BatchNorm2d-33             [-1, 64, 6, 6]             128
          Dropout-34             [-1, 64, 6, 6]               0
           Conv2d-35             [-1, 32, 4, 4]          18,432
             ReLU-36             [-1, 32, 4, 4]               0
      BatchNorm2d-37             [-1, 32, 4, 4]              64
          Dropout-38             [-1, 32, 4, 4]               0
           Conv2d-39             [-1, 10, 4, 4]           2,880
        AvgPool2d-40             [-1, 10, 1, 1]               0
================================================================
Total params: 198,432
Trainable params: 198,432
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 6.96
Params size (MB): 0.76
Estimated Total Size (MB): 7.73
----------------------------------------------------------------
```
**Logs of above network**
```
Epoch = 0, Loss = 1.4227, Batch = 390, Accuracy = 41.17: 100%|██████████| 391/391 [00:15<00:00, 24.96it/s]
Test set: Average loss = 0.0101, Accuracy = 5370/10000 (53.70%)

Epoch = 1, Loss = 1.3688, Batch = 390, Accuracy = 53.56: 100%|██████████| 391/391 [00:16<00:00, 23.62it/s]
Test set: Average loss = 0.0078, Accuracy = 6374/10000 (63.74%)

Epoch = 2, Loss = 1.1159, Batch = 390, Accuracy = 58.77: 100%|██████████| 391/391 [00:15<00:00, 24.80it/s]
Test set: Average loss = 0.0066, Accuracy = 6991/10000 (69.91%)

Epoch = 3, Loss = 1.1855, Batch = 390, Accuracy = 62.46: 100%|██████████| 391/391 [00:15<00:00, 24.55it/s]
Test set: Average loss = 0.0061, Accuracy = 7281/10000 (72.81%)

Epoch = 4, Loss = 0.9409, Batch = 390, Accuracy = 64.69: 100%|██████████| 391/391 [00:15<00:00, 25.06it/s]
Test set: Average loss = 0.0056, Accuracy = 7553/10000 (75.53%)

Epoch = 5, Loss = 1.0176, Batch = 390, Accuracy = 66.45: 100%|██████████| 391/391 [00:15<00:00, 24.57it/s]
Test set: Average loss = 0.0054, Accuracy = 7670/10000 (76.70%)

Epoch = 6, Loss = 0.9621, Batch = 390, Accuracy = 67.88: 100%|██████████| 391/391 [00:15<00:00, 24.91it/s]
Test set: Average loss = 0.0054, Accuracy = 7659/10000 (76.59%)

Epoch = 7, Loss = 0.6548, Batch = 390, Accuracy = 69.30: 100%|██████████| 391/391 [00:16<00:00, 24.03it/s]
Test set: Average loss = 0.0049, Accuracy = 7830/10000 (78.30%)

Epoch = 8, Loss = 1.1442, Batch = 390, Accuracy = 70.20: 100%|██████████| 391/391 [00:15<00:00, 24.99it/s]
Test set: Average loss = 0.0049, Accuracy = 7829/10000 (78.29%)

Epoch = 9, Loss = 0.7886, Batch = 390, Accuracy = 70.91: 100%|██████████| 391/391 [00:16<00:00, 23.84it/s]
Test set: Average loss = 0.0047, Accuracy = 7980/10000 (79.80%)

Epoch = 10, Loss = 0.6569, Batch = 390, Accuracy = 71.50: 100%|██████████| 391/391 [00:15<00:00, 25.16it/s]
Test set: Average loss = 0.0044, Accuracy = 8019/10000 (80.19%)

Epoch = 11, Loss = 0.5558, Batch = 390, Accuracy = 72.50: 100%|██████████| 391/391 [00:16<00:00, 24.13it/s]
Test set: Average loss = 0.0045, Accuracy = 8026/10000 (80.26%)

Epoch = 12, Loss = 0.6939, Batch = 390, Accuracy = 72.97: 100%|██████████| 391/391 [00:15<00:00, 25.07it/s]
Test set: Average loss = 0.0043, Accuracy = 8090/10000 (80.90%)

Epoch = 13, Loss = 0.5782, Batch = 390, Accuracy = 73.66: 100%|██████████| 391/391 [00:16<00:00, 23.44it/s]
Test set: Average loss = 0.0042, Accuracy = 8221/10000 (82.21%)

Epoch = 14, Loss = 0.8848, Batch = 390, Accuracy = 74.11: 100%|██████████| 391/391 [00:15<00:00, 24.85it/s]
Test set: Average loss = 0.0041, Accuracy = 8197/10000 (81.97%)

Epoch = 15, Loss = 0.7777, Batch = 390, Accuracy = 74.54: 100%|██████████| 391/391 [00:16<00:00, 23.95it/s]
Test set: Average loss = 0.0040, Accuracy = 8300/10000 (83.00%)

Epoch = 16, Loss = 1.0402, Batch = 390, Accuracy = 75.10: 100%|██████████| 391/391 [00:15<00:00, 24.97it/s]
Test set: Average loss = 0.0040, Accuracy = 8262/10000 (82.62%)

Epoch = 17, Loss = 0.6918, Batch = 390, Accuracy = 75.20: 100%|██████████| 391/391 [00:15<00:00, 24.46it/s]
Test set: Average loss = 0.0039, Accuracy = 8281/10000 (82.81%)

Epoch = 18, Loss = 0.7063, Batch = 390, Accuracy = 75.80: 100%|██████████| 391/391 [00:15<00:00, 25.00it/s]
Test set: Average loss = 0.0038, Accuracy = 8359/10000 (83.59%)

Epoch = 19, Loss = 0.4812, Batch = 390, Accuracy = 76.03: 100%|██████████| 391/391 [00:15<00:00, 24.51it/s]
Test set: Average loss = 0.0037, Accuracy = 8383/10000 (83.83%)

Epoch = 20, Loss = 0.6754, Batch = 390, Accuracy = 76.21: 100%|██████████| 391/391 [00:15<00:00, 24.66it/s]
Test set: Average loss = 0.0037, Accuracy = 8421/10000 (84.21%)

Epoch = 21, Loss = 0.5297, Batch = 390, Accuracy = 76.83: 100%|██████████| 391/391 [00:15<00:00, 24.58it/s]
Test set: Average loss = 0.0037, Accuracy = 8431/10000 (84.31%)

Epoch = 22, Loss = 0.7731, Batch = 390, Accuracy = 77.01: 100%|██████████| 391/391 [00:15<00:00, 25.00it/s]
Test set: Average loss = 0.0035, Accuracy = 8477/10000 (84.77%)

Epoch = 23, Loss = 0.7549, Batch = 390, Accuracy = 77.27: 100%|██████████| 391/391 [00:15<00:00, 24.94it/s]
Test set: Average loss = 0.0037, Accuracy = 8446/10000 (84.46%)

Epoch = 24, Loss = 0.5016, Batch = 390, Accuracy = 77.54: 100%|██████████| 391/391 [00:15<00:00, 25.06it/s]
Test set: Average loss = 0.0035, Accuracy = 8525/10000 (85.25%)

Epoch = 25, Loss = 0.5808, Batch = 390, Accuracy = 77.40: 100%|██████████| 391/391 [00:15<00:00, 24.83it/s]
Test set: Average loss = 0.0035, Accuracy = 8509/10000 (85.09%)

Epoch = 26, Loss = 0.6874, Batch = 390, Accuracy = 78.03: 100%|██████████| 391/391 [00:15<00:00, 24.87it/s]
Test set: Average loss = 0.0035, Accuracy = 8497/10000 (84.97%)

Epoch = 27, Loss = 0.6404, Batch = 390, Accuracy = 78.14: 100%|██████████| 391/391 [00:15<00:00, 24.98it/s]
Test set: Average loss = 0.0035, Accuracy = 8584/10000 (85.84%)

Epoch = 28, Loss = 0.6453, Batch = 390, Accuracy = 78.39: 100%|██████████| 391/391 [00:15<00:00, 24.84it/s]
Test set: Average loss = 0.0033, Accuracy = 8571/10000 (85.71%)

Epoch = 29, Loss = 0.5735, Batch = 390, Accuracy = 78.46: 100%|██████████| 391/391 [00:15<00:00, 24.99it/s]
Test set: Average loss = 0.0033, Accuracy = 8580/10000 (85.80%)

Epoch = 30, Loss = 0.6780, Batch = 390, Accuracy = 78.74: 100%|██████████| 391/391 [00:16<00:00, 23.98it/s]
Test set: Average loss = 0.0034, Accuracy = 8563/10000 (85.63%)

Epoch = 31, Loss = 0.6527, Batch = 390, Accuracy = 78.63: 100%|██████████| 391/391 [00:15<00:00, 25.01it/s]
Test set: Average loss = 0.0033, Accuracy = 8586/10000 (85.86%)

Epoch = 32, Loss = 0.6348, Batch = 390, Accuracy = 79.21: 100%|██████████| 391/391 [00:16<00:00, 24.14it/s]
Test set: Average loss = 0.0033, Accuracy = 8586/10000 (85.86%)

Epoch = 33, Loss = 0.7777, Batch = 390, Accuracy = 79.39: 100%|██████████| 391/391 [00:15<00:00, 24.71it/s]
Test set: Average loss = 0.0033, Accuracy = 8580/10000 (85.80%)

Epoch = 34, Loss = 0.6717, Batch = 390, Accuracy = 79.28: 100%|██████████| 391/391 [00:16<00:00, 24.03it/s]
Test set: Average loss = 0.0033, Accuracy = 8579/10000 (85.79%)

Epoch = 35, Loss = 0.4863, Batch = 390, Accuracy = 79.29: 100%|██████████| 391/391 [00:15<00:00, 24.86it/s]
Test set: Average loss = 0.0032, Accuracy = 8625/10000 (86.25%)

Epoch = 36, Loss = 0.5628, Batch = 390, Accuracy = 79.48: 100%|██████████| 391/391 [00:16<00:00, 23.90it/s]
Test set: Average loss = 0.0032, Accuracy = 8648/10000 (86.48%)

Epoch = 37, Loss = 0.6065, Batch = 390, Accuracy = 79.98: 100%|██████████| 391/391 [00:15<00:00, 25.24it/s]
Test set: Average loss = 0.0032, Accuracy = 8626/10000 (86.26%)

Epoch = 38, Loss = 0.5573, Batch = 390, Accuracy = 79.75: 100%|██████████| 391/391 [00:15<00:00, 24.70it/s]
Test set: Average loss = 0.0031, Accuracy = 8665/10000 (86.65%)

Epoch = 39, Loss = 0.5328, Batch = 390, Accuracy = 79.84: 100%|██████████| 391/391 [00:15<00:00, 25.03it/s]
Test set: Average loss = 0.0032, Accuracy = 8622/10000 (86.22%)

Epoch = 40, Loss = 0.5623, Batch = 390, Accuracy = 80.21: 100%|██████████| 391/391 [00:15<00:00, 25.04it/s]
Test set: Average loss = 0.0030, Accuracy = 8676/10000 (86.76%)

Epoch = 41, Loss = 0.7082, Batch = 390, Accuracy = 80.36: 100%|██████████| 391/391 [00:15<00:00, 25.10it/s]
Test set: Average loss = 0.0031, Accuracy = 8677/10000 (86.77%)

Epoch = 42, Loss = 0.3653, Batch = 390, Accuracy = 80.23: 100%|██████████| 391/391 [00:15<00:00, 25.06it/s]
Test set: Average loss = 0.0031, Accuracy = 8690/10000 (86.90%)

Epoch = 43, Loss = 0.7267, Batch = 390, Accuracy = 80.40: 100%|██████████| 391/391 [00:15<00:00, 24.81it/s]
Test set: Average loss = 0.0031, Accuracy = 8683/10000 (86.83%)

Epoch = 44, Loss = 0.5439, Batch = 390, Accuracy = 80.66: 100%|██████████| 391/391 [00:15<00:00, 25.20it/s]
Test set: Average loss = 0.0030, Accuracy = 8722/10000 (87.22%)

Epoch = 45, Loss = 0.5662, Batch = 390, Accuracy = 80.83: 100%|██████████| 391/391 [00:15<00:00, 25.01it/s]
Test set: Average loss = 0.0030, Accuracy = 8713/10000 (87.13%)

Epoch = 46, Loss = 0.6731, Batch = 390, Accuracy = 80.91: 100%|██████████| 391/391 [00:15<00:00, 25.13it/s]
Test set: Average loss = 0.0030, Accuracy = 8726/10000 (87.26%)

Epoch = 47, Loss = 0.5916, Batch = 390, Accuracy = 81.32: 100%|██████████| 391/391 [00:16<00:00, 24.33it/s]
Test set: Average loss = 0.0030, Accuracy = 8725/10000 (87.25%)

Epoch = 48, Loss = 0.4305, Batch = 390, Accuracy = 81.37: 100%|██████████| 391/391 [00:15<00:00, 24.96it/s]
Test set: Average loss = 0.0030, Accuracy = 8704/10000 (87.04%)

Epoch = 49, Loss = 0.6760, Batch = 390, Accuracy = 81.15: 100%|██████████| 391/391 [00:16<00:00, 23.98it/s]
Test set: Average loss = 0.0029, Accuracy = 8730/10000 (87.30%)

Epoch = 50, Loss = 0.6682, Batch = 390, Accuracy = 81.50: 100%|██████████| 391/391 [00:15<00:00, 24.86it/s]
Test set: Average loss = 0.0030, Accuracy = 8768/10000 (87.68%)

Epoch = 51, Loss = 0.5504, Batch = 390, Accuracy = 81.53: 100%|██████████| 391/391 [00:16<00:00, 23.74it/s]
Test set: Average loss = 0.0029, Accuracy = 8743/10000 (87.43%)

Epoch = 52, Loss = 0.5273, Batch = 390, Accuracy = 81.51: 100%|██████████| 391/391 [00:15<00:00, 24.88it/s]
Test set: Average loss = 0.0029, Accuracy = 8743/10000 (87.43%)

Epoch = 53, Loss = 0.5868, Batch = 390, Accuracy = 81.59: 100%|██████████| 391/391 [00:16<00:00, 24.31it/s]
Test set: Average loss = 0.0029, Accuracy = 8779/10000 (87.79%)

Epoch = 54, Loss = 0.5968, Batch = 390, Accuracy = 81.57: 100%|██████████| 391/391 [00:16<00:00, 24.12it/s]
Test set: Average loss = 0.0029, Accuracy = 8768/10000 (87.68%)

Epoch = 55, Loss = 0.6088, Batch = 390, Accuracy = 81.88: 100%|██████████| 391/391 [00:16<00:00, 23.42it/s]
Test set: Average loss = 0.0029, Accuracy = 8732/10000 (87.32%)

Epoch = 56, Loss = 0.5791, Batch = 390, Accuracy = 81.78: 100%|██████████| 391/391 [00:15<00:00, 24.76it/s]
Test set: Average loss = 0.0029, Accuracy = 8750/10000 (87.50%)

Epoch = 57, Loss = 0.4596, Batch = 390, Accuracy = 81.88: 100%|██████████| 391/391 [00:16<00:00, 23.64it/s]
Test set: Average loss = 0.0029, Accuracy = 8756/10000 (87.56%)

Epoch = 58, Loss = 0.4496, Batch = 390, Accuracy = 81.96: 100%|██████████| 391/391 [00:15<00:00, 24.65it/s]
Test set: Average loss = 0.0029, Accuracy = 8729/10000 (87.29%)

Epoch = 59, Loss = 0.4268, Batch = 390, Accuracy = 82.21: 100%|██████████| 391/391 [00:16<00:00, 24.09it/s]
Test set: Average loss = 0.0028, Accuracy = 8800/10000 (88.00%)

Epoch = 60, Loss = 0.5639, Batch = 390, Accuracy = 82.23: 100%|██████████| 391/391 [00:15<00:00, 24.73it/s]
Test set: Average loss = 0.0029, Accuracy = 8761/10000 (87.61%)

Epoch = 61, Loss = 0.4661, Batch = 390, Accuracy = 82.55: 100%|██████████| 391/391 [00:16<00:00, 24.39it/s]
Test set: Average loss = 0.0029, Accuracy = 8748/10000 (87.48%)

Epoch = 62, Loss = 0.5571, Batch = 390, Accuracy = 82.27: 100%|██████████| 391/391 [00:15<00:00, 25.02it/s]
Test set: Average loss = 0.0029, Accuracy = 8774/10000 (87.74%)

Epoch = 63, Loss = 0.5453, Batch = 390, Accuracy = 82.26: 100%|██████████| 391/391 [00:15<00:00, 25.08it/s]
Test set: Average loss = 0.0028, Accuracy = 8779/10000 (87.79%)

Epoch = 64, Loss = 0.3937, Batch = 390, Accuracy = 82.36: 100%|██████████| 391/391 [00:15<00:00, 24.99it/s]
Test set: Average loss = 0.0029, Accuracy = 8760/10000 (87.60%)

Epoch = 65, Loss = 0.6468, Batch = 390, Accuracy = 82.52: 100%|██████████| 391/391 [00:15<00:00, 24.84it/s]
Test set: Average loss = 0.0028, Accuracy = 8810/10000 (88.10%)

Epoch = 66, Loss = 0.5249, Batch = 390, Accuracy = 82.46: 100%|██████████| 391/391 [00:15<00:00, 25.05it/s]
Test set: Average loss = 0.0028, Accuracy = 8790/10000 (87.90%)

Epoch = 67, Loss = 0.5992, Batch = 390, Accuracy = 82.63: 100%|██████████| 391/391 [00:15<00:00, 24.74it/s]
Test set: Average loss = 0.0028, Accuracy = 8788/10000 (87.88%)

Epoch = 68, Loss = 0.7023, Batch = 390, Accuracy = 82.74: 100%|██████████| 391/391 [00:15<00:00, 24.55it/s]
Test set: Average loss = 0.0028, Accuracy = 8817/10000 (88.17%)

Epoch = 69, Loss = 0.4750, Batch = 390, Accuracy = 82.62: 100%|██████████| 391/391 [00:15<00:00, 25.11it/s]
Test set: Average loss = 0.0027, Accuracy = 8807/10000 (88.07%)

Epoch = 70, Loss = 0.3294, Batch = 390, Accuracy = 82.75: 100%|██████████| 391/391 [00:16<00:00, 23.72it/s]
Test set: Average loss = 0.0027, Accuracy = 8830/10000 (88.30%)

Epoch = 71, Loss = 0.6247, Batch = 390, Accuracy = 82.78: 100%|██████████| 391/391 [00:15<00:00, 24.94it/s]
Test set: Average loss = 0.0029, Accuracy = 8743/10000 (87.43%)

Epoch = 72, Loss = 0.5520, Batch = 390, Accuracy = 82.90: 100%|██████████| 391/391 [00:16<00:00, 23.96it/s]
Test set: Average loss = 0.0028, Accuracy = 8848/10000 (88.48%)

Epoch = 73, Loss = 0.4114, Batch = 390, Accuracy = 83.04: 100%|██████████| 391/391 [00:15<00:00, 24.97it/s]
Test set: Average loss = 0.0027, Accuracy = 8809/10000 (88.09%)

Epoch = 74, Loss = 0.3135, Batch = 390, Accuracy = 82.91: 100%|██████████| 391/391 [00:16<00:00, 23.49it/s]
Test set: Average loss = 0.0027, Accuracy = 8840/10000 (88.40%)

Epoch = 75, Loss = 0.4405, Batch = 390, Accuracy = 82.97: 100%|██████████| 391/391 [00:15<00:00, 24.49it/s]
Test set: Average loss = 0.0027, Accuracy = 8854/10000 (88.54%)

Epoch = 76, Loss = 0.5356, Batch = 390, Accuracy = 83.20: 100%|██████████| 391/391 [00:16<00:00, 23.78it/s]
Test set: Average loss = 0.0027, Accuracy = 8825/10000 (88.25%)

Epoch = 77, Loss = 0.5358, Batch = 390, Accuracy = 83.22: 100%|██████████| 391/391 [00:15<00:00, 24.92it/s]
Test set: Average loss = 0.0028, Accuracy = 8818/10000 (88.18%)

Epoch = 78, Loss = 0.5380, Batch = 390, Accuracy = 83.32: 100%|██████████| 391/391 [00:16<00:00, 24.05it/s]
Test set: Average loss = 0.0027, Accuracy = 8866/10000 (88.66%)

Epoch = 79, Loss = 0.4128, Batch = 390, Accuracy = 83.34: 100%|██████████| 391/391 [00:15<00:00, 24.46it/s]
Test set: Average loss = 0.0027, Accuracy = 8868/10000 (88.68%)

Epoch = 80, Loss = 0.4109, Batch = 390, Accuracy = 83.23: 100%|██████████| 391/391 [00:16<00:00, 24.25it/s]
Test set: Average loss = 0.0027, Accuracy = 8864/10000 (88.64%)

Epoch = 81, Loss = 0.3680, Batch = 390, Accuracy = 83.16: 100%|██████████| 391/391 [00:15<00:00, 24.73it/s]
Test set: Average loss = 0.0027, Accuracy = 8840/10000 (88.40%)

Epoch = 82, Loss = 0.4048, Batch = 390, Accuracy = 83.44: 100%|██████████| 391/391 [00:15<00:00, 24.88it/s]
Test set: Average loss = 0.0027, Accuracy = 8866/10000 (88.66%)

Epoch = 83, Loss = 0.3379, Batch = 390, Accuracy = 83.42: 100%|██████████| 391/391 [00:15<00:00, 24.82it/s]
Test set: Average loss = 0.0027, Accuracy = 8856/10000 (88.56%)

Epoch = 84, Loss = 0.3362, Batch = 390, Accuracy = 83.60: 100%|██████████| 391/391 [00:15<00:00, 24.61it/s]
Test set: Average loss = 0.0027, Accuracy = 8877/10000 (88.77%)

Epoch = 85, Loss = 0.4103, Batch = 390, Accuracy = 83.36: 100%|██████████| 391/391 [00:15<00:00, 24.93it/s]
Test set: Average loss = 0.0028, Accuracy = 8820/10000 (88.20%)

Epoch = 86, Loss = 0.3937, Batch = 390, Accuracy = 83.48: 100%|██████████| 391/391 [00:15<00:00, 24.82it/s]
Test set: Average loss = 0.0028, Accuracy = 8807/10000 (88.07%)

Epoch = 87, Loss = 0.3736, Batch = 390, Accuracy = 83.67: 100%|██████████| 391/391 [00:15<00:00, 24.85it/s]
Test set: Average loss = 0.0027, Accuracy = 8862/10000 (88.62%)

Epoch = 88, Loss = 0.4027, Batch = 390, Accuracy = 83.60: 100%|██████████| 391/391 [00:15<00:00, 24.55it/s]
Test set: Average loss = 0.0027, Accuracy = 8875/10000 (88.75%)

Epoch = 89, Loss = 0.4555, Batch = 390, Accuracy = 83.67: 100%|██████████| 391/391 [00:15<00:00, 24.79it/s]
Test set: Average loss = 0.0027, Accuracy = 8837/10000 (88.37%)

Epoch = 90, Loss = 0.5357, Batch = 390, Accuracy = 83.74: 100%|██████████| 391/391 [00:15<00:00, 25.00it/s]
Test set: Average loss = 0.0026, Accuracy = 8886/10000 (88.86%)

Epoch = 91, Loss = 0.4477, Batch = 390, Accuracy = 84.05: 100%|██████████| 391/391 [00:15<00:00, 24.79it/s]
Test set: Average loss = 0.0026, Accuracy = 8875/10000 (88.75%)

Epoch = 92, Loss = 0.4408, Batch = 390, Accuracy = 83.91: 100%|██████████| 391/391 [00:15<00:00, 24.96it/s]
Test set: Average loss = 0.0026, Accuracy = 8874/10000 (88.74%)

Epoch = 93, Loss = 0.4478, Batch = 390, Accuracy = 84.09: 100%|██████████| 391/391 [00:16<00:00, 24.32it/s]
Test set: Average loss = 0.0026, Accuracy = 8893/10000 (88.93%)

Epoch = 94, Loss = 0.4043, Batch = 390, Accuracy = 83.74: 100%|██████████| 391/391 [00:15<00:00, 24.76it/s]
Test set: Average loss = 0.0025, Accuracy = 8899/10000 (88.99%)

Epoch = 95, Loss = 0.3656, Batch = 390, Accuracy = 83.80: 100%|██████████| 391/391 [00:16<00:00, 24.19it/s]
Test set: Average loss = 0.0026, Accuracy = 8889/10000 (88.89%)

Epoch = 96, Loss = 0.5057, Batch = 390, Accuracy = 83.91: 100%|██████████| 391/391 [00:15<00:00, 24.81it/s]
Test set: Average loss = 0.0026, Accuracy = 8923/10000 (89.23%)

Epoch = 97, Loss = 0.4567, Batch = 390, Accuracy = 84.04: 100%|██████████| 391/391 [00:16<00:00, 24.02it/s]
Test set: Average loss = 0.0026, Accuracy = 8900/10000 (89.00%)

Epoch = 98, Loss = 0.5147, Batch = 390, Accuracy = 84.23: 100%|██████████| 391/391 [00:15<00:00, 24.54it/s]
Test set: Average loss = 0.0026, Accuracy = 8897/10000 (88.97%)

Epoch = 99, Loss = 0.3929, Batch = 390, Accuracy = 83.97: 100%|██████████| 391/391 [00:16<00:00, 23.82it/s]
Test set: Average loss = 0.0027, Accuracy = 8882/10000 (88.82%)

Epoch = 100, Loss = 0.5280, Batch = 390, Accuracy = 84.34: 100%|██████████| 391/391 [00:15<00:00, 24.55it/s]
Test set: Average loss = 0.0026, Accuracy = 8913/10000 (89.13%)

Epoch = 101, Loss = 0.4904, Batch = 390, Accuracy = 84.12: 100%|██████████| 391/391 [00:16<00:00, 23.77it/s]
Test set: Average loss = 0.0027, Accuracy = 8894/10000 (88.94%)

Epoch = 102, Loss = 0.5258, Batch = 390, Accuracy = 84.34: 100%|██████████| 391/391 [00:15<00:00, 24.85it/s]
Test set: Average loss = 0.0025, Accuracy = 8892/10000 (88.92%)

Epoch = 103, Loss = 0.4307, Batch = 390, Accuracy = 84.12: 100%|██████████| 391/391 [00:16<00:00, 23.67it/s]
Test set: Average loss = 0.0025, Accuracy = 8905/10000 (89.05%)

Epoch = 104, Loss = 0.4051, Batch = 390, Accuracy = 84.30: 100%|██████████| 391/391 [00:15<00:00, 24.56it/s]
Test set: Average loss = 0.0025, Accuracy = 8918/10000 (89.18%)

Epoch = 105, Loss = 0.3890, Batch = 390, Accuracy = 84.50: 100%|██████████| 391/391 [00:16<00:00, 23.67it/s]
Test set: Average loss = 0.0026, Accuracy = 8920/10000 (89.20%)

Epoch = 106, Loss = 0.3455, Batch = 390, Accuracy = 84.44: 100%|██████████| 391/391 [00:15<00:00, 24.75it/s]
Test set: Average loss = 0.0026, Accuracy = 8936/10000 (89.36%)

Epoch = 107, Loss = 0.4730, Batch = 390, Accuracy = 84.55: 100%|██████████| 391/391 [00:16<00:00, 23.64it/s]
Test set: Average loss = 0.0025, Accuracy = 8923/10000 (89.23%)

Epoch = 108, Loss = 0.5116, Batch = 390, Accuracy = 84.04: 100%|██████████| 391/391 [00:15<00:00, 24.47it/s]
Test set: Average loss = 0.0026, Accuracy = 8903/10000 (89.03%)

Epoch = 109, Loss = 0.4187, Batch = 390, Accuracy = 84.55: 100%|██████████| 391/391 [00:16<00:00, 23.98it/s]
Test set: Average loss = 0.0026, Accuracy = 8919/10000 (89.19%)

Epoch = 110, Loss = 0.5017, Batch = 390, Accuracy = 84.71: 100%|██████████| 391/391 [00:15<00:00, 24.58it/s]
Test set: Average loss = 0.0027, Accuracy = 8882/10000 (88.82%)

Epoch = 111, Loss = 0.6176, Batch = 390, Accuracy = 84.39: 100%|██████████| 391/391 [00:16<00:00, 24.36it/s]
Test set: Average loss = 0.0026, Accuracy = 8897/10000 (88.97%)

Epoch = 112, Loss = 0.4061, Batch = 390, Accuracy = 84.84: 100%|██████████| 391/391 [00:15<00:00, 24.61it/s]
Test set: Average loss = 0.0026, Accuracy = 8937/10000 (89.37%)

Epoch = 113, Loss = 0.5865, Batch = 390, Accuracy = 84.65: 100%|██████████| 391/391 [00:16<00:00, 24.32it/s]
Test set: Average loss = 0.0026, Accuracy = 8906/10000 (89.06%)

Epoch = 114, Loss = 0.4258, Batch = 390, Accuracy = 84.87: 100%|██████████| 391/391 [00:15<00:00, 24.83it/s]
Test set: Average loss = 0.0025, Accuracy = 8930/10000 (89.30%)

Epoch = 115, Loss = 0.6009, Batch = 390, Accuracy = 84.71: 100%|██████████| 391/391 [00:16<00:00, 24.24it/s]
Test set: Average loss = 0.0025, Accuracy = 8932/10000 (89.32%)

Epoch = 116, Loss = 0.4344, Batch = 390, Accuracy = 84.76: 100%|██████████| 391/391 [00:15<00:00, 24.74it/s]
Test set: Average loss = 0.0025, Accuracy = 8943/10000 (89.43%)

Epoch = 117, Loss = 0.3910, Batch = 390, Accuracy = 84.56: 100%|██████████| 391/391 [00:15<00:00, 24.64it/s]
Test set: Average loss = 0.0025, Accuracy = 8930/10000 (89.30%)

Epoch = 118, Loss = 0.2941, Batch = 390, Accuracy = 84.97: 100%|██████████| 391/391 [00:15<00:00, 24.71it/s]
Test set: Average loss = 0.0025, Accuracy = 8951/10000 (89.51%)

Epoch = 119, Loss = 0.7020, Batch = 390, Accuracy = 84.82: 100%|██████████| 391/391 [00:15<00:00, 24.75it/s]
Test set: Average loss = 0.0026, Accuracy = 8910/10000 (89.10%)

Epoch = 120, Loss = 0.6599, Batch = 390, Accuracy = 84.83: 100%|██████████| 391/391 [00:15<00:00, 24.54it/s]
Test set: Average loss = 0.0025, Accuracy = 8940/10000 (89.40%)

Epoch = 121, Loss = 0.4780, Batch = 390, Accuracy = 84.74: 100%|██████████| 391/391 [00:15<00:00, 24.62it/s]
Test set: Average loss = 0.0026, Accuracy = 8927/10000 (89.27%)

Epoch = 122, Loss = 0.6432, Batch = 390, Accuracy = 85.01: 100%|██████████| 391/391 [00:15<00:00, 24.68it/s]
Test set: Average loss = 0.0025, Accuracy = 8951/10000 (89.51%)

Epoch = 123, Loss = 0.4610, Batch = 390, Accuracy = 84.83: 100%|██████████| 391/391 [00:15<00:00, 24.73it/s]
Test set: Average loss = 0.0025, Accuracy = 8917/10000 (89.17%)

Epoch = 124, Loss = 0.4173, Batch = 390, Accuracy = 84.71: 100%|██████████| 391/391 [00:16<00:00, 24.41it/s]
Test set: Average loss = 0.0025, Accuracy = 8924/10000 (89.24%)
```
**Results**
- Minimum training loss = 0.210323364%
- Minimum testing loss = 0.002478111%
- Best training accuracy = 85.01%
- Best testing accuracy = 89.51%

**Loss Plot**

<img width="898" alt="loss_graph" src="https://user-images.githubusercontent.com/32274516/217305211-a21bd6fb-5aff-480f-8b0d-3f096251caa0.png">
