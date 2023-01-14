# Part 1
For below network, calculated the partial derivatives of loss with respect to the weights and plotted the loss for various learning rates.

## Network											
![image](https://user-images.githubusercontent.com/106446673/212459658-a0141e05-6b46-42e8-98e7-9f5b214ce28d.png)

In the above network
1. the notations used are:
```
w(x) = Weights of i (where x = 1, 2, ... , 8)			out_h(x) = Activated output of i (where x = 1, 2)
h(x) = Hidden output of i (where x = 1, 2)			out_o(x) = Activated output of i (where x = 1, 2)
o(x) = Output of i (where x = 1, 2)
```

2. the inputs are:
```
i1 = Input 1			t1 = output1 (Expected output for Input1)			η = Learning Rate
i2 = Input 2			t2 = output2 (Expected output for Input2)
```

3. and the loss function L2 is defined as: 
```
E1 = (1/2) * (t1 - out_o1)^2
E2 = (1/2) * (t2 - out_o1)^2
```

Below is the snapshot of the backpropagation calculation of the above network at ```η=1``` for 50 epochs.	 _[(Workbook)](https://github.com/Sudarshanpune/mnist_nn/blob/main/S3/Backpropagation.xlsx)_

_Note: Please refer the sheet2 for the backpropagation calculations_
![image](https://user-images.githubusercontent.com/106446673/212454317-0bdc9c29-c810-4570-b86e-db53b3c0fcb7.png)

## Plots
![image](https://user-images.githubusercontent.com/106446673/212450474-41872747-659e-44a2-9062-c7f67985c983.png) ![image](https://user-images.githubusercontent.com/106446673/212450807-3eb00693-bf8b-4fd0-9bb4-796754f00695.png)

![image](https://user-images.githubusercontent.com/106446673/212450479-11aa1a98-2d63-4a49-ad7b-4a88c3cbde10.png) ![image](https://user-images.githubusercontent.com/106446673/212450442-8927fefc-9b2f-4329-9d9b-67f63bf25989.png)


# Part 2

## Task: 
Train the [network](https://colab.research.google.com/drive/1AtqGpG8BVhnv7LS08vnG_sdzz3_Rxuzn?usp=sharing) which identifies MNIST images, and achieve the follwing:
- 99.4% validation accuracy
- use less than 20k paramerters and less than 20 epochs
- usage of batch normalization, droupout, fully connected (FC) layer, global avergae pooling (GAP)

***Solution** - [Notebook](https://github.com/Sudarshanpune/mnist_nn/blob/main/S3/EVA8_S3.ipynb) | [Colab link](https://colab.research.google.com/drive/1rhaJuAlLPwSbexikLxytMGGFVvFDGF8H?usp=sharing)*

## Architecture:

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 26, 26]              80
       BatchNorm2d-2            [-1, 8, 26, 26]              16
            Conv2d-3           [-1, 16, 24, 24]           1,168
       BatchNorm2d-4           [-1, 16, 24, 24]              32
            Conv2d-5           [-1, 16, 22, 22]           2,320
       BatchNorm2d-6           [-1, 16, 22, 22]              32
         MaxPool2d-7           [-1, 16, 11, 11]               0
           Dropout-8           [-1, 16, 11, 11]               0
            Conv2d-9             [-1, 16, 9, 9]           2,320
      BatchNorm2d-10             [-1, 16, 9, 9]              32
           Conv2d-11             [-1, 16, 7, 7]           2,320
      BatchNorm2d-12             [-1, 16, 7, 7]              32
           Conv2d-13             [-1, 16, 5, 5]           2,320
      BatchNorm2d-14             [-1, 16, 5, 5]              32
          Dropout-15             [-1, 16, 5, 5]               0
           Conv2d-16             [-1, 32, 3, 3]           4,640
      BatchNorm2d-17             [-1, 32, 3, 3]              64
           Conv2d-18             [-1, 10, 3, 3]             330
        AvgPool2d-19             [-1, 10, 1, 1]               0
================================================================
Total params: 15,738
Trainable params: 15,738
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.42
Params size (MB): 0.06
Estimated Total Size (MB): 0.48
----------------------------------------------------------------

```

## Logs:

```
  
loss=0.054516177624464035 batch_id=468: 100%|██████████| 469/469 [00:20<00:00, 23.15it/s]


Test set: Epoch:1 Average loss: 0.0607, Accuracy: 9821/10000 (98%)

loss=0.06605575233697891 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 28.22it/s]


Test set: Epoch:2 Average loss: 0.0349, Accuracy: 9892/10000 (99%)

loss=0.028613736853003502 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 28.32it/s]


Test set: Epoch:3 Average loss: 0.0343, Accuracy: 9898/10000 (99%)

loss=0.035630542784929276 batch_id=468: 100%|██████████| 469/469 [00:17<00:00, 27.37it/s]


Test set: Epoch:4 Average loss: 0.0352, Accuracy: 9900/10000 (99%)

loss=0.0051137725822627544 batch_id=468: 100%|██████████| 469/469 [00:17<00:00, 26.66it/s]


Test set: Epoch:5 Average loss: 0.0434, Accuracy: 9861/10000 (99%)

loss=0.011735579930245876 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 28.03it/s]


Test set: Epoch:6 Average loss: 0.0280, Accuracy: 9913/10000 (99%)

loss=0.04935211315751076 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 28.01it/s]


Test set: Epoch:7 Average loss: 0.0276, Accuracy: 9915/10000 (99%)

loss=0.009882115758955479 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 28.05it/s]


Test set: Epoch:8 Average loss: 0.0274, Accuracy: 9911/10000 (99%)

loss=0.001192469266243279 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 28.09it/s]


Test set: Epoch:9 Average loss: 0.0240, Accuracy: 9934/10000 (99%)

loss=0.007792516145855188 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 27.77it/s]


Test set: Epoch:10 Average loss: 0.0243, Accuracy: 9933/10000 (99%)

loss=0.009758209809660912 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 27.98it/s]


Test set: Epoch:11 Average loss: 0.0209, Accuracy: 9934/10000 (99%)

loss=0.02132534235715866 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 28.16it/s]


Test set: Epoch:12 Average loss: 0.0199, Accuracy: 9942/10000 (99%)

loss=0.029384152963757515 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 28.12it/s]


Test set: Epoch:13 Average loss: 0.0260, Accuracy: 9920/10000 (99%)

loss=0.002309575444087386 batch_id=468: 100%|██████████| 469/469 [00:17<00:00, 26.98it/s]


Test set: Epoch:14 Average loss: 0.0220, Accuracy: 9933/10000 (99%)

loss=0.0015483993338420987 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 27.90it/s]


Test set: Epoch:15 Average loss: 0.0225, Accuracy: 9934/10000 (99%)

loss=0.00731826014816761 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 27.64it/s]


Test set: Epoch:16 Average loss: 0.0198, Accuracy: 9936/10000 (99%)

loss=0.005924191791564226 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 28.03it/s]


Test set: Epoch:17 Average loss: 0.0213, Accuracy: 9928/10000 (99%)

loss=0.0163444671779871 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 28.03it/s]


Test set: Epoch:18 Average loss: 0.0237, Accuracy: 9932/10000 (99%)

loss=0.005127995740622282 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 27.88it/s]


Test set: Epoch:19 Average loss: 0.0256, Accuracy: 9918/10000 (99%)

```

**Summary:** Achieved 99.42% accuracy at epoch 12 by using 15,738 parameters.