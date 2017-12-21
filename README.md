# fashion-classifier
Some of my experiments while reading-implementing-testing different papers on Fashion-MNIST with PyTorch

Includes:

model.py - CoolNameNet is a CNN with optional shortcuts, dense-like connectivity and other stuff. Controlled by 3 intuitive parameters: width, depth and density. Recommended values for width and depth around 5-10, density 1-5. 

batchrenorm.py - implementations of BatchReNorm 1D and 2D(this one painfully slow)

trainer.py - Keras-like trainer with batch size growth, learning rate annealing and weight decay normalization and other tweaks.  

amsgradw.py - combination of AMSGrad(https://openreview.net/pdf?id=ryQu7f-RZ) with fixed weight decay(https://arxiv.org/abs/1711.05101)

Achieves 94.7 % test accuracy with 295.968 params trained for 200 epochs (settings: width=3, depth=5, density=2, default lr and weight_decay).
