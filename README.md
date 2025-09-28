## Local Block Instance Normalization

This repository contains the tensorflow 2.0 code of Local Block Instance Normalization (LBIN), a novel robust normalization methods that is used to train style-invariant CNN models that are robust under 
domain shifts and different noises including gaussian noise, different kinds of blur, etc. Our normalization method uses the mean and variance statistics from local patches of each individual image, making it independent of other images in the batch, and thus more robust to domain shifts. We also present a comparison of the accuracy degradation of the model trained with our normalization under different image corruptions used in CIFAR-10C dataset

To run the code, execute 
```
python Experiments.py
```

A comparison of the robustness of our model under different corruptions with the other normalization scheme is done in the table below:

<img src="https://imgur.com/pdpXoRm.png" width="850" height="400" alt="Descriptive Alt Text">

Here BIN refers to the Batch Instance Normalization proposed in https://arxiv.org/abs/1805.07925 as a style invariant normalization technique. However, the local nature of our normalization technique allows it to outperform BIN on several image corruptions as observed from the table above.

The methods for applying different Image corruptions included in the CIFAR-10-C are available in the ImageCorruptions.py file.


