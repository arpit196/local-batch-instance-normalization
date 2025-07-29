## Local Block Instance Normalization

This repository contains the tensorflow 2.0 code of Local Block Instance Normalization (LBIN), a novel robust normalization methods that is used to train style invariant CNN models that are robust under 
domain shifts. We also present a comparison of the accuracy degradation of the model trained with our normalization under different image corruptions used in CIFAR-10C dataset

To run the code, execute 
```
python run-mnist.py
```

A comparison of the robustness of our model under different corruptions:
![alt text](https://imgur.com/pdpXoRm)
