# NN Blocks: A deep learning framework for computational linguistics NN

## Challenge: 
- complex linguistic features that do not fit within traditional Neural Networks
- overexposure to Theano mechanisms
- can not easily handle variable-sized data or implement recursive NN

## What is Theano Mechanism?
- python library for deep learning developed by MILA
- discontinued after 2017 as Google introduced Tensorflow, carrying out similar implementation
- core function is using symbolic differentiation (producing a formula for the derivative instead of approximating the VALUE for the derivative)
- creates a large mathematical graph including all symbols, variables and parameters, and treats the NN as a singular mathematical formula
- as it has all symbols of the loss function, it uses them to create an expression for the derivative using the chain rule
- it only needs to form the forward expression, backward propagation is automatically done.

- WHY DISCONTINUED?
- 
