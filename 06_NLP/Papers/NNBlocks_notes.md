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
- large competitive industries with higher budgets introduced a similar implementation such as Google introducing Tensorflow and Meta introducing Pytorch
- initially built for academic research; once purpose was served and large industries took over to solve core problems, it contributed less
- slow and tedious debugging process as uses separate static graphs

## Solution:
- abstraction is used by NN blocks to use theano as an underlying implementation
- supports model-into-model structure where variable-sized complex trees can be used and allows other trees/models to be inserted into it
- allows models to be treated as objects and interchanged so that they can be easily joined together

## Today's upgrades/changes made to research:
- use of dynamic compilation graphs over static graphs to make tree networks less cumbersome
- use of Pytorch and Tensorflow instead of relying on Theano (loops and recursions can be passed directly into the forward propagation without having to separately define or process them)
- use of Just-In-Time (JIT) compilers to carry out algebraic operations instead of relying on manual Theano calculations and code compilations before execution
