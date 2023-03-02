The code is a modification of the code for the work: physics informed neural networks by Raissi, Maziar at https://github.com/maziarraissi/PINNs under MIT License. The License is copied here:

--------------------------------------------------------------------------
MIT License

Copyright (c) 2018 maziarraissi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
----------------------------------------------------------------------------------------
Cells beginned with: def init_model, def get_grad, model = init_model(), from time import time,
in the jupyter notebooks are modifications of some portions of the code at 
https://github.com/maziarraissi/PINNs
and therefore follow the above copyright notice and permmision notice.

# Transfer-function-prediction-using-machine-learning
Deep learning (feedforward neural network) code is in the form of jupyter notebook using Tensorflow. 
The main script for RBF interpolation is RBF_train_predict_modelname.m. 
The code of generating the discrete reduced transfer function  
is in the folder “compute_Vand_Reduced_transferfunction”. 
The folder “training data generation” contains the code for computing the 
training data and testing data for machine learning that can be directly used 
as input data of the RBF and feedforward neural networks. 
