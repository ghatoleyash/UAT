# Universal Approximation Theorem

## Theory
* There was a need to determine whether a neural network had the potential to approximate any type of functions.
* Initial results from K. Hornik '89 focuses on sigmoids
* M. Leshno et al '93 generalizes to other functions (e.g., ReLU)
* Original results showed that shallow networks with unconstrained width can approximate any continuous functions
* More recent results provide some trade-off between widht and depth

* Consider a Fully Connected Network (FCN) with input dimensions min and output dimensions mout, and widhts for each layer min = m0, m1, m2, ...mk = mout

we can represent the fully connected network as 
fnn(x) = Ak*g*Ak-1*...*A2*g*A1(x)
where Ak:Rmk-1 -> Rmk


* We say that Fnn is an epsilon-approximation of a function f:Rmin->Rmout over a compact set S if
sup|| f(x) - fnn(x)|| <=epsilon



In the mathematical theory of artificial neural networks, universal approximation theorems are results that establish the density of an algorithmically generated class of functions within a given function space of interest. Typically, these results concern the approximation capabilities of the feedforward architecture on the space of continuous functions between two Euclidean spaces, and the approximation is with respect to the compact convergence topology.

However, there are also a variety of results between non-Euclidean spaces and other commonly used architectures and, more generally, algorithmically generated sets of functions, such as the convolutional neural network (CNN) architecture,N radial basis-functions,[6] or neural networks with specific properties.[7] Most universal approximation theorems can be parsed into two classes. The first quantifies the approximation capabilities of neural networks with an arbitrary number of artificial neurons ("arbitrary width" case) and the second focuses on the case with an arbitrary number of hidden layers, each containing a limited number of artificial neurons ("arbitrary depth" case).

Universal approximation theorems imply that neural networks can represent a wide variety of interesting functions when given appropriate weights. On the other hand, they typically do not provide a construction for the weights, but merely state that such a construction is possible.


An activation layer is applied right after a linear layer in the Neural Network to provide non-linearities. Non-linearities help Neural Networks perform more complex tasks. An activation layer operates on activations (h₁, h2 in this case) and modifies them according to the activation function provided for that particular activation layer. Activation functions are generally non-linear except for the identity function. Some commonly used activation functions are ReLu, sigmoid, softmax, etc. With the introduction of non-linearities along with linear terms, it becomes possible for a neural network to model any given function approximately on having appropriate parameters(w₁, w₂, b₁, etc in this case). The parameters converge to appropriateness on training suitably. You can get better acquainted mathematically with the Universal Approximation theorem from.


## Results
Approximating the sinusoidal function

* Shallow/Wide Network
<img src="https://github.com/ghatoleyash/UAT/blob/main/images/WideNet.png" width=80% height=50%>


* Narrow/Deep Network
<img src="https://github.com/ghatoleyash/UAT/blob/main/images/DeepNet.png" width=80% height=50%>

*Note: Used "tanh" as activation function since its range is (-1,1) as the approximating function is sinusoidal*
