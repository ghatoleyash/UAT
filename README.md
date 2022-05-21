# Universal Approximation Theorem

## Theory
* There was a need to determine whether a neural network had the potential to approximate any type of functions.
* Initial results from K. Hornik '89 focuses on sigmoids
* M. Leshno et al '93 generalizes to other functions (e.g., ReLU)
* Original results showed that shallow networks with unconstrained width can approximate any continuous functions
* More recent results provide some trade-off between widht and depth

* Consider a Fully Connected Network (FCN) with input dimensions min and output dimensions mout, and widhts for each layer min = m0, m1, m2, ...mk = mout

### Theorem
<img src="https://github.com/ghatoleyash/UAT/blob/main/images/UAT_1.jpg" width=80% height=80%>

#### UAT 1 (Shallow/Wide Network)
**Defintion**

<img src="https://github.com/ghatoleyash/UAT/blob/main/images/UA1%20DEFINITION_1.jpg" width=80% height=80%>

**Proof**

<img src="https://github.com/ghatoleyash/UAT/blob/main/images/UA1%20PROOF_1.jpg" width=80% height=80%>


#### UAT 2 (Narrow/Deep Network)
**Defintion**

<img src="https://github.com/ghatoleyash/UAT/blob/main/images/UA2%20DEFINITION_1.jpg" width=80% height=80%>

**Proof**

<img src="https://github.com/ghatoleyash/UAT/blob/main/images/UA2%20PROPOSITION%201_1.jpg" width=80% height=80%>
<img src="https://github.com/ghatoleyash/UAT/blob/main/images/UA2%20PROOF_1.jpg" width=80% height=80%>


## Results
Approximating the sinusoidal function

* Shallow/Wide Network
<img src="https://github.com/ghatoleyash/UAT/blob/main/images/WideNet.png" width=80% height=50%>


* Narrow/Deep Network
<img src="https://github.com/ghatoleyash/UAT/blob/main/images/DeepNet.png" width=80% height=50%>

*Note: Used "tanh" as activation function since its range is (-1,1) as the approximating function is sinusoidal*
