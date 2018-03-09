## The simple essence of automatic differentiation

This paper is based on an [invited talk for PEPM 2018](https://github.com/conal/talk-2018-essence-of-ad/blob/master/readme.md) (with slides and video) by the same name.

### Abstract

Automatic differentiation (AD) is often presented in two forms: forward mode and reverse mode.
Forward mode is quite simple to implement and package via operator overloading, but is inefficient for many problems of practical interest such as deep learning and other uses of gradient-based optimization.
Reverse mode (including its specialization, backpropagation) is much more efficient for these problems, but is also typically given much more complicated explanations and implementations.
This paper develops a very simple specification and implementation for mode-independent AD based on the vocabulary of categories (``generalized functions'').
Although the categorical vocabulary would be awkward to write in directly, one can instead write regular Haskell programs to be converted to this vocabulary automatically (via a compiler plugin) and then interpreted as differentiable functions.
The result is direct, exact, and efficient differentiation with no notational overhead.
The specification and implementation are generalized considerably by parameterizing over an underlying category.
This generalization is then easily specialized to two variations of reverse-mode AD.
These reverse-mode implementations are much simpler than previously known and are composed from two generally useful category transformers: continuation-passing and dualization.
All of the implementations are calculated from simple, homomorphic specifications and so are correct by construction.
The dualized variant is suitable for gradient-based optimization and is particularly compelling in simplicity and efficiency, requiring no matrix-level representations or computation.
