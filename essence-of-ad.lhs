%% -*- latex -*-

\documentclass{article}

\usepackage[margin=0.9in]{geometry}

\input{macros}

\usepackage{scalerel}

%include polycode.fmt
%include forall.fmt
%include greek.fmt
%include formatting.fmt

\title{The simple essence of automatic differentiation}
\author{Conal Elliott}
\date{Draft of \today}
%% \institute[]{Target}

%% \setlength{\itemsep}{2ex}
%% \setlength{\parskip}{1ex}
%% \setlength{\blanklineskip}{1.5ex}
%% \setlength\mathindent{4ex}

%% \nc\wow\emph

\nc{\der}{\mathop{\mathcal{D}}}

\begin{document}

\maketitle

\begin{abstract}

Automatic differentiation (AD) is often presented in two forms: forward mode and reverse mode.
Forward mode is quite simple to implement and package via operator overloading but is inefficient for many problems of practical interest such as deep learning and other uses of gradient-based optimization.
Reverse mode (including its specialization, back-propagation) is much more efficient for these problems, but is also typically given much more complicated explanations and implementations, involving mutation, graph construction, and ``tapes''.
This talk develops a very simple specification and Haskell implementation for mode-independent AD based on the vocabulary of categories (generalized functions).
Although the categorical vocabulary would be difficult to write in directly, one can instead write regular Haskell programs to be converted to this vocabulary automatically (via a compiler plugin) and then interpreted as differentiable functions.
The result is direct, exact, and efficient differentiation with no notational overhead.
The specification and implementation are then generalized considerably by parameterizing over an underlying category.
This generalization is then easily specialized to forward and reverse modes, with the latter resulting from a simple dual construction for categories.
Another instance of generalized AD is automatic incremental evaluation of functional programs, again with no notational impact to the programmer.

\end{abstract}

\section{What's a derivative?}

%format eps = epsilon

%format Rm = R"^m"
%format Rn = R"^n"

\nc\set[1]{\{#1\}}

Since automatic differentiation (AD) has to do with computing derivatives, let's begin by considering what derivatives are.
If your introductory calculus class was like mine, you learned that the derivative |f' x| of a function |f :: R -> R| at a point |x| (in the domain of |f|) is a \emph{number}, defined as follows:
\begin{align} \label{eq:scalar-deriv}
|f' x = lim(eps -> 0)(frac(f (x+eps) - f x) eps)|
\end{align}
That is, |f' x| tells us how fast |f| is scaling input changes at |x|.

How well does this definition hold up beyond functions of type |R -> R|?
It will do fine with complex numbers (|C -> C|), where division is also defined.
Extending to |R -> Rn| also works if we interpret the ratio as dividing a vector (in |Rn|) by a scalar in the usual way.
When we extend to |Rm -> Rn| (or even |Rm -> R|), however, this definition no longer makes sense, as it would rely on dividing \emph{by} a vector |eps :: Rm|.

vThis difficulty of differentiation with non-scalar domains is usually addressed with the notion of ``partial derivatives'' with respect to the |m| scalar components of the domain |Rm|, often written ``$\partial f / \partial x_j$'' for $j \in \set{1,\ldots,m}$.
When the codomain |Rn| is also non-scalar (i.e., |n > 1|), we have a \emph{matrix} $\mathbf J$ (the \emph{Jacobian}), with $\mathbf J_{ij} = \partial f_i / \partial x_j$ for $i \in \set{1,\ldots,n}$, where each $f_i$ projects out the $i^{\text{th}}$ scalar value from the result of $f$.

So far, we've seen that the derivative of a function could be a single number (for |R -> R|), or a vector (for |R -> Rn|), or a matrix (for |Rm -> Rn|).
Moreover, each of these situations has an accompanying chain rule, which says how to differentiate the composition of two functions.
Where the scalar chain rule involves multiplying two scalar derivatives, the vector chain rule involves ``multiplying'' two \emph{matrices} $A$ and $B$ (the Jacobians), defined as follows:
$$ (\mathbf{A} \cdot \mathbf{B})_{ij} = \sum_{k=1}^m A_{ik} \cdot B_{kj} $$
Since once can think of scalars as a special case of vectors, and scalar multiplication as a special case of matrix multiplication, perhaps we've reached the needed generality.
When we turn our attention to higher derivatives (which are derivatives of derivatives), however, the situation get more complicated, and we need yet higher-dimensional representations, with correspondingly more complex chain rules.

Fortunately, there is a single, elegant generalization of differentiation with a correspondingly simple chain rule. 
First, change Definition \ref{eq:scalar-deriv} above to say that |f' x| is the unique |v :: Rn| such that
$$ |lim(eps -> 0)(frac(f (x+eps) - f x) eps) - v = 0| $$
or (equivalently)
$$ |lim(eps -> 0)(frac(f (x+eps) - (f x + eps *^ v)) eps) = 0|. $$
Notice that |v| is used to linearly transform |eps|.
Next, generalize this condition to say that the derivative of |f| at |x| is the unique \emph{linear map} |T| such that
$$|lim(eps -> 0)(frac(norm (f (x+eps) - (f x + T eps)))(norm eps)) = 0| .$$
This definition comes from \citet[chapter 2]{calculus-on-manifolds}, along with a proof that |T| is indeed unique when it exists.

The derivative of a function |f :: u -> v| at some |x :: u| is thus not a number, vector, matrix, or higher-dimensional variant, but rather a \emph{linear map} (also called ``linear transformations'') from |u| to |v|, which we will write as ``|u :-* v|''.
Here, |u| and |v| must be vector spaces that share a common underlying field.
Written as a Haskell-style type signature,

%% %format der = "\mathcal{D}"
%format der = "\der"

> der :: NOP (u -> v) -> (u -> (u :-* v))

The chain rule now has a lovely form, namely that the derivative of a composition is the \emph{composition} of derivatives:

> der (g . f) x = der g (f x) . der f x

If |f :: u -> v|, |g :: v -> w|, and |x :: u|, then |der f x :: u :-* v|, and |der g (f x) :: v :-* w|, so both sides of this equation have type |u :-* w|.
The numbers, vectors, matrices, etc mentioned above are all different \emph{representations} of linear maps; and the various forms of ``multiplication'' appearing in their associated chain rules are all implementations of linear map composition for those representations.

%format der2 = der "^2"

It follows that differentiating twice as the following type\footnote{As with |(->)|, we take |(:-*)| to associate rightward, so |a :-* b :-* c| means |a :-* (b :-* c)|}:

> der2 = der . der :: NOP (u -> v) -> (u -> (u :-* u :-* v))

The type |u :-* u :-* v| is a linear map that yields a linear map, which is the curried form of a \emph{bilinear} map.
Likewise, differentiating $k$ times yields a $k$-linear map curried $k-1$ times.
In particular, the \emph{Hessian} matrix $H$ corresponds to the second derivative of a function |f :: Rm -> R|, having $m$ rows and $m$ columns and satisfying the symmetry condition $H_{i,j} = H_{j,i}$.
Considering the shape of the matrix |H|, it would be easy to mistakenly treat it as representing the first derivative of some other function |g :: Rm -> Rm|.
Doing so would be unsafe, however, since second derivatives are (curried) bilinear maps, not linear maps.
By providing an explicit abstract type for linear maps rather than using a bare matrix representation, such unsafe uses become type errors, easily caught at compile-time.

\bibliography{bib}

\end{document}


