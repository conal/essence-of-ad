%% -*- latex -*-

\documentclass{article}

\usepackage[margin=0.9in]{geometry}

\input{macros}

\usepackage{scalerel}

%include polycode.fmt
%include forall.fmt
%include greek.fmt
%include formatting.fmt

%% \pagestyle{headings}
%% \pagestyle{myheadings}

%% \markboth{...}{...}

\title{The simple essence of automatic differentiation}
\author{Conal Elliott\\[1ex]Target}
\date{Draft of \today}
%% \institute[]{Target}

%% \setlength{\itemsep}{2ex}
%% \setlength{\parskip}{1ex}
%% \setlength{\blanklineskip}{1.5ex}
%% \setlength\mathindent{4ex}

%% \nc\wow\emph

\nc{\der}{\mathop{\mathcal{D}}}

\nc\ruleLabel[1]{\label{rule:#1}}
\nc\ruleRef[1]{Rule (\ref{rule:#1})}

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
In other words, |T| is a \emph{local linear approximation} of |f| at |x|.
This definition comes from \citet[chapter 2]{Spivak65}, along with a proof that |T| is indeed unique when it exists.

The derivative of a function |f :: a -> b| at some value in |a| is thus not a number, vector, matrix, or higher-dimensional variant, but rather a \emph{linear map} (also called ``linear transformations'') from |a| to |b|, which we will write as ``|a :-* b|''.
The numbers, vectors, matrices, etc mentioned above are all different \emph{representations} of linear maps; and the various forms of ``multiplication'' appearing in their associated chain rules are all implementations of linear map composition for those representations.
Here, |a| and |b| must be vector spaces that share a common underlying field.
Written as a Haskell-style type signature,

%% %format der = "\mathcal{D}"
%format der = "{}\der{} "

> der :: (a -> b) -> (a -> (a :-* b))

With the shift to linear maps, there is one general chain rule, having a lovely form, namely that the derivative of a composition is a \emph{composition} of the derivatives \cite[Theorem 2-2]{Spivak65}:
\begin{align} \ruleLabel{chain}
|der (g . f) a = der g (f a) . der f a|
\end{align}
If |f :: a -> b| and |g :: b -> c|\out{, and |a :: a|}, then |der f a :: a :-* b|, and |der g (f a) :: b :-* c|, so both sides of this equation have type |a :-* c|.\footnote{I adopt the common, if sometimes confusing, Haskell convention of sharing names between type and value variables, e.g., with |a| (a value variable) having type |a| (a type variable).
Haskell value and type variable names live in different name spaces and are distinguished by syntactic context.}

%format der2 = der "^2"

From the type of |der|, it follows that differentiating twice has the following type\footnote{As with ``|->|'', we will take ``|:-*|'' to associate rightward, so |u :-* v :-* w| is equivalent to |u :-* (v :-* w)|}:

> der2 = der . der :: NOP (a -> b) -> (a -> (a :-* a :-* b))

The type |a :-* a :-* b| is a linear map that yields a linear map, which is the curried form of a \emph{bilinear} map.
Likewise, differentiating $k$ times yields a $k$-linear map curried $k-1$ times.
In particular, the \emph{Hessian} matrix $H$ corresponds to the second derivative of a function |f :: Rm -> R|, having $m$ rows and $m$ columns and satisfying the symmetry condition $H_{i,j} = H_{j,i}$.

\emph{A comment on type safety:}
Considering the shape of the matrix |H|, it would be easy to mistakenly treat it as representing the first derivative of some other function |g :: Rm -> Rm|.
Doing so would be unsafe, however, since second derivatives are (curried) bilinear maps, not linear maps.
By providing an explicit abstract type for linear maps rather than using a bare matrix representation, such unsafe uses become type errors, easily caught at compile-time.
\mynote{Hm. I guess one could say that |H| really does represent a first derivative, namely of |f'| itself considered as a vector.
However, |f'| is a covector, not a vector.
Noodle more on this explanation.}

\section{Compositionality}

Strictly speaking, the chain rule in \ruleRef{chain} is not compositional, i.e., it is \emph{not} the case |der (g . f)| can be constructed solely from |der g| and |der f|.
Instead, it also needs |f| itself.
Compositionality is very helpful for the implementation style used in this paper, and fortunately, there is a simple way to restore compositionality.
Instead of constructing just the derivative of a function |f|, suppose we \emph{augment} |f| with its derivative:

%format ad = der"\!^+\!"
%format ad0 = der"\!_{\scriptscriptstyle 0}\!\!^+\!"

\begin{code}
ad0 :: (a -> b) -> ((a -> b) :* (a -> (a :-* b)))   -- first guess
ad0 f = (f, der f)
\end{code}
As desired, this altered specification is compositional:
\begin{code}
   ad0 (g . f)
==  {- definition of |ad0| -}
   (g . f, der (g . f))
==  {- chain rule -}
   (g . f, \ a -> der g (f a) . der f a)
\end{code}

Note that |ad0 (g . f)| is assembled entirely out of the parts of |ad0 g| and |ad0 f|, which is to say from |g|, |der g|, |f|, and |der f|.
Writing out |g . f| as |\ a -> g (f a)| underscores that the two parts of |ad0 (g . f)| when applied to |a| both involve |f a|.
Computing these parts independently thus requires redundant work.
Moreover, the chain rule itself requires applying a function and its derivative (namely |f| and |der f|) to the same |a|.
Since the chain gets applied recursively to nested compositions, the redundant work multiplies greatly, resulting in an impractically expensive algorithm.

This efficiency problem is also easily fixed.
Instead of pairing |f| and |der f|, let's instead \emph{combine} them into a single function\footnote{The precedence of ``|:*|'' is tighter than that of ``|->|'' and ``|:-*|'', so |a -> b :* (a :-* b)| is equivalent to |a -> (b :* (a :-* b))|}:
\begin{code}
ad :: (a -> b) -> (a -> b :* (a :-* b))   -- better!
ad f a = (f a, der f a)
\end{code}
Combining |f| and |der f| into a single function in this way allows us to eliminate the redundant composition of |f a| in |ad (g . f) a|:
\begin{code}
   ad (g . f) a
==  {- definition of |ad| -}
   ((g . f) a, der (g . f) a)
==  {- definition of |(.)|; chain rule -}
   (g (f a), der g (f a) . der f a)
==  {- refactoring to share |f a| -}
   let b = f a in (g b, der g b . der f a)
==  {- refactoring to show compositionality -}
   let { (b,f') = ad f a ; (d,g') = ad g b } in (d, g' . f')
\end{code}

\mynote{Clarify that this |ad| definition is a specification, not an implementation.}

\section{Other forms of composition}

The chain rule, telling how to differentiate sequential compositions, gets a lot of attention in calculus classes and in automatic and symbolic differentiation.\notefoot{To do: introduce AD and SD early.}
There are other important ways to combine functions, however, and examining them yields more helpful tools.

One other tool combines two functions sharing a domain type into a single function that pairs the result:
\begin{code}
(&&&) :: (a -> c) -> (a -> d) -> (a -> c :* d)
f &&& g = \ a -> (f a, g a)
\end{code}
We will sometimes refer to the |(&&&)| operation as ``fork'' \citep{Gibbons2002:Calculating}.
While the derivative of the (sequential) composition is a composition of derivatives, the derivative of a fork is the fork of the derivatives:\notefoot{Is there a name for this rule? I've never seen it mentioned.}
\begin{align} \ruleLabel{fork}
|der (f &&& g) a = der f a &&& der g a|
\end{align}
If |f :: a -> c| and |g :: a -> d|, then |der f a :: a :-* c| and |der g a :: a :-* d|, so |der f a &&& der g a :: a :-* c :* d|, as needed.

\ruleRef{fork} gives us what we need to construct |ad (f &&& g)| compositionally:
\begin{code}
   ad (f &&& g) a
==  {- definition of |ad| -}
   ((f &&& g) a, der (f &&& g) a)
==  {- definition of |(&&&)| -}
   ((f a, g a), der (f &&& g) a)
==  {- \ruleRef{fork} -}
   ((f a, g a), der f a &&& der g a)
==  {- refactoring -}
   let { (c,f') = (f a, der f a) ; (d,g') = (g a, der g a) } in ((c,d), (f' &&& g'))
==  {- definition of |ad| -}
   let { (c,f') = ad f a ; (d,g') = ad g a } in ((c,d), (f' &&& g'))
\end{code}
%%    ((f &&& g) &&& der (f &&& g)) a
%% ==  {- definition of |(&&&)| -}

The |(&&&)| operation can be used to give a terser specification for |ad|:

> ad f = f &&& der f

There is another, dual, form of composition as well, pronounced ``join'' and defined as follows \citep{Gibbons2002:Calculating}:
\begin{code}
(|||) :: (a -> c) -> (b -> c) -> (a :* b -> c)
f ||| g = \ a -> f a + g a
\end{code}
Where |(&&&)| combines two functions with the same domain and pairs their results, |(###)| combines two functions with the same codomain and \emph{adds} their results.\footnote{You may have expected a different type and definition, using \emph{sums} instead of products:
\begin{code}
(|||) :: (a -> c) -> (b -> c) -> (a :+ b -> c)
(f ||| g) (Left   a) = f a
(f ||| g) (Right  b) = g b
\end{code}
More generally, |(&&&)| and |(###)| work with categorical products and coproducts.
The categories involved in this paper (functions on additive types, linear maps, and differentiable functions) are all \emph{biproduct} categories, where categorical products and coproducts coincide \needcite{}.
}

Happily, there is differentiation rule for |(###)| as well, having the same poetry as the rules for |(.)| and |(&&&)|, namely that the derivative of a join is a join of the derivatives:
\begin{align} \ruleLabel{join}
|der (f ### g) (a,b) = der f a ### der g b|
\end{align}
If |f :: a -> c| and |g :: b -> c|, then |der f a :: a :-* c| and |der g b :: b :-* c|, so |der f a ### der g b :: a :* b :-* c|, as needed.

\ruleRef{join} is exactly what we need to construct |ad (f ### g)| compositionally:
\begin{code}
   ad (f ||| g) (a,b)
==  {- definition of |ad| -}
   ((f ||| g) (a,b), der (f ||| g) (a,b))
==  {- definition of |(###)| -}
   ((f a + g b), der (f ||| g) (a,b))
==  {- \ruleRef{join} -}
   ((f a + g b), der f a ||| der g b)
==  {- refactoring -}
   let { (c,f') = (f a, der f a) ; (d,g') = (g b, der g b) } in ((c + d), (f' ||| g'))
==  {- definition of |ad| -}
   let { (c,f') = ad f a ; (d,g') = ad g b } in (c + d, (f' ||| g'))
\end{code}

An important point left implicit in the discussion above is that our three combining forms |(.)|, |(&&&)|, and |(###)| all preserve linearity.
This property is what makes it meaningful to use these forms to combine derivatives, i.e., linear maps, as we've done above.

\section{Linear functions}

A function |f :: a -> b| is said to be \emph{linear} when |f| distributes over (preserves the structure of) vector addition and scalar multiplication, i.e.,
\begin{code}
f (a + a')  == f a + f a'
f (s *^ a)  == s *^ f a
\end{code}
for all |a,a' :: a| and |s| taken from the scalar field underlying |a| and |b|.

In addition to the derivative rules for |(.)|, |(&&&)|, and |(###)|, there is one more broadly useful tool to be added to our collection, which we'll call the ``linearity rule'': \emph{the derivative of every linear function is itself, everywhere}, i.e., for all linear functions |f|,
\begin{align} \ruleLabel{linear}
|der f a = f|
\end{align}
This statement may sound surprising, but less so when we recall that the |der f a| is a local linear approximation of |f| at |a|, so we're simply saying that linear functions are their own perfect linear approximations.

For example, consider the (linear) function |id = \ a -> a|.
The linearity rule says that |der id a = id|.
When expressed in terms of typical \emph{representations} of linear maps, this property may be expressed as saying that |der id a| is the number one or as an identity matrix (with ones on the diagonal and zeros elsewhere).

%format Rmn = R"^{m+n}"

As another example, consider the function |fst (a,b) = a|, for which the linearity rule says |der fst (a,b) = fst|.
This property, when expressed in terms of typical \emph{representations} of linear maps, would appear as saying that |der fst a| comprises the partial derivatives one and zero if |a, b :: R|.
More generally, if |a :: Rm| and |b :: Rn|, then the Jacobian matrix representation has shape |m :* (m+n)| (ie |m| rows and |m + n| columns) and is formed by the horizontal abutment of an |m :* m| identity matrix on the left with an |m :* n| zero matrix on the right.
This |m :* (m+n)| matrix, however, represents |fst :: Rmn :-* Rm|.
Note how much simpler it is to say |der fst (a,b) = fst|, and with no loss of precision!

Given \ruleRef{linear}, we can construct |ad f| for all linear |f|:
\begin{code}
   ad f
==  {- definition of |ad| -}
   f &&& der f
==  {- definition of |(&&&)| -}
   \ a -> (f a, der f a)
==  {- \ruleRef{linear} -}
   \ a -> (f a, f)
==  {- definition of |(&&&)| -}
   f &&& const f
\end{code}

%if False
%endif

\section{To do}

\begin{itemize}
\item AD for linear functions.
\item The rest of the talk.
\item More biproduct operations: |(***)|, |dup|, |jam|, |(+)| (arrow addition).
\item Indexed biproducts.
\end{itemize}


\bibliography{bib}

\end{document}


