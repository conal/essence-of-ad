%% -*- latex -*-

%% While editing/previewing, use 12pt and tiny margin.
\documentclass[12pt]{article}  % fleqn,
\usepackage[margin=0.2in]{geometry}  % 0.9in

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

\nc\thmLabel[1]{\label{thm:#1}}
\nc\thmRef[1]{Theorem \ref{thm:#1}}
\nc\thmRefTwo[2]{Theorems \ref{thm:#1} and \ref{thm:#2}}

\nc\corLabel[1]{\label{cor:#1}}
\nc\corRef[1]{Corollary \ref{cor:#1}}
\nc\corRefTwo[2]{Corollaries \ref{cor:#1} and \ref{cor:#2}}

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

\newtheorem{theorem}{Theorem}%[section]
\newtheorem{corollary}{Corollary}[theorem]
% \newtheorem{lemma}[theorem]{Lemma}

\sectionl{What's a derivative?}

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

This difficulty of differentiation with non-scalar domains is usually addressed with the notion of ``partial derivatives'' with respect to the |m| scalar components of the domain |Rm|, often written ``$\partial f / \partial x_j$'' for $j \in \set{1,\ldots,m}$.
When the codomain |Rn| is also non-scalar (i.e., |n > 1|), we have a \emph{matrix} $\mathbf J$ (the \emph{Jacobian}), with $\mathbf J_{ij} = \partial f_i / \partial x_j$ for $i \in \set{1,\ldots,n}$, where each $f_i$ projects out the $i^{\text{th}}$ scalar value from the result of $f$.

So far, we've seen that the derivative of a function could be a single number (for |R -> R|), or a vector (for |R -> Rn|), or a matrix (for |Rm -> Rn|).
Moreover, each of these situations has an accompanying chain rule, which says how to differentiate the composition of two functions.
Where the scalar chain rule involves multiplying two scalar derivatives, the vector chain rule involves ``multiplying'' two \emph{matrices} $A$ and $B$ (the Jacobians), defined as follows:
$$ (\mathbf{A} \cdot \mathbf{B})_{ij} = \sum_{k=1}^m A_{ik} \cdot B_{kj} $$
Since once can think of scalars as a special case of vectors, and scalar multiplication as a special case of matrix multiplication, perhaps we've reached the needed generality.
When we turn our attention to higher derivatives (which are derivatives of derivatives), however, the situation get more complicated, and we need yet higher-dimensional representations, with correspondingly more complex chain rules.

Fortunately, there is a single, elegant generalization of differentiation with a correspondingly simple chain rule. 
First, change Definition \ref{eq:scalar-deriv} above to say that |f' x| is the unique |v :: Rn| such that\footnote{For clarity, throughout this paper we will use ``|A = B|'' to mean ``|A| is defined as |B|'' and ``|==|'' to mean (more broadly) that ``|A| is equal to |B|'', using the former to introduce |A|, and the latter to claim that an well-defined statement equality is in fact true.}
$$ |lim(eps -> 0)(frac(f (x+eps) - f x) eps) - v == 0| $$
or (equivalently)
$$ |lim(eps -> 0)(frac(f (x+eps) - (f x + eps *^ v)) eps) == 0|. $$
Notice that |v| is used to linearly transform |eps|.
Next, generalize this condition to say that the derivative of |f| at |x| is the unique \emph{linear map} |T| such that
$$|lim(eps -> 0)(frac(norm (f (x+eps) - (f x + T eps)))(norm eps)) == 0| .$$
In other words, |T| is a \emph{local linear approximation} of |f| at |x|.
This definition comes from \citet[chapter 2]{Spivak65}, along with a proof that |T| is indeed unique when it exists.

The derivative of a function |f :: a -> b| at some value in |a| is thus not a number, vector, matrix, or higher-dimensional variant, but rather a \emph{linear map} (also called ``linear transformations'') from |a| to |b|, which we will write as ``|a :-* b|''.
The numbers, vectors, matrices, etc mentioned above are all different \emph{representations} of linear maps; and the various forms of ``multiplication'' appearing in their associated chain rules are all implementations of linear map composition for those representations.
Here, |a| and |b| must be vector spaces that share a common underlying field.
Written as a Haskell-style type signature,

%% %format der = "\mathcal{D}"
%% %format der = "{}\der{} "
%format der = "\mathop{\mathcal{D}}"

\begin{code}
der :: (a -> b) -> (a -> (a :-* b))
\end{code}

%format der2 = der "^2"

From the type of |der|, it follows that differentiating twice has the following type\footnote{As with ``|->|'', we will take ``|:-*|'' to associate rightward, so |u :-* v :-* w| is equivalent to |u :-* (v :-* w)|}:

\begin{code}
der2 = der . der :: NOP (a -> b) -> (a -> (a :-* a :-* b))
\end{code}

The type |a :-* a :-* b| is a linear map that yields a linear map, which is the curried form of a \emph{bilinear} map.
Likewise, differentiating $k$ times yields a $k$-linear map curried $k-1$ times.
In particular, the \emph{Hessian} matrix $H$ corresponds to the second derivative of a function |f :: Rm -> R|, having $m$ rows and $m$ columns and satisfying the symmetry condition $H_{i,j} == H_{j,i}$.

\emph{A comment on type safety:}
Considering the shape of the matrix |H|, it would be easy to mistakenly treat it as representing the first derivative of some other function |g :: Rm -> Rm|.
Doing so would be unsafe, however, since second derivatives are (curried) bilinear maps, not linear maps.
By providing an explicit abstract type for linear maps rather than using a bare matrix representation, such unsafe uses become type errors, easily caught at compile-time.
\mynote{Hm. I guess one could say that |H| really does represent a first derivative, namely of |f'| itself considered as a vector.
However, |f'| is a covector, not a vector.
Noodle more on this explanation.}

\sectionl{Rules for differentiation}

\subsectionl{Sequential composition}

With the shift to linear maps, there is one general chain rule, having a lovely form, namely that the derivative of a composition is a \emph{composition} of the derivatives \cite[Theorem 2-2]{Spivak65}:
\begin{theorem}[compose/``chain'' rule] \thmLabel{compose}
$$|der (g . f) a == der g (f a) . der f a|$$
\end{theorem}
If |f :: a -> b| and |g :: b -> c|\out{, and |a :: a|}, then |der f a :: a :-* b|, and |der g (f a) :: b :-* c|, so both sides of this equation have type |a :-* c|.\footnote{I adopt the common, if sometimes confusing, Haskell convention of sharing names between type and value variables, e.g., with |a| (a value variable) having type |a| (a type variable).
Haskell value and type variable names live in different name spaces and are distinguished by syntactic context.}

Strictly speaking, Theorem \thmRef{compose} is not a compositional recipe for differentiating compositions, i.e., it is \emph{not} the case |der (g . f)| can be constructed solely from |der g| and |der f|.
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
\begin{corollary} \corLabel{compose}
|ad| is (efficiently) compositional with respect to |(.)|. Specifically,
\begin{code}
   ad (g . f) a
==  {- definition of |ad| -}
   ((g . f) a, der (g . f) a)
==  {- definition of |(.)|; chain rule -}
   (g (f a), der g (f a) . der f a)
==  {- refactoring to share |f a| -}
   let b = f a in (g b, der g b . der f a)
==  {- refactoring to show compositionality -}
   let { (b,f') = ad f a ; (c,g') = ad g b } in (c, g' . f')
\end{code}
\end{corollary}

\subsectionl{Products in the codomain}

The chain rule, telling how to differentiate sequential compositions, gets a lot of attention in calculus classes and in automatic and symbolic differentiation.\notefoot{To do: introduce AD and SD early.}
There are other important ways to combine functions, however, and examining them yields more helpful tools.

One other tool combines two functions sharing a domain type into a single function that pairs the result:
\begin{code}
(&&&) :: (a -> c) -> (a -> d) -> (a -> c :* d)
f &&& g = \ a -> (f a, g a)
\end{code}
We will sometimes refer to the |(&&&)| operation as ``fork'' \citep{Gibbons2002:Calculating}.
As an example, the |(&&&)| operation can be used to give a terser specification: |ad f = f &&& der f|.

While the derivative of the (sequential) composition is a composition of derivatives, the derivative of a fork is the fork of the derivatives \citep[Theorem 2-3 (3)]{Spivak65}:\notefoot{Is there a name for this rule? I've never seen it mentioned.}
\begin{theorem}[fork rule] \thmLabel{fork}
$$|der (f &&& g) a == der f a &&& der g a|$$
\end{theorem}
If |f :: a -> c| and |g :: a -> d|, then |der f a :: a :-* c| and |der g a :: a :-* d|, so |der f a &&& der g a :: a :-* c :* d|, as needed.

\thmRef{fork} gives us what we need to construct |ad (f &&& g)| compositionally:
\begin{corollary} \corLabel{fork}
|ad| is compositional with respect to |(###)|. Specifically,
\begin{code}
   ad (f &&& g) a
==  {- definition of |ad| -}
   ((f &&& g) a, der (f &&& g) a)
==  {- definition of |(&&&)| -}
   ((f a, g a), der (f &&& g) a)
==  {- \thmRef{fork} -}
   ((f a, g a), der f a &&& der g a)
==  {- refactoring -}
   let { (c,f') = (f a, der f a) ; (d,g') = (g a, der g a) } in ((c,d), (f' &&& g'))
==  {- definition of |ad| -}
   let { (c,f') = ad f a ; (d,g') = ad g a } in ((c,d), (f' &&& g'))
\end{code}
\end{corollary}

\subsectionl{Products in the domain}

There is another, dual, form of composition as well, defined as follows and which we will pronounce ``join'':
\begin{code}
(|||) :: Additive c => (a -> c) -> (b -> c) -> (a :* b -> c)
f ||| g = \ a -> f a + g a
\end{code}
Where |(&&&)| combines two functions with the same domain and pairs their results, |(###)| combines two functions with the same codomain and \emph{adds} their results.\footnote{\mynote{Move this commentary to a later place when I've introduced categories, and cite \cite{Gibbons2002:Calculating}.}

You may have expected a different type and definition, using \emph{sums} instead of products:
\begin{code}
(|||) :: (a -> c) -> (b -> c) -> (a :+ b -> c)
(f ||| g) (Left   a) = f a
(f ||| g) (Right  b) = g b
\end{code}
More generally, |(&&&)| and |(###)| work with categorical products and coproducts.
The categories involved in this paper (functions on additive types, linear maps, and differentiable functions) are all \emph{biproduct} categories, where categorical products and coproducts coincide \needcite{}.
}

Happily, there is differentiation rule for |(###)| as well, having the same poetry as the rules for |(.)| and |(&&&)|, namely that the derivative of a join is a join of the derivatives:\notefoot{Prove, or cite \citep{Spivak65}.}
\begin{theorem}[join rule] \thmLabel{join}
$$|der (f ### g) (a,b) = der f a ### der g b|$$
\end{theorem}
If |f :: a -> c| and |g :: b -> c|, then |der f a :: a :-* c| and |der g b :: b :-* c|, so |der f a ### der g b :: a :* b :-* c|, as needed.

\thmRef{join} is exactly what we need to construct |ad (f ### g)| compositionally:
\begin{corollary} \corLabel{join}
|ad| is compositional with respect to |(###)|. Specifically,
\begin{code}
   ad (f ||| g) (a,b)
==  {- definition of |ad| -}
   ((f ||| g) (a,b), der (f ||| g) (a,b))
==  {- definition of |(###)| -}
   ((f a + g b), der (f ||| g) (a,b))
==  {- \thmRef{join} -}
   ((f a + g b), der f a ||| der g b)
==  {- refactoring -}
   let { (c,f') = (f a, der f a) ; (d,g') = (g b, der g b) } in ((c + d), (f' ||| g'))
==  {- definition of |ad| -}
   let { (c,f') = ad f a ; (d,g') = ad g b } in (c + d, (f' ||| g'))
\end{code}
\end{corollary}

An important point left implicit in the discussion above is that our three combining forms |(.)|, |(&&&)|, and |(###)| all preserve linearity.
This property is what makes it meaningful to use these forms to combine derivatives, i.e., linear maps, as we've done above.

\subsectionl{Linear functions}

A function |f :: a -> b| is said to be \emph{linear} when |f| distributes over (preserves the structure of) vector addition and scalar multiplication, i.e.,
\begin{code}
f (a + a')  == f a + f a'
f (s *^ a)  == s *^ f a
\end{code}
for all |a,a' :: a| and |s| taken from the scalar field underlying |a| and |b|.

In addition to the derivative rules for |(.)|, |(&&&)|, and |(###)|, there is one more broadly useful tool to be added to our collection: \emph{the derivative of every linear function is itself, everywhere}, i.e., for all linear functions |f|,
\begin{theorem}[linear rule] \thmLabel{linear}
$$|der f a == f|$$
\end{theorem}
This statement \citep[Theorem 2-3 (2)]{Spivak65} may sound surprising, but less so when we recall that the |der f a| is a local linear approximation of |f| at |a|, so we're simply saying that linear functions are their own perfect linear approximations.

For example, consider the (linear) function |id = \ a -> a|.
The linearity rule says that |der id a == id|.
When expressed in terms of typical \emph{representations} of linear maps, this property may be expressed as saying that |der id a| is the number one or as an identity matrix (with ones on the diagonal and zeros elsewhere).

%format Rmn = R"^{m+n}"

As another example, consider the function |fst (a,b) = a|, for which the linearity rule says |der fst (a,b) == fst|.
This property, when expressed in terms of typical \emph{representations} of linear maps, would appear as saying that |der fst a| comprises the partial derivatives one and zero if |a, b :: R|.
More generally, if |a :: Rm| and |b :: Rn|, then the Jacobian matrix representation has shape |m :* (m+n)| (ie |m| rows and |m + n| columns) and is formed by the horizontal abutment of an |m :* m| identity matrix on the left with an |m :* n| zero matrix on the right.
This |m :* (m+n)| matrix, however, represents |fst :: Rmn :-* Rm|.
Note how much simpler it is to say |der fst (a,b) == fst|, and with no loss of precision!

Given \thmRef{linear}, we can construct |ad f| for all linear |f|:
\begin{corollary} \corLabel{linear}
|ad| is compositional with respect to linear functions. Specifically,
\begin{code}
   ad f
==  {- definition of |ad| -}
   f &&& der f
==  {- definition of |(&&&)| -}
   \ a -> (f a, der f a)
==  {- \thmRef{linear} -}
   \ a -> (f a, f)
\end{code}
%% ==  {- definition of |(&&&)| -}
%%    f &&& const f
\end{corollary}

\sectionl{Putting the pieces together}

The definition of |ad| is a well-defined specification, not an implementation, since |D| itself is not computable.
Corollaries \ref{cor:compose} through \ref{cor:linear} provide insight into the compositional nature of |ad|, in exactly the form we can now assemble into an efficient, correct-by-construction implementation.

Although differentiation is not computable when given just an arbitrary computable function, we can instead build up differentiable functions compositionally, using exactly the combining forms introduced above, namely |(.)|, |(&&&)|, |(###)|, and linear functions, together with various non-linear primitives.
Computations constructed using that vocabulary are differentiable by construction thanks to Corollaries \ref{cor:compose} through \ref{cor:linear}.
The building blocks above are not just a random assortment, but rather a fundamental language of mathematics, logic, and computation, known as \emph{category theory} \needcite.
Although it would be unpleasant to program directly in this language, its fundamental nature enables instead an automatic conversion from conventionally written functional programs \citep{Lambek:1980:LambdaToCCC,Lambek:1985:CCC,Elliott-2017-compiling-to-categories}.

%format (arr c) = "\mathbin{\to_{"c"}}"

%format CU = "\mathcal{U}"
%format CV = "\mathcal{V}"
%format CF = "\mathcal{F}"

%format <- = `elem`

\subsectionl{Categories}

The central notion in category theory is that of a \emph{category}, comprising \emph{objects} (generalizing sets or types) and \emph{morphisms} (generalizing functions between sets or types).
For the purpose of this paper, we will take objects to be types in our program, and morphisms to be enhanced functions.
We will introduce morphisms using Haskell-style type signatures, such as ``|f :: a ~> b <- CU|'', where ``|~>|'' refers to the morphisms for a category |CU|, with |a| and |b| being the \emph{domain} and \emph{codomain} objects/types (respectively) for |f|.
In most cases, we will omit the ``|<- CU|'', where choice of category is (hopefully) clear from context.
Each category |CU| has a distinguished \emph{identity} morphism |id :: a ~> a <- CU| for every object/type |a| in the category.
For any two morphisms |f :: a ~> b <- CU| and |g :: b ~> c <- CU| (note same category and matching types/objects |b|), there is also the composition |g . f :: a ~> c <- CU|.
The category laws state that
(a) |id| is the left and right identity for composition, and (b) composition is associative.

%format `k` = "\leadsto"
%format k = "(\leadsto)"

Although Haskell's type system is not expressive enough to capture the category laws explicitly, we can express the two required operations as a Haskell type class \needcite:\notefoot{Mention that Haskell doesn't really support infix type constructor variables like |(~>)|.}
\begin{code}
class Category k where
  id   :: a `k` a
  (.)  :: (b `k` c) -> (a `k` b) -> (a `k` c)
\end{code}

You are probably already familiar with at least one example of a category, namely functions, in which |id| and |(.)| are the identity function and function composition.
Another example is the restriction to \emph{computable} functions.
Another are \emph{linear} functions, which we've written ``|a :-* b|'' above.
Still another example is \emph{differentiable} functions, which we can see by noting two facts:
\begin{itemize}
\item The identity function is differentiable, as witnessed by \thmRef{linear} and the linearity if |id|; and
\item The composition of differentiable functions is differentiable, as \thmRef{compose} attests.
\end{itemize}
That the category laws (identity and associativity) hold follows from differentiable functions being a subset of all functions.\footnote{There are many examples of categories besides restricted forms of functions, including relations, logics, partial orders, and even matrices.}

%format --> = "\dashrightarrow"

Each category forms its own world, with morphisms relating objects within that category.
To bridge between these worlds, there are \emph{functors} that connect a category |CU| to a (possibly different) category |CV|.
Such a functor |F| maps objects in |CU| to objects in |CV|, \emph{and} morphisms in |CU| to morphisms in |CV|.
If |f :: u ~> v <- CU| is a morphism, then a \emph{functor} |F| from |CU| to |CV| transforms |f <- CU| to a morphism |F f :: F u --> F v <- CV|, i.e., the domain and codomain of the transformed morphism |F f <- CV| must be the transformed versions of the domain and codomain of |f <- CU|.
The categories in this paper use types as objects, while the functors in this paper map these types to themselves.%
\footnote{In contrast, Haskell's functors stay within the same category and can do change types.}
The functor must also preserve ``categorical'' structure:\footnote{Making the categories explicit, |F (id <- CU) == id <- CV| and |F (g . f <- CU) == F g . F f <- CV|.}
\begin{code}
F id == id

F (g . f) == F g . F f
\end{code}

Crucially to the topic of this paper, \corRefTwo{linear}{compose} say more than that differentiable functions form a category.
They also point us to a new, easily implemented category, for which |ad| is in fact a functor.
This new category is simply the representation that |ad| produces: |a -> b :* (a :-* b)|, considered as having domain |a| and codomain |b|.
The functor nature of |ad| will be exactly what we need to in order to program in a familiar and direct way in a pleasant functional language such as Haskell and have a compiler convert to differentiable functions automatically.

To make the new category more explicit, package the result type of |ad| in a new data type:\notefoot{Maybe format |D a b| as |a ~> b| or some other infix form.}
\begin{code}
newtype D a b = D (a -> b :* (a :-* b))
\end{code}
Then adapt |ad| to use this new data type by simply applying the |D| constructor to the result of |ad|:

%format (hat(x)) = "\hat{"x"}"

%% Why doesn't the following definition work?
%format adf = hat(der)

%format adf = "\hat{"der"}"

\begin{code}
adf :: (a -> b) -> D a b
adf f = D (ad f)
\end{code}

Our goal is to give a |Category| instance for |D| such that |adf| is a functor.
This goal is essentially an algebra problem, and the desired |Category| is the solution to that problem.
Saying that |adf| is a functor is equivalent to the following two conditions for all (suitably typed) functions |f| and |g|:\footnote{The |id| and |(.)| on the left-hand sides are for |D|, while the ones on the right are for |(->)|.}
\begin{code}
id == adf id
D (ad g) . D (ad f) == adf (g . f)
\end{code}
Equivalently, by the definition of |adf|,
\begin{code}
id == D (ad id)
adf g . adf f == D (ad (g . f))
\end{code}
Now recall the following results from \corRefTwo{linear}{compose}:
\begin{code}
ad id == \ a -> (id a, id)
ad (g . f) == \ a -> let { (b,f') = ad f a ; (c,g') = ad g b } in (c, g' . f')
\end{code}
Now use these two facts to rewrite the right-hand sides of the functor specification for |adf|:
\begin{code}
id == D (\ a -> (a,id))
D (ad g) . D (ad f) == D (\ a -> let { (b,f') = ad f a ; (c,g') = ad g b } in (c, g' . f'))
\end{code}
The |id| equation is trivially solvable by \emph{defining} |id = D (\ a -> (a,id))|.
To solve the |(.)| equation, generalize it to a \emph{stronger} condition:\footnote{The new |f| is the old |ad f| and so has changed from type |a -> b| to type |a -> b :* (a :-* b)|.}
\begin{code}
D g . D f == D (\ a -> let { (b,f') = f a ; (c,g') = g b } in (c, g' . f'))
\end{code}
The of this stronger condition is immediate, leading to the following instance as a sufficient condition for |adf| being a functor:
\begin{code}
linearD :: (a -> b) -> D a b
linearD f = D (\ a -> (f a,f))
NOP
instance Category D where
  id == linearD id
  D g . D f == D (\ a -> let { (b,f') = f a ; (c,g') = g b } in (c, g' . f'))
\end{code}
Factoring out |linearD| will also tidy up treatment of other linear functions as well.

Before getting too pleased with this definition, let's remember that for |D| to be a category requires more than having definitions for |id| and |(.)|.
These definitions must also satisfy the identity and composition laws.
How might we go about proving that they do?
Perhaps the most obvious route is take those laws, substitute our definitions of |id| and |(.)|, and reason equationally toward the desired conclusion.
For instance, let's prove that |id . D f == D f| for all |D f :: D a b|:\footnote{Note that \emph{every} morphism in |D| has the form |D f| for some |f|, so it suffices to consider this form.}
\begin{code}
   id . D f
==  {- definition of |id| for |D| -}
   D (\ b -> (b,id)) . D f
==  {- definition of |(.)| for |D| -}
   D (\ a -> let { (b,f') = f a ; (c,g') = (b,id) } in (c, g' . f'))
==  {- substitute |b| for |c| and |id| for |g'| -}
   D (\ a -> let { (b,f') = f a } in (b, id . f'))
==  {-  |id . f' == f'| (category law for functions) -}
   D (\ a -> let { (b,f') = f a } in (b, f'))
==  {- Replace |(b,f')| by its definition -}
   D (\ a -> f a)
==  {- $\eta$-reduction -}
   D f
\end{code}

We can prove the other required properties similarly.
Fortunately, there is a way to bypass the need for these painstaking proofs, and instead rely \emph{only} on our original specification for this |Category| instance, namely that |ad| is a functor.
Well, to buy this prove convenience, we have to make one concession, namely that we consider only morphisms in |D| that arise from |adf|, i.e., only |hat f :: D a b| such that |hat f = adf f| for some |f :: a -> b|.
We can ensure that indeed only such |hat f| do arise by making |D a b| an \emph{abstract} type, i.e., hiding its data |constructor|.\notefoot{%
For the |Category D| instance given above, the painstaking proofs appear to succeed even without this condition.
Am I missing something?}
The slightly more specialized requirement of our first identity property is that |id . adf f == adf f| for any |f :: a -> b|, which we prove as follows:
\begin{code}
   id . adf f
==  {- functor law for |id| (specification of |adf|) -}
   adf id . adf f
==  {- functor law for |(.)| (specification of |adf|) -}
   adf (id . f)
==  {- category law for functions -}
   adf f
\end{code}
The other identity law is proved similarly.
Associativity has a similar flavor as well:
\begin{code}
   adf h . (adf g . adf f)
==  {- functor law for |(.)| (specification of |adf|) -}
   adf h . adf (g . f)
==  {- functor law for |(.)| (specification of |adf|) -}
   adf (h . (g . f))
==  {- category law for functions -}
   adf (h . g) . adf f
==  {- category law for functions -}
   (adf h . adf g) . adf f
\end{code}

Note how mechanical these proofs are.
Each one uses only the functor laws plus the particular category law on functions that corresponds to the one being proved for |D|.
The proofs do \emph{not} rely on anything about the nature |D| or |adf| other than the functor laws.
The importance of this observation is that we \emph{never} need to do these proofs when we specify category instances via a functor.

\subsectionl{Cartesian}

%format ProductCat = Cartesian
%format CoproductCat = Cocartesian
%format CoproductPCat = Cocartesian

% \nc\scrk[1]{_{\hspace{#1}\scaleto{(\leadsto)\!}{4pt}}}
\nc\scrk[1]{}

%format (Prod (k) a b) = a "\times\scrk{-0.4ex}" b
%format (Coprod (k) a b) = a "+\scrk{-0.4ex}" b
%% %format (Exp (k) a b) = a "\Rightarrow\scrk{-0.2ex}" b

\secref{Products in the codomain} introduced another combining form\out{ (pronounced ``fork'')}:\footnote{Consider instead going with |(***)| and |dup| for |Cartesian| and add |jam| for biproducts.}
\begin{code}
(&&&) :: (a -> c) -> (a -> d) -> (a -> c :* d)
f &&& g = \ a -> (f a, g a)
\end{code}
This operation generalizes to play an important role in category theory as part of the notion of a \emph{cartesian category}, along with two projection operations:
\begin{code}
class Category k => ProductCat k where
  exl    ::  (Prod k a b) `k` a
  exr    ::  (Prod k a b) `k` b
  (&&&)  ::  (a `k` c)  -> (a `k` d)  -> (a `k` (Prod k c d))
\end{code}
More generally, each category |k| can have its own notion of products, but cartesian products (ordered pairs) suffice for this paper.
For functions,\notefoot{Give a similar instance for |Category (->)|, and don't bother repeating the definition of |(&&&)| just above.}
\begin{code}
instance ProductCat (->) where
  exl = \ (a,b) -> a
  exr = \ (a,b) -> b
  f &&& g = \ a -> (f a, g a)
\end{code}

Two cartesian categories can be related by a \emph{cartesian functor}, which is a functor that also preserves the cartesian structure.
That is, a cartesian functor |F| from cartesian category |CU| to cartesian category |CV|, besides mapping objects and morphisms in |CU| to counterparts in |CV| while preserving the category structure (|id| and |(.)|), \emph{also} preserves the cartesian structure:
\begin{code}
F exl  == exl

F exr  == exr

F (f &&& g) == F f &&& F g
\end{code}
Just as \corRefTwo{linear}{compose} were key to deriving a correct-by-construction |Category| instance from the specification that |adf| is a functor, \corRefTwo{linear}{fork} guides correct-by-construction |ProductCat| instance from the specification that |adf| is a cartesian functor.

Let |F| be |adf| in the reversed forms of cartesian functor equations above, and expand |adf| to its definition as |D . ad|:
\begin{code}
exl == D (ad exl)

exr == D (ad exr)

D (ad f) &&& D (ad g) == D (ad (f &&& g))
\end{code}
Next, by \corRefTwo{linear}{fork}, together with the linearity of |exl| and |exr|,
\begin{code}
ad exl == \ (a,b) -> (a, exl)

ad exr == \ (a,b) -> (b, exr)

ad (f &&& g) == \ a -> let { (c,f') = ad f a ; (d,g') = ad g a } in ((c,d), f' &&& g')
\end{code}
Now substitute the left-hand sides of these three properties into the right-hand sides of the of the cartesian functor properties for |adf|, and \emph{strengthen} the last condition (on |(&&&)|) by generalizing from |ad f| and |ad g|:
\begin{code}
exl == linearD exr

exr == linearD exr

D f &&& D g == D(\a -> let { (c,f') = f a ; (d,g') = g a } in ((c,d), f' &&& g'))
\end{code}
This somewhat strengthened form of the specification can be turned directly into a sufficient definition:
\begin{code}
instance ProductCat D where
  exl = linearD exl
  exr = linearD exr
  D f &&& D g = D (\a -> let { (c,f') = f a ; (d,g') = g a } in ((c,d), f' &&& g'))
\end{code}

\subsectionl{Cocartesian}

Cartesian categories have a dual, known as \emph{cocartesian categories}, with each cartesian operation having a mirror image with morphisms reversed (swapping domain and codomain):\notefoot{Mention sums again.}
\begin{code}
class Category k => CoproductPCat k where
  inl    ::  Additive b => a `k` (Prod k a b)
  inr    ::  Additive a => b `k` (Prod k a b)
  (|||)  ::  Additive c => (a `k` c)  -> (b `k` c)  -> ((Prod k a b) `k` c)
\end{code}
Unlike |Category| and |ProductCat|, we've had to add an additivity requirement (having a notion of addition and corresponding identity) to the types involved, in order to have an instance for functions\out{\footnote{Alternatively, we can skip the instance for |(->)| and instead begin in a category of functions on additive types.}}:
%format zero = 0
%format ^+^ = +
\begin{code}
instance CoproductPCat (->) where
  inl  = \ a -> (a,zero)
  inr  = \ b -> (zero,b)
  f ||| g = \ a -> f a ^+^ g a
\end{code}
Unsurprisingly, there is a notion of \emph{cocartesian functor}, saying that the cocartesian structure is preserved, i.e.,
\begin{code}
F inl  == inl

F inr  == inr

F (f ||| g) == F f ||| F g
\end{code}
From the specification that |adf| is a cocartesian functor, along with \corRef{join} and the linearity of |inl| and |inr|, we can derive a correct-by-construction |CoproductPCat| instance for differentiable functions:
\begin{code}
instance CoproductPCat D where
  inl  = linearD inl
  inr  = linearD inr
  D f ||| D g = D (\ (a,b) -> let { (c,f') = f a ; (d,g') = g b } in (c ^+^ d, f' ||| g'))
\end{code}

\subsectionl{Numeric operations}

So far, the vocabulary we've considered comprises linear functions and combining forms (|(.)|, |(&&&)|, and |(###)|) that preserve linearity.
To make differentiation at all interesting, we'll need some non-linear primitives as well.
Let's now add these primitives, while continuing to derive correct implementations from simple, regular specifications in terms of structure preservation.
We'll define a collection of interfaces for numeric operations, roughly imitating Haskell's numeric type class hierarchy\needcite.

Haskell provides the following basic class:
\begin{code}
class Num a where
  negate :: a -> a
  (+), (*) :: a -> a -> a
  ...
\end{code}
Although this class can accommodate many different types of ``numbers'', the class operations are all committed to being functions.
A more flexible alternative allows operations to be non-functions as well:
\begin{code}
class NumCat k a where
  negateC :: a `k` a
  addC, mulC :: (a :* a) `k` a
  ...
\end{code}
Besides generalizing from |(->)| to |k|, we've also uncurried the operations, so as demand less of supporting categories |k|.
There are similar classes for other operations, such as division, powers and roots, and transcendental functions (|sin|, |cos|, |exp| etc).
Instances for functions use the operations from |Num| etc:
%format * = "\cdot"
\begin{code}
instance Num a => NumCat (->) a where
  negateC = negate
  addC  = uncurry (+)
  mulC  = uncurry (*)
  ...
\end{code}

Differentiation rules for these operations are part of basic differential calculus:
\begin{code}
der (negate u) == negate (der u)
der (u + v) == der u + der v
der (u * v) == u * der v + v * der u
\end{code}
This conventional form is unnecessarily complex, as each of these rules involves not just a numeric operation, but also the chain rule itself.
This form is also imprecise about the nature of |u| and |v|.
If they are functions, then one needs to explain arithmetic on functions; and if they are not functions, then differentiation of non-functions needs explanation.

A simpler presentation is to remove the arguments and talk about differentiating the primitive operations \emph{themselves}, without context.
We already have the chain rule to account for context, so we do not need to involve it in every numeric operation.
Since negation and (uncurried) addition are linear, we already know how to differentiate them.
Multiplication is a little more involved \citep[Theorem 2-3 (2)]{Spivak65}:
\begin{code}
der mulC (a,b) = \ (da,db) -> a*db + da*b
\end{code}
Note the linearity of the right-hand side, so that the derivative of |mulC| at |(a,b)| for real values has the expected type: |R :* R :-* R|.\footnote{The derivative of uncurried multiplication generalizes to an arbitrary \emph{bilinear} function |f :: a :* b -> c| \citep[Problem 2-12]{Spivak65}:
\begin{code}
der f (a,b) = \ (da,db) -> f (a,db) + f (da,b)
\end{code}
}

This product rule, along with the linearity of negation and uncurried addition, enables using the same style of derivation as with operations from |Category|, |Cartesian|, and |Cocartesian| above.
As usual, specify the |NumCat| instance for differentiable functions by saying that |adf| preserves |NumCat| structure, i.e., |adf negateC == negateC|, |adf addC == addC|, and |adf mulC == mulC|.
Reasoning as before, we get another correct-by-construction instance for differentiable functions:
\begin{code}
instance NumCat D where
  negateC = linearD negateC
  addC  = linearD addC
  mulC  = D (\ (a,b) -> (a * b, \ (da,db) -> b*da + a*db))
\end{code}

Similar reasoning applies to other numeric operations, e.g.,
\begin{code}
instance FloatingCat D where
  sinC = D (\ a -> (sin a, \ da -> cos a * da))
  cosC = D (\ a -> (cos a, \ da -> - sin a * da))
  expC = D (\ a -> let e = exp a in (e, \ da -> e * da))
  ...
\end{code}
A bit of refactoring makes for tidier definitions:
\begin{code}
scale :: a -> a -> a
scale a = \ da -> a * da
NOP
instance FloatingCat D where
  sinC = D (\ a -> (sin a, scale (cos a)))
  cosC = D (\ a -> (cos a, scale (- sin a)))
  expC = D (\ a -> let e = exp a in (e, scale e))
\end{code}
In what follows, the |scale| operation will play a more important role than merely tidying definitions.

\sectionl{Programming as defining and solving algebra problems}

Stepping back to consider what we've done, a general recipe emerges:\notefoot{Go over the wording of this section to make as clear as I can.}
\begin{itemize}
\item Start with an expensive or even non-computable specification (here involving differentiation).
\item Build the desired result into the representation of a new data type (here as the combination of a function and its derivative).
\item Try to show that conversion from a simpler form (here regular functions) to the new data type---even if not computable---is \emph{compositional} with respect to a well-understood algebraic abstraction (here category).
\item If compositionality fails (as with |der|, unadorned differentiation, in \secref{Sequential composition}), then examine the failure to find an augmented specification, iterating as needed until converging on a representation and corresponding specification that \emph{is} compositional.
\item Set up an algebra problem whose solution will be an instance of the well-understood algebraic abstraction for the chosen representation.
These algebra problems always have a particular stylized form, namely that the operation being solved for is a \emph{homomorphism} for the chosen abstraction (here a category homomorphism, also called a ``functor'').
\item Solve the algebra problem by using the compositionality properties.
\item Rest assured that the solution satisfies the required laws, at least when the new data type is kept abstract, thanks to the homomorphic specification.
\end{itemize}
The result of this recipe is not quite an implementation of our homomorphic specification, which may after all be non-computable.
Rather, it gives a computable alternative that is almost as useful: if the input to the specified conversion is expressed in vocabulary of the chosen algebraic abstraction, then a re-interpretation of that vocabulary in the new data type is the result of the (possibly non-computable) specification.
Furthermore, if we can \emph{automatically} convert conventionally written functional programs into the chosen algebraic vocabulary (as in \citep{Elliott-2017-compiling-to-categories}), then those programs can be re-interpreted to compute the desired specification.

\sectionl{Examples}

Let's look at some AD examples.
In this section and later ones, we will use a few running examples:
\begin{code}
sqr :: Num a => a -> a
sqr a = a * a

magSqr :: Num a => a :* a -> a
magSqr (a,b) = sqr a + sqr b

cosSinProd :: Floating a => a :* a -> a :* a
cosSinProd (x,y) = (cos z, sin z) where z = x * y
\end{code}
A compiler plugin converts these definitions to categorical vocabulary \citep{Elliott-2017-compiling-to-categories}:
\begin{code}
sqr = mulC . (id &&& id)

magSqr = addC . (mulC . (exl &&& exl) &&& mulC . (exr &&& exr))

cosSinProd = (cosC &&& sinC) . mulC
\end{code}
To visualize computations before differentiation, we can interpret these categorical expressions in a category of graphs \citep[Section 7]{Elliott-2017-compiling-to-categories}, with the results rendered in \figreftwo{magSqr}{cosSinProd}.
\figp{
\figoneW{0.35}{magSqr}{|magSqr|}}{
\hspace{1in}
\figoneW{0.35}{cosSinProd}{|cosSinProd|}}
To see the differentiable versions, interpret these same expressions in the category of differentiable functions (|D| from \secref{Categories}), remove the |D| constructors to reveal the function representation, convert these functions to categorical form as well, and finally interpret the result in the graph category.
The results are rendered in \figreftwo{magSqr-adf}{cosSinProd-adf}.
\figp{
\figone{magSqr-adf}{|adf magSqr|}}{
\figone{cosSinProd-adf}{|adf cosSinProd|}}
Note that the derivatives are (linear) functions, as depicted in boxes.
Also note the sharing of work between the a function's result and its derivative in \figref{cosSinProd-adf}.\notefoot{Introduce the term ``primal'' early on and use it throughout.}

%if False
%endif

\sectionl{To do}

\begin{itemize}
\item The rest of the talk:
  \begin{itemize}
  \item {Generalizing automatic differentiation}
  \item {A vocabulary for linear arrows}
  \item {Extracting a data representation}
  \item {Generalized matrices}
  \item {Efficiency of composition}
  \item {Left-associating composition (RAD)}
  \item {Continuation category}
  \item {Reverse-mode AD without tears}
  \item {Duality}
  \item {Backpropagation}
  \item {Reverse AD examples}
  \item {Incremental evaluation}
  \item {Symbolic vs automatic differentiation}
  \item {Conclusions}
  \end{itemize}
\item More biproduct operations: |(***)|, |dup|, |jam|, |(+)| (arrow addition).
\item Indexed biproducts.
\item Relate the methodology of this paper to denotational design \citep{Elliott2009-type-class-morphisms-TR}.
They're quite similar.
\item Relate also to algebra of programming \citep{BirddeMoor96:Algebra}.
\end{itemize}

\bibliography{bib}

\end{document}


