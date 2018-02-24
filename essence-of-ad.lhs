%% -*- latex -*-

%% While editing/previewing, use 12pt and tiny margin.
\documentclass[12]{article}  % fleqn,
\usepackage[margin=0.12in]{geometry}  % 0.9in

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

\newtheorem{theorem}{Theorem}%[section]
\nc\thmLabel[1]{\label{thm:#1}}
\nc\thmRef[1]{Theorem \ref{thm:#1}}
\nc\thmRefTwo[2]{Theorems \ref{thm:#1} and \ref{thm:#2}}
\nc\thmRefs[2]{Theorems \ref{thm:#1} through \ref{thm:#2}}

\newtheorem{corollary}{Corollary}[theorem]
\nc\corLabel[1]{\label{cor:#1}}
\nc\corRef[1]{Corollary \ref{cor:#1}}
\nc\corRefTwo[2]{Corollaries \ref{cor:#1} and \ref{cor:#2}}
\nc\corRefs[2]{Corollaries \ref{cor:#1} through \ref{cor:#2}}

\newtheorem{lemma}[theorem]{Lemma}
\nc\lemLabel[1]{\label{lem:#1}}
\nc\lemRef[1]{Lemma \ref{lem:#1}}
\nc\lemRefTwo[2]{Lemma \ref{lem:#1} and \ref{lem:#2}}
\nc\lemRefs[2]{Lemma \ref{lem:#1} through \ref{lem:#2}}

\setlength{\blanklineskip}{2ex}

%% Needs a "%"after \end{closerCodePars} to avoid a blank space. Fixable?
\newenvironment{closerCodePars}{\setlength{\blanklineskip}{1.3ex}}{}

%format ith = i"th"
%format ith = i"^{\text{th}}"

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
When the codomain |Rn| is also non-scalar (i.e., |n > 1|), we have a \emph{matrix} $\mathbf J$ (the \emph{Jacobian}), with $\mathbf J_{ij} = \partial f_i / \partial x_j$ for $i \in \set{1,\ldots,n}$, where each $f_i$ projects out the |ith| scalar value from the result of $f$.

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
In particular, the \emph{Hessian} matrix $H$ corresponds to the second derivative of a function |f :: Rm -> R|, having $m$ rows and $m$ columns and satisfying the symmetry condition $H_{i,j} \equiv H_{j,i}$.

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
==  {- definition of |(.)| -}
   (g (f a), der (g . f) a)
==  {- \thmRef{compose} -}
   (g (f a), der g (f a) . der f a)
==  {- refactoring to share |f a| -}
   let b = f a in (g b, der g b . der f a)
==  {- refactoring to show compositionality -}
   let { (b,f') = ad f a ; (c,g') = ad g b } in (c, g' . f')
\end{code}
\end{corollary}

\subsectionl{Parallel composition}

The chain rule, telling how to differentiate sequential compositions, gets a lot of attention in calculus classes and in automatic and symbolic differentiation.\notefoot{To do: introduce AD and SD early.}
There are other important ways to combine functions, however, and examining them yields additional helpful tools.
One other tool combines two functions in \emph{parallel}:\footnote{By ``parallel'', I simply mean without data dependencies. Operationally, the two functions can be applied simultaneously or not.}
\begin{code}
(***) :: (a -> c) -> (b -> d) -> (a :* b -> c :* d)
f *** g = \ (a,b) -> (f a, g b)
\end{code}
We will sometimes refer to the |(***)| operation as ``cross'' \citep{Gibbons2002:Calculating}.

%% %% Move to the later introduction of |(&&&)|.
%% Note that it can be used to give a terser specification: |ad f = f &&& der f|.

While the derivative of the sequential composition is a sequential composition of derivatives, the derivative of a parallel composition is a parallel composition of the derivatives \citep[variant of Theorem 2-3 (3)]{Spivak65}:\notefoot{Is there a name for this rule? I've never seen it mentioned.}
\begin{theorem}[cross rule] \thmLabel{cross}
$$|der (f *** g) (a,b) == der f a *** der g b|$$
\end{theorem}
If |f :: a -> c| and |g :: b -> d|, then |der f a :: a :-* c| and |der g b :: b :-* d|, so |der f a *** der g b :: a :* b :-* c :* d|, as needed.

\thmRef{cross} gives us what we need to construct |ad (f *** g)| compositionally:
\begin{corollary} \corLabel{cross}
|ad| is compositional with respect to |(***)|. Specifically,
\begin{code}
   ad (f *** g) (a,b)
==  {- definition of |ad| -}
   ((f *** g) (a,b), der (f *** g) (a,b))
==  {- definition of |(***)| -}
   ((f a, g b), der (f *** g) (a,b))
==  {- \thmRef{cross} -}
   ((f a, g b), der f a *** der g b)
==  {- refactoring -}
   let { (c,f') = (f a, der f a) ; (d,g') = (g b, der g b) } in ((c,d), (f' *** g'))
==  {- definition of |ad| -}
   let { (c,f') = ad f a ; (d,g') = ad g b } in ((c,d), (f' *** g'))
\end{code}
\end{corollary}

%if False

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

%endif

An important point left implicit in the discussion above is that sequential and parallel composition preserve linearity.
This property is what makes it meaningful to use these forms to combine derivatives, i.e., linear maps, as we've done above.

\subsectionl{Linear functions}

A function |f :: u -> v| is said to be \emph{linear} when |f| distributes over (preserves the structure of) vector addition and scalar multiplication, i.e.,
\begin{code}
f (a + a')  == f a + f a'
f (s *^ a)  == s *^ f a
\end{code}
for all |a,a' :: v| and |s| taken from the scalar field underlying |u| and |v|.

In addition to the derivative rules for |(.)|, |(&&&)|, and |(###)|, there is one more broadly useful tool to be added to our collection: \emph{the derivative of every linear function is itself, everywhere}, i.e., for all linear functions |f|,
\begin{theorem}[linear rule] \thmLabel{linear}
$$|der f a == f|$$
\end{theorem}
This statement \citep[Theorem 2-3 (2)]{Spivak65} may sound surprising at first, but less so when we recall that the |der f a| is a local linear approximation of |f| at |a|, so we're simply saying that linear functions are their own perfect linear approximations.

For example, consider the (linear) function |id = \ a -> a|.
The linearity rule says that |der id a == id|.
When expressed in terms of typical \emph{representations} of linear maps, this property may be expressed as saying that |der id a| is the number one or as an identity matrix (with ones on the diagonal and zeros elsewhere).

%% %format Rmn = R"^{m+n}"

As another example, consider the (linear) function |fst (a,b) = a|, for which the linearity rule says |der fst (a,b) == fst|.
This property, when expressed in terms of typical \emph{representations} of linear maps, would appear as saying that |der fst a| comprises the partial derivatives one and zero if |a, b :: R|.
More generally, if |a :: Rm| and |b :: Rn|, then the Jacobian matrix representation has shape |m :* (m+n)| (ie |m| rows and |m + n| columns) and is formed by the horizontal abutment of an |m :* m| identity matrix on the left with an |m :* n| zero matrix on the right.
This |m :* (m+n)| matrix, however, represents |fst :: Rm :* Rn :-* Rm|.
Note how much simpler it is to say |der fst (a,b) == fst|, and with no loss of precision!

Given \thmRef{linear}, we can construct |ad f| for all linear |f|:
\begin{corollary} \corLabel{linear}
|ad| is compositional with respect to linear functions. Specifically, if |f| is linear, then
\begin{code}
   ad f
==  {- definition of |ad| -}
   \ a -> (f a, der f a)
==  {- \thmRef{linear} -}
   \ a -> (f a, f)
\end{code}
%% ==  {- definition of |(&&&)| -}
%%    f &&& const f
\end{corollary}

\sectionl{Putting the pieces together}

The definition of |ad| is a well-defined specification, but it is not an implementation, since |D| itself is not computable.
\corRefs{compose}{linear} provide insight into the compositional nature of |ad|, in exactly the form we can now assemble into an efficient, correct-by-construction implementation.

Although differentiation is not computable when given just an arbitrary computable function, we can instead build up differentiable functions compositionally, using exactly the combining forms introduced above, namely |(.)|, |(***)|, and linear functions, together with various non-linear primitives.
Computations constructed using that vocabulary are differentiable by construction thanks to \corRefs{compose}{linear}.
The building blocks above are not just a random assortment, but rather a fundamental language of mathematics, logic, and computation, known as \emph{category theory} \needcite.
Although it would be unpleasant to program directly in this language, its foundational nature enables instead an automatic conversion from conventionally written functional programs \citep{Lambek:1980:LambdaToCCC,Lambek:1985:CCC,Elliott-2017-compiling-to-categories}.

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
The category laws (identity and associativity) hold, because differentiable functions form a subset of all functions.\footnote{There are many examples of categories besides restricted forms of functions, including relations, logics, partial orders, and even matrices.}

%format --> = "\dashrightarrow"

Each category forms its own world, with morphisms relating objects within that category.
To bridge between these worlds, there are \emph{functors} that connect a category |CU| to a (possibly different) category |CV|.
Such a functor |F| maps objects in |CU| to objects in |CV|, \emph{and} morphisms in |CU| to morphisms in |CV|.
If |f :: u ~> v <- CU| is a morphism, then a \emph{functor} |F| from |CU| to |CV| transforms |f <- CU| to a morphism |F f :: F u --> F v <- CV|, i.e., the domain and codomain of the transformed morphism |F f <- CV| must be the transformed versions of the domain and codomain of |f <- CU|.
The categories in this paper use types as objects, while the functors in this paper map these types to themselves.%
\footnote{In contrast, Haskell's functors stay within the same category and can do change types.}
The functor must also preserve ``categorical'' structure:\footnote{Making the categories explicit, |F (id <- CU) == id <- CV| and |F (g . f <- CU) == F g . F f <- CV|.}
\begin{closerCodePars}
\begin{code}
F id == id

F (g . f) == F g . F f
\end{code}
\end{closerCodePars}%

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

\begin{closerCodePars}
Our goal is to give a |Category| instance for |D| such that |adf| is a functor.
This goal is essentially an algebra problem, and the desired |Category| instance is the solution to that problem.
Saying that |adf| is a functor is equivalent to the following two conditions for all suitably typed functions |f| and |g|:\footnote{The |id| and |(.)| on the left-hand sides are for |D|, while the ones on the right are for |(->)|.}
\begin{code}
id == adf id

adf g . adf f == adf (g . f)
\end{code}
Equivalently, by the definition of |adf|,
\begin{code}
id == D (ad id)

D (ad g) . D (ad f) == D (ad (g . f))
\end{code}
Now recall the following results from \corRefTwo{linear}{compose}:
\begin{code}
ad id == \ a -> (id a, id)

ad (g . f) == \ a -> let { (b,f') = ad f a ; (c,g') = ad g b } in (c, g' . f')
\end{code}
Then use these two facts to rewrite the right-hand sides of the functor specification for |adf|:
\begin{code}
id == D (\ a -> (a,id))

D (ad g) . D (ad f) == D (\ a -> let { (b,f') = ad f a ; (c,g') = ad g b } in (c, g' . f'))
\end{code}
The |id| equation is trivially solvable by \emph{defining} |id = D (\ a -> (a,id))|.
To solve the |(.)| equation, generalize it to a \emph{stronger} condition:\footnote{The new |f| is the old |ad f| and so has changed type from |a -> b| to |a -> b :* (a :-* b)|. Likewise for |g|.}
\begin{code}
D g . D f == D (\ a -> let { (b,f') = f a ; (c,g') = g b } in (c, g' . f'))
\end{code}
The solution of this stronger condition is immediate, leading to the following instance as a sufficient condition for |adf| being a functor:
\end{closerCodePars}
\begin{code}
linearD :: (a -> b) -> D a b
linearD f = D (\ a -> (f a,f))

instance Category D where
  id == linearD id
  D g . D f == D (\ a -> let { (b,f') = f a ; (c,g') = g b } in (c, g' . f'))
\end{code}
Factoring out |linearD| will also tidy up treatment of other linear functions.

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
==  {-  |id . f' == f'| (category law) -}
   D (\ a -> let { (b,f') = f a } in (b, f'))
==  {- Replace |(b,f')| by its definition -}
   D (\ a -> f a)
==  {- $\eta$-reduction -}
   D f
\end{code}

We can prove the other required properties similarly.
Fortunately, there is a way to bypass the need for these painstaking proofs, and instead rely \emph{only} on our original specification for this |Category| instance, namely that |ad| is a functor.
To buy this proof convenience, we have to make one concession, namely that we consider only morphisms in |D| that arise from |adf|, i.e., only |hat f :: D a b| such that |hat f = adf f| for some |f :: a -> b|.
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
==  {- category law -}
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
==  {- category law -}
   adf ((h . g) . f)
==  {- functor law for |(.)| (specification of |adf|) -}
   adf (h . g) . adf f
==  {- functor law for |(.)| (specification of |adf|) -}
   (adf h . adf g) . adf f
\end{code}

Note how mechanical these proofs are.
Each one uses only the functor laws plus the particular category law on functions that corresponds to the one being proved for |D|.
The proofs do \emph{not} rely on anything about the nature |D| or |adf| other than the functor laws.
The importance of this observation is that we \emph{never} need to do these proofs when we specify category instances via a functor.

\subsectionl{Monoidal categories}

%format MonoidalPCat = Monoidal

% \nc\scrk[1]{_{\hspace{#1}\scaleto{(\leadsto)\!}{4pt}}}
\nc\scrk[1]{}

%format Prod (k) a b = a "\times\scrk{-0.4ex}" b

\secref{Parallel composition} introduced parallel composition.
This operation generalizes to play an important role in category theory as part of the notion of a \emph{monoidal category}:
\begin{code}
class Category k => MonoidalPCat k where
  (***) :: (a `k` c) -> (b `k` d) -> ((Prod k a b) `k` (Prod k c d))

instance MonoidalPCat (->) where
  f *** g = \ (a,b) -> (f a, g b)
\end{code}
More generally, a category |k| can be monoidal over constructions other than products, but cartesian products (ordered pairs) suffice for this paper.

Two monoidal categories can be related by a \emph{monoidal functor}, which is a functor that also preserves the monoidal structure.
That is, a monoidal functor |F| from monoidal category |CU| to monoidal category |CV|, besides mapping objects and morphisms in |CU| to counterparts in |CV| while preserving the category structure (|id| and |(.)|), \emph{also} preserves the monoidal structure:
\begin{code}
F (f *** g) == F f *** F g
\end{code}
Just as \corRefTwo{compose}{linear} were key to deriving a correct-by-construction |Category| instance from the specification that |adf| is a functor, \corRef{cross} guides correct-by-construction |MonoidalPCat| instance from the specification that |adf| is a monoidal functor.

Let |F| be |adf| in the reversed forms of monoidal functor equation above, and expand |adf| to its definition as |D . ad|:
\begin{code}
D (ad f) *** D (ad g) == D (ad (f *** g))
\end{code}
By \corRef{cross},
\begin{code}
ad (f *** g) == \ (a,b) -> let { (c,f') = ad f a ; (d,g') = ad g b } in ((c,d), f' *** g')
\end{code}
Now substitute the left-hand side of this property into the right-hand side of the of the monoidal functor property for |adf|, and \emph{strengthen} the condition by generalizing from |ad f| and |ad g|:
\begin{code}
D f *** D g == D (\ (a,b) -> let { (c,f') = f a ; (d,g') = g b } in ((c,d), f' *** g'))
\end{code}
This somewhat strengthened form of the specification can be turned directly into a sufficient definition:
\begin{code}
instance MonoidalPCat D where
  D f *** D g = D (\ (a,b) -> let { (c,f') = f a ; (d,g') = g b } in ((c,d), f' *** g'))

\end{code}

\subsectionl{Cartesian categories}

%format TerminalCat = Terminal
%format CoterminalCat = Coterminal

%format ProductCat = Cartesian
%format CoproductCat = Cocartesian
%format CoproductPCat = Cocartesian

%format (Coprod (k) a b) = a "+\scrk{-0.4ex}" b
%% %format (Exp (k) a b) = a "\Rightarrow\scrk{-0.2ex}" b

The |MonoidalPCat| abstraction gives a way to combine two functions but not separate them.
It also gives no way to duplicate or discard information.
These additional abilities require another algebraic abstraction, namely that of \emph{cartesian category}, adding operations for projection and duplication:
\begin{code}
class Category k => ProductCat k where
  exl  :: (Prod k a b) `k` a
  exr  :: (Prod k a b) `k` b
  dup  :: a `k` (Prod k a a)
\end{code}
For functions,\notefoot{Give a similar instance for |Category (->)|, and don't bother repeating the definition of |(&&&)| just above.}
\begin{code}
instance ProductCat (->) where
  exl  = \ (a,b) -> a
  exr  = \ (a,b) -> b
  dup  = \ a -> (a,a)
\end{code}

\begin{closerCodePars}
Two cartesian categories can be related by a \emph{cartesian functor}, which is a functor that also preserves the cartesian structure.
That is, a cartesian functor |F| from cartesian category |CU| to cartesian category |CV|, besides mapping objects and morphisms in |CU| to counterparts in |CV| while preserving the category structure (|id| and |(.)|), \emph{also} preserves the cartesian structure:
\begin{code}
F exl  == exl
F exr  == exr
F dup  == dup
\end{code}
Just as \corRefs{compose}{linear} were key to deriving a correct-by-construction |Category| and |MonoidalPCat| instances from the specification that |adf| is a functor and a monoidal functor respectively, \corRef{linear} guides correct-by-construction |ProductCat| instance from the specification that |adf| is a cartesian functor.

Let |F| be |adf| in the reversed forms of cartesian functor equations above, and expand |adf| to its definition as |D . ad|:
\begin{code}
exl  == D (ad exl)
exr  == D (ad exr)
dup  == D (ad dup)
\end{code}
Next, by \corRef{linear}, together with the linearity of |exl|, |exr|, and |dup|,
\begin{code}
ad exl  == \ p  -> (exl  p  , exl  )
ad exr  == \ p  -> (exr  p  , exr  )
ad dup  == \ a  -> (dup  a  , dup  )
\end{code}
Now substitute the left-hand sides of these three properties into the right-hand sides of the of the cartesian functor properties for |adf|, and recall the definition of |linearD|:
\begin{code}
exl  == linearD exr
exr  == linearD exr
dup  == linearD dup
\end{code}
\end{closerCodePars}%
This form of the specification can be turned directly into a sufficient definition:
\begin{code}
instance ProductCat D where
  exl  = linearD exl
  exr  = linearD exr
  dup  = linearD dup
\end{code}

\subsectionl{Cocartesian categories}

%format inlP = inl
%format inrP = inr
%format jamP = jam

Cartesian categories have a dual, known as \emph{cocartesian categories}, with each cartesian operation having a mirror image with morphisms reversed (swapping domain and codomain):\notefoot{Mention sums again.}
\begin{code}
class Category k => CoproductPCat k where
  inl  ::  Additive b => a `k` (Prod k a b)
  inr  ::  Additive a => b `k` (Prod k a b)
  jam  ::  Additive c => (Prod k a a) `k` a
\end{code}
Unlike |Category| and |ProductCat|, we've had to add an additivity requirement (having a notion of addition and corresponding identity) to the types involved, in order to have an instance for functions:\notefoot{Alternatively, skip the instance for |(->)| and instead begin in a category |(-+>)| of functions on additive types. I guess I'll have to change the category used in |Cont k r| from |(->)| to |(-+>)|.}
%format zero = 0
%format ^+^ = +
\begin{code}
instance CoproductPCat (->) where
  inl  = \ a -> (a,zero)
  inr  = \ b -> (zero,b)
  jam  = \ (a,b) -> a ^+^ b
\end{code}
Unsurprisingly, there is a notion of \emph{cocartesian functor}, saying that the cocartesian structure is preserved, i.e.,
\begin{closerCodePars}
\begin{code}
F inl  == inl
F inr  == inr
F jam  == jam
\end{code}
\end{closerCodePars}
From the specification that |adf| is a cocartesian functor and the linearity of |inl|, |inr|, and |jam|, we can derive a correct-by-construction |CoproductPCat| instance for differentiable functions:
\begin{code}
instance CoproductPCat D where
  inl  = linearD inl
  inr  = linearD inr
  jam  = linearD jam
\end{code}

\subsectionl{Derived operations}

With |dup|, we can define an alternative to |(***)| that takes two morphisms sharing a domain:
\begin{code}
(&&&) :: Cartesian k => (a `k` c) -> (a `k` d) -> (a `k` (Prod k c d))
f &&& g = (f *** g) . dup
\end{code}
The |(&&&)| operation is sometimes called ``fork'' \citep{Gibbons2002:Calculating} and is particularly useful for translating from the $\lambda$-calculus to categorical form \citep[Section 3]{Elliott-2017-compiling-to-categories}.

Dually, |jam| lets us define a second alternative to |(***)| for two morphisms sharing a \emph{codomain}:\notefoot{Do I use |(###)|?}
\begin{code}
(|||) :: Cocartesian k => (c `k` a) -> (d `k` a) -> ((Prod k c d) `k` a)
f ||| g = jam . (f +++ g)
\end{code}
The |(###)| operation is sometimes called ``join'' \citep{Gibbons2002:Calculating}.

In their uncurried form, these two operations are invertible:\notefoot{For proofs, cite \cite{Gibbons2002:Calculating}.}
\begin{code}
fork    :: Cartesian    k => (a `k` c) :* (a `k` d) -> (a `k` (Prod k c d))
unfork  :: Cartesian    k => (a `k` ((Prod k c d))) -> (a `k` c) :* (a `k` d)

join    :: Cocartesian  k => (c `k` a) :* (d `k` a) -> ((Prod k c d) `k` a)
unjoin  :: Cocartesian  k => ((Prod k c d) `k` a) -> (c `k` a) :* (d `k` a)
\end{code}
where
\begin{code}
fork (f,g) = f &&& g
unfork h = (exl . h, exr . h)

join (f,g) = f ||| g
unjoin h = (h . inl, h . inr)
\end{code}

\subsectionl{Abelian categories}

Another perspective on the operations we've considered is that morphisms of any particular domain and codomain form an abelian group.
The zero for |a `k` b| results from the composition of initial and terminal morphisms\notefoot{Define |TerminalCat| and |CoterminalCat| earlier.}:
%format zeroC = 0
%format zeroC = "\mathbf{0}"
%format `plusC` = "\boldsymbol{+}"
%format plusC = (`plusC`)
\begin{code}
type AbelianCat k = (ProductCat k, CoproductPCat k, TerminalCat k, CoterminalCat k)

zeroC :: AbelianCat k => a `k` b
zeroC = ti . it

plusC :: AbelianCat k => Binop (a `k` b)
f `plusC` g = jamP . (f *** g) . dup
\end{code}

\mynote{Relate |zeroC| and |plusC| to existing vocabulary in other ways as well.}

\subsectionl{Numeric operations}

So far, the vocabulary we've considered comprises linear functions and combining forms (|(.)|, |(&&&)|, and |(###)|) that preserve linearity.
To make differentiation interesting, we'll need some non-linear primitives as well.
Let's now add these primitives, while continuing to derive correct implementations from simple, regular specifications in terms of structure preservation.
We'll define a collection of interfaces for numeric operations, roughly imitating Haskell's numeric type class hierarchy \needcite.

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
Besides generalizing from |(->)| to |k|, we've also uncurried the operations, so as to demand less of supporting categories |k|.
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

A simpler presentation is to remove the arguments and talk about differentiating the primitive operations directly.
We already have the chain rule to account for context, so we do not need to involve it in every numeric operation.
Since negation and (uncurried) addition are linear, we already know how to differentiate them.
Multiplication is a little more involved \citep[Theorem 2-3 (2)]{Spivak65}:
\begin{code}
der mulC (a,b) = \ (da,db) -> da*b + a*db
\end{code}
Note the linearity of the right-hand side, so that the derivative of |mulC| at |(a,b)| for real values has the expected type: |R :* R :-* R|.\footnote{The derivative of uncurried multiplication generalizes to an arbitrary \emph{bilinear} function |f :: a :* b -> c| \citep[Problem 2-12]{Spivak65}:
\begin{code}
der f (a,b) = \ (da,db) -> f (da,b) + f (a,db)
\end{code}
}
To make the linearity more apparent, and to prepare for variations later in this paper, let's now rephrase |der mulC| without using lambda directly.
Just as |Category|, |Cartesian|, |Cocartesian|, |NumCat|, etc generalize operations beyond functions, it will also be handy to generalize scaling as well:
\begin{code}
class ScalarCat k a where
  scale :: a -> (a `k` a)

instance Num a => ScalarCat (->) a where
  scale a = \ da -> a * da
\end{code}
Since uncurried multiplication is bilinear, its partial application as |scale a| (for functions) is linear for all |a|.
Now we can rephrase the product rule in terms of more general, linear language, using the derived |(###)| operation defined in \secref{Derived operations}.
\begin{code}
der mulC (a,b) = scale b ||| scale a
\end{code}

This product rule, along with the linearity of negation and uncurried addition, enables using the same style of derivation as with operations from |Category|, |MonoidalPCat|, |Cartesian|, and |Cocartesian| above.
As usual, specify the |NumCat| instance for differentiable functions by saying that |adf| preserves |NumCat| structure, i.e., |adf negateC == negateC|, |adf addC == addC|, and |adf mulC == mulC|.
Reasoning as before, we get another correct-by-construction instance for differentiable functions:
\begin{code}
instance NumCat D where
  negateC = linearD negateC
  addC  = linearD addC
  mulC  = D (\ (a,b) -> (a * b, scale b ||| scale a))
\end{code}

Similar reasoning applies to other numeric operations, e.g.,
\begin{code}
instance FloatingCat D where
  sinC  = D (\ a -> (sin a, scale (cos a)))
  cosC  = D (\ a -> (cos a, scale (- sin a)))
  expC  = D (\ a -> let e = exp a in (e, scale e))
        ...
\end{code}
In what follows, the |scale| operation will play a more important role than merely tidying definitions.

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
\figone{magSqr}{|magSqr|}}{
\figone{cosSinProd}{|cosSinProd|}}
To see the differentiable versions, interpret these same expressions in the category of differentiable functions (|D| from \secref{Categories}), remove the |D| constructors to reveal the function representation, convert these functions to categorical form as well, and finally interpret the result in the graph category.
The results are rendered in \figreftwo{magSqr-adf}{cosSinProd-adf}.
\figp{
\figone{magSqr-adf}{|adf magSqr|}}{
\figone{cosSinProd-adf}{|adf cosSinProd|}}
Note that the derivatives are (linear) functions, as depicted in boxes.
Also note the sharing of work between the a function's result and its derivative in \figref{cosSinProd-adf}.\notefoot{Introduce the term ``primal'' early on and use it throughout.}

\sectionl{Programming as defining and solving algebra problems}

Stepping back to consider what we've done, a general recipe emerges:\notefoot{Go over the wording of this section to make as clear as I can.}
\begin{itemize}
\item Start with an expensive or even non-computable specification (here involving differentiation).
\item Build the desired result into the representation of a new data type (here as the combination of a function and its derivative).
\item Try to show that conversion from a simpler form (here regular functions) to the new data type---even if not computable---is \emph{compositional} with respect to a well-understood algebraic abstraction (here |Category|).
\item If compositionality fails (as with |der|, unadorned differentiation, in \secref{Sequential composition}), then examine the failure to find an augmented specification, iterating as needed until converging on a representation and corresponding specification that \emph{is} compositional.
\item Set up an algebra problem whose solution will be an instance of the well-understood algebraic abstraction for the chosen representation.
These algebra problems always have a particular stylized form, namely that the operation being solved for is a \emph{homomorphism} for the chosen abstraction (here a category homomorphism, also called a ``functor'').
\item Solve the algebra problem by using the compositionality properties.
\item Rest assured that the solution satisfies the required laws, at least when the new data type is kept abstract, thanks to the homomorphic specification.
\end{itemize}
The result of this recipe is not quite an implementation of our homomorphic specification, which may after all be non-computable.
Rather, it gives a computable alternative that is almost as useful: if the input to the specified conversion is expressed in vocabulary of the chosen algebraic abstraction, then a re-interpretation of that vocabulary in the new data type is the result of the (possibly non-computable) specification.
Furthermore, if we can \emph{automatically} convert conventionally written functional programs into the chosen algebraic vocabulary (as in \citep{Elliott-2017-compiling-to-categories}), then those programs can be re-interpreted to compute the desired specification.

\sectionl{Generalizing automatic differentiation}

\corRefs{compose}{linear} all have the same form: an operation on |D| (differentiable functions) is defined entirely via the same operation on |(:-*)| (linear maps).
Specifically, the composition of differentiable functions relies on the composition of linear maps, and likewise for |(&&&)|, |(###)|, and linear functions.
These corollaries follow closely from \thmRefs{compose}{linear}, which relate derivatives for these operations to the corresponding operations on linear maps.
These properties make for a pleasantly poetic theory, but they also have a powerful, tangible benefit, which is that we can replace linear maps by any of a much broader variety of underlying categories to arrive at a greatly generalized notion of AD.

%format GD = GAD
The generalized AD definitions, shown in \figref{GAD} result from making a few small changes to the non-generalized definitions derived in \secref{Putting the pieces together}:
\begin{itemize}
\item The new category |GD| takes as parameter a category |k| that replaces |(:-*)| in |D|.
\item The |linearD| function to take two arrows, previously identified.\notefoot{Alternatively, posit an embedding function |lin :: (a :-* b) -> (a -> b)|, write \thmRef{linear} as |der (lin f) a = f|, and change to |linearD :: (a :-* b) -> D a b|.
Then retroactively make |linearD| a method of a new class.}
\item The functionality needed of the underlying category becomes explicit.
\end{itemize}

\begin{figure}
\begin{center}
\begin{code}
newtype GD k a b = D (a -> b :* (a `k` b))

linearD :: (a -> b) -> (a `k` b) -> D a b
linearD f f' = D (\ a -> (f a,f'))

instance Category k => Category (GD k) where
  id = linearD id id
  D g . D f = D (\ a -> let { (b,f') = f a ; (c,g') = g b } in (c, g' . f'))

instance MonoidalPCat k => MonoidalPCat (GD k) where
  D f *** D g = D (\ (a,b) -> let { (c,f') = f a ; (d,g') = g b } in ((c,d), f' *** g'))

instance Cartesian k => Cartesian (GD k) where
  exl  = linearD exl  exl
  exr  = linearD exr  exr
  dup  = linearD dup  dup

instance Cocartesian k => Cocartesian (GD k) where
  inl  = linearD inl  inl
  inr  = linearD inr  inr
  jam  = linearD jam  jam

instance ScalarCat k s => NumCat GD s where
  negateC = linearD negateC
  addC  = linearD addC
  mulC  = D (\ (a,b) -> (a * b, scale b ||| scale a))
\end{code}
\caption{Generalized automatic differentiation}
\figlabel{GAD}
\end{center}
\end{figure}

\sectionl{Matrices}

As an aside, let's consider matrices---the representation typically used in linear algebra---and especially the property of rectangularity.
There are three possibilities (with the last two non-exclusive) for a nonempty matrix |W|:
\begin{itemize}
\item |width W == height W == 1|;
\item |W| is the horizontal juxtaposition of two matrices |U| and |V| with |height W == height U == height V|, and |width W = width U + width V|; or
\item |W| is the vertical juxtaposition of two matrices |U| and |V| with |width W == width U == width V|, and |height W = height U + height V|.
\end{itemize}
These three shape constraints establish and preserve rectangularity.

The vocabulary we have needed from generalized linear maps so far is exactly that of |Category|, |Cartesian|, |Cocartesian|, and |ScalarCat|.
Let's now extract just three operation from this vocabulary:
\begin{closerCodePars}
\begin{code}
  scale  :: a -> (a `k` a)

  (|||)  :: (a `k` c) -> (b `k` c) -> ((a :* b) `k` c)

  (&&&)  :: (a `k` c) -> (a `k` d) -> (a `k` (c :* d))
\end{code}
\end{closerCodePars}%
These three operations exactly correspond to the three possibilities above for a nonempty matrix |W|, with the width and height constraints captured neatly by types.
When matrices are used to represent linear maps, the domain and codomain types for the corresponding linear map are determined by the width and height of the matrix, respectively (assuming the convention of matrix on the left multiplied by a column vector on the right), together with the type of the matrix elements.

\mynote{Say something about block matrices and their use in efficient matrix computations.}


\sectionl{Extracting a data representation}

%format R2
%format R3
The generalized form of AD in \secref{Generalizing automatic differentiation} allows for different representations of linear maps (as well as alternatives to linear maps, still to be explored).
One simple choice is to use functions, as in \figreftwo{magSqr-adf}{cosSinProd-adf}.
Although this choice is simple and reliable, sometimes we need a \emph{data} representation, e.g.,
\begin{itemize}
\item Gradient-based optimization (including deep learning) works by searching for local minima in the domain of a differentiable function |f :: a -> s|, where |a| is a vector space over the scalar field |s|.
      Each step in the search is in the direction opposite of the gradient of |f|, which is a vector form of |der f|.
\item Computer graphics shading models rely on normal vectors.
      For surfaces represented in parametric form, i.e., as |f :: R2 -> R3|, normal vectors are calculated from the partial derivatives of |f| as vectors, which are the rows of the $2 \times 3$ Jacobian matrix that represents the derivative of |f| at any given point |p :: R2|.
\end{itemize}

Given a linear map |f' :: U :-* V| represented as a function, it is possible to extract a Jacobian matrix from (including the special case of a gradient vector), by applying |f'| to every vector in a basis of |U|.
A particularly convenient basis is the sequence of column vectors of an identity matrix, where the |ith| such vector has a one in the |ith| position and zeros elsewhere.
If |U| has dimension |n| (e.g., |U = Rm|), this sampling requires |m| passes.
Considering the nature of the sparse vectors used as arguments, each pass likely computes inefficiently.
Alternatively, the computations can be done using a sparse vector representation, but such an implementation involves considerable complexity and poses difficulties for efficient, massively parallel, SIMD implementations, such as graphics processors \needcite.

If |U| has very low dimension, then this method of extracting a Jacobian is tolerably efficient, but as dimension grows, it becomes quite expensive.
In particular, many useful problems involve gradient-based optimization over very high-dimensional spaces, which is the worst case for this technique.

\sectionl{Generalized matrices}

Rather than representing derivatives as functions and then extracting a (Jacobian) matrix, a more conventional alternative is to construct and combine matrices in the first place.
These matrices are usually rectangular arrays, representing |Rm -> Rn|, which interferes with the composability we get from  organizing around binary cartesian products, as in the |Cartesian| and |Cocartesian| categorical interfaces.

There is an especially perspective on linear algebra, known as \emph{free vector spaces}.
Given a scalar field |s|, any free vector space has the form |p -> s| for some |p|.
The size of |p| is the dimension of the vector space.
Scaling a vector |v :: p -> s| or adding two such vectors is defined in the usual was as for functions.
Rather than using functions directly as a representation, one can instead use any representation isomorphic to such a function.
In particular, we can represent vector spaces over a given field as a \emph{representable functor}, i.e., a functor |F| such that |F s =~= p -> s| for some |p| (where ``|=~=|'' denotes isomorphism).\notefoot{Relate this notion of \emph{functor} to the one used for specifying |adf|.}
This method is convenient in a richly typed functional language like Haskell, which comes with libraries of functor-level building blocks.
Four such building blocks are functor product, functor composition, and their corresponding identities, which are the unit functor (containing no elements) and the identity functor (containing one element) \citep{Magalhaes:2010,HaskellWikiGhcGenerics}.
\begin{code}
data     (f  :*:  g)  a = f a :*: g a               -- product
newtype  (g  :.:  f)  a = Comp1 (g (f a)) NOP       -- composition

newtype  U1           a = U1                        -- unit
newtype  Par1         a = Par1 a                    -- identity
\end{code}
Use of these functors gives data representation of functions that saves recomputation over a native function representation, as a form of functional memoization \cite{Hinze00memofunctions}.

One way to relate these representable functors to the types that appear in our categorical operations is to use associated types \needcite, associating a functor representation to various types.
Given a scalar field |s| and type |a| of values, presumably built up from a scalar type |s|, the associated |V s a| is a functor such that |V s a s =~= a|.
In other words, the type |a| is modeled as a structure of |s| values, where the structure is given by the associated functor |V s a|.
A ``generalized matrix'' for the linear map type |a :-* b| is the composition of two functors, an outer functor for |b| and an inner functor for |a|, together containing elements from the underlying scalar field |s|:
\begin{code}
newtype L s a b = L (V s b (V s a s))
\end{code}
For a given type |t|, in addition to the choice of functor |V s t|, there must be functions to convert from |t| to |V s t s| and back:
%format toV = to"_"V
%format unV = un"_"V
%format Type = "\ast"
\begin{code}
class HasV s t where
  type V s t :: Type -> Type -- Free vector space as representable functor
  toV  :: t -> V s t s
  unV  :: V s t s -> t
\end{code}
%format Double = R
Some |HasV| instances (vector representations and conversions) are shown in \figref{HasV instances}.
Note that products are represented as functor products, and uses of existing functors such as length-typed vectors \citep{vector-sized} are represented by functor compositions.
\begin{figure}
\begin{minipage}[b]{0.31\textwidth}
\begin{code}
instance HasV s () where
  type V s () = U1
  toV () = U1
  unV U1 = ()

instance HasV Double Double where
  type V Double Double = Par1
  toV x = Par1 x
  unV (Par1 x) = x
\end{code}
\end{minipage}
\begin{minipage}[b]{0ex}{\rule[2.8ex]{0.5pt}{1.4in}}\end{minipage}
\begin{minipage}[b]{0.45\textwidth}
\mathindent1em
\begin{code}
instance (HasV s a, HasV s b) => HasV s (a :* b) where
  type V s (a :* b) = V s a :*: V s b
  toV (a,b) = toV a :*: toV b
  unV (f :*: g) = (unV f,unV g)

instance (HasV s b, KnownNat n) => HasV s (Vector n b) where
  type V s (Vector n b) = Vector n :.: V s b
  toV bs = Comp1 (fmap toV bs)
  unV (Comp1 vs) = fmap unV vs
\end{code}
\end{minipage}
\caption{Some ``vector'' representations}
\figlabel{HasV instances}
\end{figure}
Finally, one must define the standard functionality for linear maps in the form of instances of the following form, whose details are left as an exercise for the reader:\footnote{In particular, the operations of matrix/vector multiplication (representing linear map application) and matrix/matrix multiplication (representing linear map composition) are easily implemented in terms of standard functional programming maps, zips, and folds.}
\begin{closerCodePars}
\begin{code}
instance Category       (L s)    where ...

instance ProductCat     (L s)    where ...

instance CoproductPCat  (L s)    where ...

instance ScalarCat      (L s) s  where ...
\end{code}
\end{closerCodePars}

\mynote{Mention upcoming categorical generalizations, which rely on \emph{indexed} biproducts.}

\sectionl{Efficiency of composition}

With the function representation of linear maps, composition is simple and efficient, but extracting a matrix can be quite expensive, as described in \secref{Extracting a data representation}.
The generalized matrix representation of \secref{Generalized matrices} eliminates the need for this expensive extraction step but at the cost of more expensive composition operation used throughout.

One particularly important efficiency concern is that of (generalized) matrix multiplication.
Although matrix multiplication is associative (because it correctly implements composition of linear maps represented as matrices), different associations can result in very different computational cost.
The problem of optimally associating a chain of matrix multiplications can be solved via dynamic programming in $O(n^3)$ time \citep[Section 15.2]{CLRS} or in $O(n \log n)$ time with a more subtle algorithm \citep{Hu:Shing:1981}.
Solving this problem requires knowing only the sizes (heights and widths) of the matrices involved, and those sizes depend only on the types involved for the sort of strongly typed linear map representation |L s a b| above.
One can thus choose an optimal association at compile time rather than waiting for run-time and then solving the problem repeatedly.
\mynote{Read, grok, and cite \cite {Naumann2008OptimalJA}.}

Alternatively, for some kinds of problems we might want to choose a particular association for sequential composition.
For instance, gradient-based optimization (including its use in deep learning) uses ``reverse-mode'' automatic differentiation (RAD), which is to say fully left-associated compositions.\notefoot{Is RAD always optimal for gradient problems?}
(Dually, ``foward-mode'' AD fully right-associates.)
Reverse mode (including its specialization, back-propagation) is much more efficient for these problems, but is also typically given much more complicated explanations and implementations, involving mutation, graph construction, and ``tapes'' \needcite.
One the main purposes of this paper is to demonstrate that RAD can be accounted for quite simply, while retaining or improving its efficiency of implementation.

\sectionl{Reverse mode AD}

The AD algorithm derived in \secref{Putting the pieces together} and generalized in \figref{GAD} can be thought of as a family of algorithms.
For fully right-associated compositions, it becomes forward mode AD; for fully left-associated compositions, reverse mode AD; and for all other associations, various mixed modes.

Let's now look at how to separate the associations used in formulating a differentiable function from the associations used to compose its derivatives.
A practical reason for making this separation is that we want to do gradient-based optimization (calling for left association), while modular program organization leads to a mixture of compositions.
Fortunately, a fairly simple technique resolves the tension between program organization and efficient execution.

Given any category |U|, we can represent its morphisms by the intent to left-compose with some to-be-given morphism.
That is, represent |f :: a `k` b| by the function |(\ h -> h . f) :: (b `k` r) -> (a `k` r)|, where |r| is any object in |U|.
The morphism |h| will be a \emph{continuation}, finishing the journey from |f| all the way to the codomain of the overall function being assembled.
Building a category around this idea results in turning \emph{all} patterns of composition into fully left-associated.

%format (rcomp f) = (. SPC f)

First, package up the continuation representation as a transformation from one category |k| to a new category, |Cont k r|:\footnote{Following Haskell's notation for ``sections'', write ``|rcomp f|'' as shorthand for |\ h -> h . f|.}
\begin{code}
newtype Cont k r a b = Cont ((b `k` r) -> (a `k` r))

cont :: Category k => (a `k` b) -> Cont k r a b
cont f = Cont (rcomp f)
\end{code}
As usual, let's derive instances for our new category by homomorphic specification.
To say that |cont| is a functor (|Category| homomorphism), is equivalent to the following two equations:
\begin{closerCodePars}
\begin{code}
cont id == id

cont (g . f) == cont g . cont f
\end{code}
\end{closerCodePars}
Simplify the first homomorphism equation:
\begin{code}
   cont id
==  {- definition of |cont| -}
   Cont (rcomp id)
==  {- definition of right section -}
   Cont (\ h -> h . id)
==  {- category law -}
   Cont (\ h -> h)
==  {- definition of |id| -}
   Cont id
\end{code}
The first homomorphism equation is thus equivalent to |id == Cont id|, which is in solved form.
For the second homomorphism equation, simplify both sides:
\begin{code}
   cont g . cont f
==  {- definition of |cont| -}
   Cont (rcomp g) . Cont (rcomp f)
   
   cont (g . f)
==  {- definition of |cont| -}
   cont (rcomp (g . f))
==  {- definition of right section -}
   cont (\ h -> h . (g . f))
==  {- category law -}
   cont (\ h -> (h . g) . f)
==  {- definition of right section -}
   cont (\ h -> (rcomp f) ((rcomp g) h))
==  {- definition of |(.)| -}
   Cont (rcomp f . rcomp g)
\end{code}
The simplified requirement:
\begin{code}
Cont (rcomp g) . Cont (rcomp f) == Cont ((rcomp f) . (rcomp g))
\end{code}
Generalize to a stronger condition, replacing |(rcomp g)| and |(rcomp f)| with |g| and |f| (appropriately re-typed):
\begin{code}
Cont g . Cont f == Cont (f . g)
\end{code}
This strengthened condition is also in solved form, so we have a sufficient definition:
\begin{code}
instance Category k => Category (Cont k r) where
  id = Cont id
  Cont g . Cont f = Cont (f . g)
\end{code}
Notice the reversal of composition (and, more subtly, of |id|).

Similarly, we can derive a |MonoidalPCat| instance from the specification that |cont| is a monoidal functor (i.e., a |MonoidalPCat| homomorphism):
\begin{code}
cont (f *** g) == cont f *** cont g
\end{code}
Simplify both sides of this property:
%format ha = h"_{"a"}"
%format hb = h"_{"b"}"
\begin{code}
   cont f *** cont g
==  {- definition of |cont| -}
   Cont (rcomp f) *** Cont (rcomp g)

   cont (f *** g)
==  {- definition of |cont| -}
   Cont (rcomp (f *** g))
==  {- definition of right section -}
   Cont (\ h -> h . (f *** g))
==  {- |join . unjoin == id| -}
   Cont (\ h -> join (unjoin h) . (f *** g))
==  {- refactor -}
   Cont (\ h -> let (ha,hb) = unjoin h in join (ha,hb) . (f *** g))
==  {- definition of |join| -}
   Cont (\ h -> let (ha,hb) = unjoin h in (ha ||| hb) . (f *** g))
==  {- |Cocartesian| identity \needcite{} -}
   Cont (\ h -> let (ha,hb) = unjoin h in (ha . f ||| hb . g))
==  {- definition of right section -}
   Cont (\ h -> let (ha,hb) = unjoin h in ((rcomp f) ha ||| (rcomp g) hb))
==  {- definition of |join| -}
   Cont (\ h -> let (ha,hb) = unjoin h in join ((rcomp f) ha , (rcomp g) hb))
==  {- definition of |(***)| -}
   Cont (\ h -> let (ha,hb) = unjoin h in join (((rcomp f) *** (rcomp g)) (ha,hb)))
==  {- eliminate |let| -}
   Cont (\ h -> join (((rcomp f) *** (rcomp g)) (unjoin h)))
==  {- definition of |(.)| -}
   Cont (join . ((rcomp f) *** (rcomp g)) . unjoin)
\end{code}
The crucial trick here was to note that the continuation |h :: (a :* b) `k` r| can be split into two continuations |ha :: a `k` r| and |hb :: b `k` r| thanks to |join|/|unjoin| isomorphism from \secref{Derived operations}.\notefoot{In general, this splitting can lose efficiency, since |ha| and |hb| could duplicate work that was shared in |h|. Return to this concern later.}
Now, strengthen the massaged specification, generalizing from |rcomp f| and |rcomp g| as usual.
The result is in solved form and so translates directly to a implementation:
\begin{code}
instance MonoidalPCat k => MonoidalPCat (Cont k r) where
  Cont f *** Cont g = Cont (join . (f *** g) . unjoin)
\end{code}

Next, derive |ProductCat| and |CoproductPCat| instances from the specification that |cont| is a cartesian functor and a cocartesian functor (i.e., |ProductCat| and |CoproductPCat| homomorphisms), i.e.,\\
{\mathindent2.5em
\begin{minipage}[b]{0.45\textwidth}
\begin{code}
cont exl  == exl
cont exr  == exr
cont dup  == dup
\end{code}
\end{minipage}
\begin{minipage}[b]{0ex}{\rule[3ex]{0.5pt}{0.43in}}\end{minipage}
\begin{minipage}[b]{0.0\textwidth}
\begin{code}
cont inl  == inl
cont inr  == inr
cont jam  == jam
\end{code}
\end{minipage}}\\
Reversing each of these equations gives simple, correct instances:\\
\begin{minipage}[b]{0.45\textwidth}
\begin{code}
instance  ProductCat k =>
          ProductCat (Cont k r) where
  exl  = cont exl
  exr  = cont exr
  dup  = cont dup
\end{code}
\end{minipage}
\begin{minipage}[b]{0ex}{\rule[2.5ex]{0.5pt}{0.8in}}\end{minipage}
\begin{minipage}[b]{0.0\textwidth}
\begin{code}
instance  CoproductCat k =>
          CoproductCat (Cont k r) where
  inl  = cont inl
  inr  = cont inr
  jam  = cont jam
\end{code}
\end{minipage}

While these definitions are correct, they can be made more efficient.
For instance,
\begin{code}
   cont exl
==  {- definition of |cont| -}
   Cont (\ h -> h . exl)
==  {- \secref{Abelian categories} -}
   Cont (\ h -> h ||| zeroC)
==  {- definition of |join| -}
   Cont (\ h -> join (h,zeroC))
==  {- definition of |inl| for functions -}
   Cont (\ h -> join (inl h))
==  {- definition of |(.)| for functions -}
   Cont (join . inl)
\end{code}
Similarly,
\begin{code}
   cont exr == Cont (join . inr)
\end{code}
For |dup :: a `k` (a :* a)|, we'll have |h :: (a :* a) ~> r|, so we can split |h| with |unjoin|:
\begin{code}
   cont dup
==  {- definition of |cont| -}
   Cont (\ h -> h . dup)
==  {- |join . unjoin == id| -}
   Cont (\ h -> join (unjoin h) . dup)
==  {- refactor; definition of |join| -}
   Cont (\ h -> let (ha,hb) = unjoin h in (ha ||| hb) . dup)
==  {- \secref{Abelian categories} -}
   Cont (\ h -> let (ha,hb) = unjoin h in ha `plusC` hb)
==  {- definition of |uncurry| -}
   Cont (\ h -> let (ha,hb) = unjoin h in uncurry plusC (ha,hb))
==  {- eliminate the |let| -}
   Cont (\ h -> uncurry plusC (unjoin h))
==  {- definition of |(.)| on functions -}
   Cont (uncurry plusC . unjoin)
==  {- definition of |jamP| for functions -}
   Cont (jamP . unjoin)
\end{code}

For |CoproductCat|, we reason dually:
\begin{code}
   cont inl
==  {- definition of |inl| -}
   Cont (\ h -> h . inl)
==  {- |join . unjoin == id| -}
   Cont (\ h -> join (unjoin h) . inl)
==  {- definition of |join| -}
   Cont (\ h -> let (ha,hb) = unjoin h in (ha ||| hb) . inl)
==  {- axiom for |(###)|/|inl| -}
   Cont (\ h -> let (ha,hb) = unjoin h in ha)
==  {- definition of |exl| for functions -}
   Cont (\ h -> exl (unjoin h))
==  {- definition of |(.)| for functions -}
   Cont (exl . unjoin)
\end{code}
Similarly,
\begin{code}
   cont inr == Cont (exr . unjoin)
\end{code}
Then
\begin{code}
   cont jam
==  {- definition of |cont| -}
   Cont (\ h -> h . jam)
==  {- A law for |jam| and |(###)| -}
   Cont (\ h -> h . (id ||| id))
==  {- A law for |(.)| and |(###)| -}
   Cont (\ h -> h . id ||| h . id)
==  {- Law for |(.)| and |id| -}
   Cont (\ h -> h ||| h)
==  {- definition of |join| -}
   Cont (\ h -> join (h,h))
==  {- definition of |dup| on functions -}
   Cont (join . dup)
\end{code}

Note the pleasant symmetries in these definitions.
Each |ProductCat| or |CoproductPCat| operation on |Cont k r| is defined via the dual |CoproductPCat| or |ProductCat| operation, together with the |join|/|unjoin| isomorphism.

The final element of our linear vocabulary is scalar multiplication.
From \secref{Numeric operations},
\begin{code}
class ScalarCat k a where
   scale :: a -> (a `k` a)
\end{code}
The |Cont| version:\notefoot{Is there a more general argument to make? I haven't wanted to say that |h| is linear.}
\begin{code}
   cont (scale s)
==  {- definition of |cont| -}
   Cont (\ h -> h . scale s)
==  {- linearity of |h| -}
   Cont (\ h -> scale s . h)
==  {- definition of |scale| for functions/maps -}
   Cont (\ h -> scale s h)
==  {- $\eta$-reduction -}
   Cont (scale s)
\end{code}

\mynote{Mention Cayley's Theorem: that any monoid is equivalent to a monoid of functions under composition.
I think |Cont| is a generalization from |Monoid| to |Category|.}

The instances above for |Cont k r| constitute a simple algorithm for reverse mode automatic differentiation.
\mynote{Contrast with other presentations.}

\mynote{Explain better how |Cont k r| performs full left-association. Also, how to use it by applying to |id|.}

%format adr = adf
\figreftwo{magSqr-adr}{cosSinProd-adr} show the results of reverse mode AD via |Cont| corresponding to \figreftwo{magSqr}{cosSinProd} and \figreftwo{magSqr-adf}{cosSinProd-adf}
\figp{
\figoneW{0.40}{magSqr-adr}{|adr magSqr| via |Cont (-+>) R|}}{
\figoneW{0.58}{cosSinProd-adr}{|adr cosSinProd| via |Cont (-+>) R|}}
The derivatives are represented as (linear) functions again, but reversed (mapping from codomain to domain).

\sectionl{Gradients and duality}

As a special case of reverse mode automatic differentiation, let's consider its use to compute \emph{gradients}, i.e., derivatives of functions with a scalar codomain.
This case is very important local minimization by means of gradient descent, e.g., as used in deep learning \needcite{}.

Given a vector space |A| over a scalar field |s|, the \emph{dual} of |A| is |A :-* s|, i.e., the space of linear maps to the underlying field.
This dual space is also a vector space, and when |A| has finite dimension, it is isomorphic to its dual.
In particular, every linear map in |A :-* s| has the form |dot u| for some |u :: A|, where |dot| is the curried dot product \needcite{}.

The |Cont k r| construction from \secref{Reverse mode AD} works for \emph{any} type/object |r|, so let's take |r| to be the scalar field |s|.
The internal representation of |Cont (:-*) s a b| is |(b :-* s) -> (a :-* s)|, which is isomorphic to |b -> a|.\notefoot{Maybe I don't need this isomorphism, and it suffices to consider those linear maps that do correspond to |dot u| for some |u|.}
Call this representation the \emph{dual} of |k|:
\begin{code}
newtype Dual k a b = Dual (b `k` a)
\end{code}
To construct dual representations of (generalized) linear maps, it suffices to convert from |Cont k s| to |Dual k| by a functor we will now derive.
Composing this new functor with |cont :: (a `k` b) -> Cont k s a b| will give us a functor from |a `k` b| to |Dual k a b|.
The new to-be-derived functor:
\begin{code}
asDual :: Cont k s a b -> Dual k a b
asDual (Cont f) = Dual (onDot f)
\end{code}
where |onDot| uses both halves of the isomorphism between |a :-* s| and |a|:\notefoot{Maybe drop |onDot| in favor of its definition.}
\begin{code}
onDot :: ((b :-* s) -> (a :-* s)) -> (b :-* a)
onDot f = unDot . f . dot

dot    :: u -> (u :-* s)
unDot  :: (u :-* s) -> u
\end{code}
To derive instances for |Dual k|, we'll need some properties.
\begin{lemma} \lemLabel{dot-properties}
The following identities hold:
\begin{enumerate}
\item |dot| and |unDot| are linear. \label{dot-linear}
\item |unjoin . dot = dot *** dot| \label{unjoin-dot}
\item |unDot . join = unDot *** unDot| \label{unDot-join}
\item |dot u ### dot v = dot (u,v)| \label{dot-dot-join}
\item |dot zeroV = zeroC| (zero vector vs zero morphism) \label{dot-zeroV}
\end{enumerate}
\mynote{Proofs, perhaps in an appendix.}
\end{lemma}
\nc\lemDot[1]{\lemRef{dot-properties}, part \ref{#1}}
\nc\lemDotTwo[2]{Lemma \ref{lem:dot-properties}, parts \ref{#1} \& \ref{#2}}

For the |Category| instance, we'll need that |id == asDual id|.
Simplifying the RHS,
\begin{code}
   asDual id
==  {- definition of |id| for |Cont| -}
   asDual (Cont id)
==  {- definition of |asDual| -}
   Dual (unDot . id . dot)
==  {- |Category| law for |id|/|(.)| -}
   Dual (unDot . dot)
==  {- |unDot . dot == id| -}
   Dual id
\end{code}
We also need |asDual (g . f) == asDual g . asDual f|, or (without loss of generality) |asDual (Cont g . Cont f) == asDual (Cont g) . asDual (Cont f)|.
Simplifying both sides,
\begin{code}
   asDual (Cont (f . g))
==  {- definition of |asDual| -}
   Dual (unDot . f . g . dot)
==  {- |dot . unDot == id| -}
   Dual (unDot . f . dot . unDot . g . dot)
==  {- definition of |onDot| -}
   Dual (onDot f . onDot g)

   asDual (Cont g) . asDual (Cont f)
==  {- definition of |asDual| -}
   Dual (onDot g) . asDual (onDot f)
\end{code}
As usual, strengthen this equality by replacing |onDot g| and |onDot f| by re-typed |g| and |f|, and read off a sufficient definition.

For |MonoidalPCat|, the homomorphism condition is |asDual (f *** g) == asDual f *** asDual g|.
Simplify both sides:
\begin{code}
   asDual (Cont f) *** asDual (Cont g)
==  {- definition of |asDual| -}
   Dual (onDot f) *** Dual (onDot g)

   asDual (Cont f *** Cont g)
==  {- definition of |(***)| on |Cont| -}
   asDual (Cont (join . (f *** g) . unjoin))
==  {- definition of |asDual| -}
   Dual (onDot (join . (f *** g) . unjoin))
==  {- definition of |onDot| -}
   Dual (unDot . join . (f *** g) . unjoin . dot)
==  {- \lemDotTwo{unjoin-dot}{unDot-join}  -}
   Dual ((unDot *** unDot) . (f *** g) . (dot *** dot))
==  {- Law about |(***)|/|(.)| -}
   Dual (unDot . f . dot *** unDot . g . unDot)
==  {- definition of |onDot| -}
   Dual (onDot f *** onDot g)
\end{code}
Strengthening from |onDot f| and |onDot g| gives a simple sufficient condition:
\begin{code}
Dual f *** Dual g == Dual (f *** g)
\end{code}

For |ProductCat|, 
\begin{code}
   exl
==  {- specification -}
   asDual exl
==  {- definition of |exl| for |Cont| -}
   asDual (Cont (join . inl))
==  {- definition of |asDual| -}
   Dual (onDot (join . inl))
==  {- definition of |onDot|, and associativity of |(.)| -}
   Dual (unDot . join . inl . dot)
==  {- definition of |(.)| for functions -}
   Dual (\ u -> unDot (join (inl (dot u))))
==  {- definition of |inl| for functions -}
   Dual (\ u -> unDot (join (dot u, zeroC)))
==  {- definition of |join| -}
   Dual (\ u -> unDot (dot u ||| zeroC))
==  {- \lemDot{dot-zeroV} -}
   Dual (\ u -> unDot (dot u ||| dot zeroV))
==  {- \lemDot{dot-dot-join} -}
   Dual (\ u -> unDot (dot (u,zeroV)))
==  {- |unDot . dot == id| -}
   Dual (\ u -> (u,zeroV))
==  {- definition of |inl| for functions -}
   Dual (\ u -> inl u)
==  {- $\eta$-reduction -}
   Dual inl
\end{code}
Similarly, the homomorphism property for |exr| becomes |exr == Dual inr|.
For |dup|,
\begin{code}
   dup
==  {- specification -}
   asDual dup
==  {- definition of |dup| for |Cont| -}
   asDual (Cont (jamP . unjoin))
==  {- definition of |asDual| -}
   Dual (onDot (jamP . unjoin))
==  {- definition of |onDot| -}
   Dual (unDot . jamP . unjoin . dot)
==  {- definition of |(.)| for functions -}
   Dual (\ (u,v) -> unDot (jamP (unjoin (dot (u,v)))))
==  {- \lemDot{unjoin-dot} -}
   Dual (\ (u,v) -> unDot (jamP (dot u, dot v)))
==  {- definition of |jamP| for functions -}
   Dual (\ (u,v) -> unDot (dot u + dot v))
==  {- \lemDot{dot-linear} -}
   Dual (\ (u,v) -> unDot (dot u) + unDot (dot v))
==  {- |unDot . dot == id| -}
   Dual (\ (u,v) -> u + v)
==  {- definition of |jamP| for functions -}
   Dual jamP
\end{code}

The |CoproductPCat| instance comes out similarly:
\begin{code}
   inlP
==  {- specification -}
   asDual inlP
==  {- definition of |inlP| for |Cont| -}
   asDual (Cont (exl . unjoin))
==  {- definition of |asDual| -}
   Dual (onDot (exl . unjoin))
==  {- definition of |onDot| -}
   Dual (unDot . exl . unjoin . dot)
==  {- definition of |(.)| for functions -}
   Dual (\ (u,v) -> unDot (exl (unjoin (dot (u,v)))))
==  {- \lemDot{unjoin-dot} -}
   Dual (\ (u,v) -> unDot (exl (dot u, dot v)))
==  {- definition of |exl| on functions -}
   Dual (\ (u,v) -> unDot (dot u))
==  {- |unDot . dot == id| -}
   Dual (\ (u,v) -> u)
==  {- definition of |exl| for functions -}
   Dual exl

   inrP
==  {- \ldots{} as with |inlP| \ldots -}
   Dual exr

   jam
==  {- specification -}
   asDual jam
==  {- definition of |jam| on |Cont| -}
   asDual (Cont (join . dup))
==  {- definition of |asDual| -}
   Dual (onDot (join . dup))
==  {- definition of |onDot| -}
   Dual (unDot . join . dup . dot)
==  {- definition of |(.)| on functions -}
   Dual (\ u -> unDot (join (dup (dot u))))
==  {- definition of |dup| for functions -}
   Dual (\ u -> unDot (join (dot u, dot u)))
==  {- definition of |join| -}
   Dual (\ u -> unDot (dot u ||| dot u))
==  {- \lemDot{dot-dot-join} -}
   Dual (\ u -> unDot (dot (u,u)))
==  {- |unDot . dot == id| -}
   Dual (\ u -> (u,u))
==  {- definition of |dup| on functions -}
   Dual (\ u -> dup u)
==  {- $\eta$-reduction -}
   Dual dup
\end{code}

Finally, scaling:
\begin{code}
   scale s
==  {- specification -}
   asDual (scale s)
==  {- definition of |scale| for |Cont| -}
   asDual (Cont (scale s))
==  {- definition of |asDual| -}
   Dual (onDot (scale s))
==  {- definition of |onDot| -}
   Dual (unDot . scale s . dot)
==  {- \lemDot{dot-linear} -}
   Dual (scale s . unDot . dot)
==  {- |unDot . dot == id| -}
   Dual (scale s)
\end{code}

These calculations lead to the following elegant instances for |Dual k|:
\begin{code}
instance Category k => Category (Dual k) where
   id = Dual id
   Dual g . Dual f = Dual (f . g)

instance MonoidalPCat k => MonoidalPCat (Dual k) where
   Dual f *** Dual g = Dual (f *** g)

instance ProductCat k => ProductCat (Dual k) where
   exl  = Dual inlP
   exr  = Dual inrP
   dup  = Dual jamP

instance CoproductCat k => CoproductCat (Dual k) where
   inlP  = Dual exl
   inrP  = Dual exr
   jamP  = Dual dup

instance ScalarCat k => ScalarCat (Dual k) where
   scale s = Dual (scale s)
\end{code}
Note that these instances exactly dualize a computation, reversing sequential compositions and swapping corresponding |ProductCat| and |CoproductCat| operations.
Recall from \secref{Matrices}, that |scale| forms $1 \times 1$ matrices, while |(###)| and |(&&&)| correspond to horizontal and vertical juxtaposition, respectively.
Thus, from a matrix perspective, duality is \emph{transposition}, turning an $m \times n$ matrix into an $n \times m$ matrix.
Note, however, that |Dual k| involves no actual matrix computations unless |k| does.
In particular, we can simply use the category of linear functions |(-+>)|.%
\notefoot{I don't think I've defined |a -+> b| yet.}

%% \workingHere

\figreftwo{magSqr-gradr}{cos-2xx-gradr} show the results of reverse mode AD via |Dual|.
Compare \figref{magSqr-gradr} with the same example in \figreftwo{magSqr-adf}{magSqr-adr}.
\figp{
\figoneW{0.40}{magSqr-gradr}{|adr magSqr| via |Dual (-+>)|}}{
\figoneW{0.56}{cos-xpytz-gradr}{|adr (\ ((x,y),z) -> cos (x + y * z)| via |Dual (-+>)|}}

\sectionl{To do}
\begin{itemize}
\item Move the current examples to a new \emph{Examples} section after gradients and duality.
      For each example, show the function, |andDerivF|, |andDerivR|, and |andGradR|.
\item The rest of the talk:
  \begin{itemize}
  \item {Reverse AD examples}
  \item {Incremental evaluation}
  \item {Symbolic vs automatic differentiation}
  \item {Conclusions}
  \end{itemize}
\item Move many proofs to appendices.
\item Maybe remove the |Additive| constraints in |Cocartesian|, along with the |Cocartesian (->)| instance.
\item Indexed biproducts.
\item Relate to
  \begin{itemize}
  \item Cayley's theorem
  \item Categorical pullbacks
  \item \emph{Kan Extensions for Program Optimisation} \citep{Hinze2012KanEF}
  \item Denotational design \citep{Elliott2009-type-class-morphisms-TR}.
        The methodologies are quite similar.
  \item Algebra of programming \citep{BirddeMoor96:Algebra}.
  \item Backprop as functor \citep{Fong2017BackpropAF}.
  \end{itemize}
\item |ConstCat| for |Dual| and for linear arrows in general.
\item What is ``generalized AD''?
      Is it AD at all or something else?
\item Misc:
  \begin{itemize}
  \item Mention graph optimization
  \item Examples with generalized matrices
  \end{itemize}
\end{itemize}

\bibliography{bib}

\end{document}
