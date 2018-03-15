%% -*- latex -*-

%% %let anonymous = True

%% TODO: replace latex if with lhs2tex if

%% %let extended = False

%let icfp = not extended

%% %let draft = True

%let indexed = True

%if icfp

%% \documentclass[acmsmall,screen]{acmart} % ,authorversion=true,

\documentclass[acmsmall=true,authorversion
%if anonymous
,anonymous
%endif
%if icfp
,review
%endif
]{acmart}
%% \settopmatter{printfolios=true,printccs=false,printacmref=false}

\author{Conal Elliott}
\email{conal@@conal.net}
\affiliation{%
  \institution{Target}
  % \city{Sunnyvale}
  % \state{California}
  \country{USA}
}

%if False
\acmJournal{PACMPL}
\acmVolume{1}
\acmNumber{ICFP}
\acmArticle{1}
\acmYear{2018}
\acmMonth{1}
\acmDOI{} % \acmDOI{10.1145/nnnnnnn.nnnnnnn}
\startPage{1}
%endif

\bibliographystyle{ACM-Reference-Format}

%% Copyright information
%% Supplied to authors (based on authors' rights management selection;
%% see authors.acm.org) by publisher for camera-ready submission;
%% use 'none' for review submission.
\setcopyright{none}
%\setcopyright{acmcopyright}
%\setcopyright{acmlicensed}
\setcopyright{rightsretained}
%\copyrightyear{2018}           %% If different from \acmYear

%else

%% While editing/previewing, use 12pt and tiny margin.
\documentclass[12]{article}  % fleqn,
\usepackage[margin=0.7in]{geometry}  % 0.12in, 0.9in

\usepackage{natbib}
\bibliographystyle{plainnat}
%if anonymous
\author{Anonymous \\[1.5ex](supplement to ICFP submission)}
%else
\author{Conal Elliott \\[1.5ex]Target\\conal@@conal.net}
%endif

\newcommand\subtitle\footnote

%endif

%% With the article (non-ACM) font, I sometimes need a small negative space
%% before sub- or super-scripts.
%if icfp
%format QQ = "{}"
%else
%format QQ = "\!"
%endif

\usepackage{scalerel}

\usepackage{datetime}
\usdate

\input{macros}
\citestyle{acmauthoryear}

%if not draft
\rnc\indraft[1]{}
%endif

%include polycode.fmt
%include forall.fmt
%include greek.fmt
%include formatting.fmt

\nc\tit{The simple essence of automatic differentiation}
\nc\alttit{Differentiable functional programming made easy}
%if not anonymous
\date{Draft\footnote{In this draft, \mynote{red bracketed text} indicates notes to be addressed and eliminated as writing progresses.}~\ of \today{} \currenttime \out{\\[1ex] For submission to ICFP 2018 ---} \emph{(comments requested)}}
%endif

%if icfp
\title{\tit} \subtitle{\alttit}
%else
\title{\tit \\[1ex] \large \alttit
%if extended
\\[1ex](extended version)
%endif
}
%endif

\newtheorem{theorem}{Theorem}%[section]
\nc\thmLabel[1]{\label{theorem:#1}}
\nc\thmRef[1]{Theorem \ref{theorem:#1}}
\nc\thmRefTwo[2]{Theorems \ref{theorem:#1} and \ref{theorem:#2}}
\nc\thmRefs[2]{Theorems \ref{theorem:#1} through \ref{theorem:#2}}

\newtheorem{corollary}{Corollary}[theorem]
\nc\corLabel[1]{\label{corollary:#1}}
\nc\corRef[1]{Corollary \ref{corollary:#1}}
\nc\corRefTwo[2]{Corollaries \ref{corollary:#1} and \ref{corollary:#2}}
\nc\corRefs[2]{Corollaries \ref{corollary:#1} through \ref{corollary:#2}}

\newtheorem{lemma}[theorem]{Lemma}
\nc\lemLabel[1]{\label{lemma:#1}}
\nc\lemRef[1]{Lemma \ref{lemma:#1}}
\nc\lemRefTwo[2]{Lemma \ref{lemma:#1} and \ref{lemma:#2}}
\nc\lemRefs[2]{Lemma \ref{lemma:#1} through \ref{lemma:#2}}

\nc\proofLabel[1]{\label{proof:#1}}
%if icfp
\nc\provedIn[1]{\textnormal{See proof \citep[Appendix]{Elliott-2018-ad-extended-anon}}}
%else
\nc\proofRef[1]{Appendix \ref{proof:#1}}
\nc\provedIn[1]{\textnormal{Proved in \proofRef{#1}}}
%endif


\setlength{\blanklineskip}{2ex}
\setlength\mathindent{3ex}

%% Needs a "%"after \end{closerCodePars} to avoid a blank space. Fixable?
\newenvironment{closerCodePars}{\setlength{\blanklineskip}{1.3ex}}{}

%format ith = i"th"
%format ith = i"^{\text{th}}"

\begin{document}

%if not icfp
\maketitle
%endif

\begin{abstract}

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
The dualized variant is suitable for gradient-based optimization and is particularly compelling in simplicity and efficiency, requiring no matrix-level representations or computations.

\end{abstract}

%if icfp
\maketitle
%endif

%format Type = "\ast"
%format Vec (s) = V"\!_{"s"}"
%format (HasV (s)) = HasV "\!_{"s"}"

\sectionl{Introduction}

Accurate, efficient, and reliable computation of derivatives has become increasingly important over the last several years, thanks in large part to the successful use of \emph{backpropagation} in machine learning, including multi-layer neural networks, also known as ``deep learning'' \citep{LecunBengioHinton2015DLNature,Goodfellow2016DL}.
Backpropagation is a specialization and independent invention of the \emph{reverse mode} of automatic differentiation (AD) and is used to tune a parametric model to closely match observed data, using the \emph{gradient descent} (or \emph{stochastic} gradient descent) optimization algorithm.
Machine learning and other gradient-based optimization problems typically rely on derivatives of functions with very high dimensional domains\out{---often in the hundreds of millions \citep{LecunBengioHinton2015DLNature}---} and a scalar codomain---exactly the conditions under which reverse-mode AD is much more efficient than forward-mode AD (by a factor proportional to the domain dimension).
Unfortunately, while forward-mode AD (FAD) is easily understood and implemented\needcite, reverse-mode AD (RAD) and backpropagation have had much more complicated explanations and implementations, involving mutation, graph construction and traversal, and ``tapes'' (sequences of reified, interpretable assignments, also called ``traces'' or ``Wengert lists'')\needcite.
The use of mutation, while motivated by efficiency concerns, makes parallel execution difficult and so undermines efficiency as well.
Construction and interpretation (or compilation) of graphs and tapes also adds execution overhead.
The importance of the RAD algorithm makes its current complicated and bulky implementations especially problematic.
The increasingly large machine learning (and other optimization) problems being solved with RAD (usually via backpropagation) suggest the need to find more streamlined, efficient implementations, especially with the massive hardware parallelism now readily and inexpensively available in the form of graphics processors (GPUs) and FPGAs.

Another difficulty in the practical application of AD in machine learning (ML) comes from the nature of many currently popular ML frameworks, including\out{ Theano \citep{Bergstra10theano}\notefoot{Theano doesn't seem to expose graphs.},} Caffee \citep{Jia2014Caffe}, TensorFlow \citep{Abadi2016TensorFlow}, and Keras \citep{Chollet2016KerasResources}.
These frameworks are designed around the notion of a ``graph'' (or ``network'') of interconnected nodes, each of which represents a mathematical operation---a sort of data flow graph.
Application programs construct these graphs \emph{explicitly}, creating nodes and connecting them to other nodes.
After construction, the graphs must then be processed into a representation that is more efficient to train and to evaluate.
These graphs are essentially mathematical expressions with sharing, hence directed acyclic graphs (DAGs).
This paradigm of graph construction, compilation, and execution bears a striking resemblance to what programmers and compilers do all the time:
\begin{itemize}
\item Programs are written by a human.
\item The compiler or interpreter front-end parses the program into a DAG representation.
\item The compiler back-end transforms the DAGs into a form efficient for execution.
\item A human runs the result of compilation.
\end{itemize}
When using a typical ML framework, programmers experience this sequence of steps \emph{at two levels}: working with their code \emph{and} with the graphs that their code generates.
Both levels have notions of operations, variables, information flow, values, types, and parametrization.
Both have execution models that must be understood.

\mynote{Maybe relate traditional, graph-centered ML frameworks to deep DSELs.}

A much simpler and cleaner foundation for ML would be to have just the programming language, omitting the graphs/networks altogether.
Since ML is about (mathematical) functions, one would want to choose a programming language that supports functions well, i.e., a functional language, or at least a language with strong functional features.
One might call this alternative ``differentiable functional programming''.
In this paradigm, programmers directly define their functions of interest, using the standard tools of functional programming, with the addition of a differentiation operator (a typed higher-order function, though partial since not all computable functions are differentiable).
Assuming a \emph{purely} functional language or language subset (with simple and precise mathematical denotation), the meaning of differentiation is exactly as defined in traditional calculus.

How can we realize this vision of differentiable functional programming?
One way is to create new languages, but doing so requires enormous effort to define and implement efficiently, and perhaps still more effort to evangelize.
Alternatively, we might choose a suitable purely functional language like Haskell and then add differentiation.
The present paper embodies the latter choice, augmenting the popular Haskell compiler GHC with a plugin that converts standard Haskell code into categorical form to be instantiated in any of a variety of categories, including differentiable functions \citep{Elliott-2017-compiling-to-categories}.

This paper makes the following specific contributions:
\begin{itemize}
\item
  Beginning with a simple category of derivative-augmented functions, specify AD simply and precisely by requiring this augmentation (relative to regular functions) to be homomorphic with respect to a collection of standard categorical abstractions and primitive mathematical operations.
\item
  Calculate a correct-by-construction AD implementation from the homomorphic specification.
\item
  Generalizing AD by replacing linear maps (general derivative values) with an arbitrary cartesian category \citep{Elliott-2017-compiling-to-categories}, define several AD variations, all stemming from different representations of linear maps: functions (satisfying linearity), ``generalized matrices'' (composed representable functors), continuation-based transformations of any linear map representation, and dualized version of any linear map representation.
  The latter two variations yield correct-by-construction implementations of reverse-mode AD that are much simpler than previously known and are composed from generally useful components.
  The choice of dualized linear functions for gradient computations is particularly compelling in simplicity and efficiency.
  It requires no matrix-level representations or computation and is suitable for\out{ scalar codomains, as in the case of} gradient-based optimization, e.g., for machine learning.
  Closely related techniques yield forward-mode AD.
\end{itemize}

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

\nc\A{\mathbf A}
\nc\B{\mathbf B}
So far, we've seen that the derivative of a function could be a single number (for |R -> R|), or a vector (for |R -> Rn|), or a matrix (for |Rm -> Rn|).
Moreover, each of these situations has an accompanying chain rule, which says how to differentiate the composition of two functions.
Where the scalar chain rule involves multiplying two scalar derivatives, the vector chain rule involves ``multiplying'' two \emph{matrices} $\A$ and $\B$ (the Jacobians), defined as follows:
$$ (\A \cdot \B)_{ij} = \sum_{k=1}^m \A_{ik} \cdot \B_{kj} $$
Since one can think of scalars as a special case of vectors, and scalar multiplication as a special case of matrix multiplication, perhaps we've reached the needed generality.
When we turn our attention to higher derivatives (which are derivatives of derivatives), however, the situation gets more complicated, and we need yet higher-dimensional representations, with correspondingly more complex chain rules.

Fortunately, there is a single, elegant generalization of differentiation with a correspondingly simple chain rule.
First, reword Definition \ref{eq:scalar-deriv} above as follows:\footnote{For clarity, throughout this paper we will use ``|A = B|'' to mean ``|A| is defined as |B|'' and ``|==|'' to mean (more broadly) that ``|A| is equal to |B|''. The former introduces |A|, while the latter asserts that a well-defined statement of equality is in fact true.}
$$ |lim(eps -> 0)(frac(f (x+eps) - f x) eps) - f' x == 0| $$
Equivalently,
$$ |lim(eps -> 0)(frac(f (x+eps) - (f x + eps *^ f' x)) eps) == 0| $$
Notice that |f' x| is used to linearly transform |eps|.
Next, generalize this condition to say that |f' x| is a \emph{linear map} such that
$$|lim(eps -> 0)(frac(norm (f (x+eps) - (f x + f' x eps)))(norm eps)) == 0| .$$
In other words, |f' x| is a \emph{local linear approximation} of |f| at |x|.
When an |f' x| satisfying this condition exists, it is indeed unique \citep[chapter 2]{Spivak65}

The derivative of a function |f :: a -> b| at some value in |a| is thus not a number, vector, matrix, or higher-dimensional variant, but rather a \emph{linear map} (also called ``linear transformation'') from |a| to |b|, which we will write as ``|a :-* b|''.
The numbers, vectors, matrices, etc mentioned above are all different \emph{representations} of linear maps; and the various forms of ``multiplication'' appearing in their associated chain rules are all implementations of linear map \emph{composition} for those representations.
Here, |a| and |b| must be vector spaces that share a common underlying field.
Written as a Haskell-style type signature (but omitting vector space constraints),
%% %format der = "\mathop{\mathcal{D}}"
%% %format der = "\raisebox{0pt}{$\mathcal{D}$}"
%format der = "\mathcal{D}"
\begin{code}
der :: (a -> b) -> (a -> (a :-* b))
\end{code}
%format der2 = der "^2"
From the type of |der|, it follows that differentiating twice has the following type:\footnote{As with ``|->|'', we will take ``|:-*|'' to associate rightward, so |u :-* v :-* w| is equivalent to |u :-* (v :-* w)|.}
\begin{code}
der2 = der . der :: NOP (a -> b) -> (a -> (a :-* a :-* b))
\end{code}

The type |a :-* a :-* b| is a linear map that yields a linear map, which is the curried form of a \emph{bilinear} map.
Likewise, differentiating $k$ times yields a $k$-linear map curried $k-1$ times.
In particular, the \emph{Hessian} matrix $H$ corresponds to the second derivative of a function |f :: Rm -> R|, having $m$ rows and $m$ columns (and satisfying the symmetry condition $H_{i,j} \equiv H_{j,i}$).

%if False
\emph{A comment on type safety:}
Considering the shape of the matrix |H|, it would be easy to mistakenly treat it as representing the first derivative of some other function |g :: Rm -> Rm|.
Doing so would be unsafe, however, since second derivatives are (curried) bilinear maps, not linear maps.
By providing an explicit abstract type for linear maps rather than using a bare matrix representation, such unsafe uses become type errors, easily caught at compile-time.
\mynote{Hm. I guess one could say that |H| really does represent a first derivative, namely of |f'| itself considered as a vector.
However, |f'| is a covector, not a vector.
Noodle more on this explanation, and maybe remove it.}
%endif

\sectionl{Rules for differentiation}

\subsectionl{Sequential composition}

With the shift to linear maps, there is one general chain rule, having a lovely form, namely that the derivative of a composition is a \emph{composition} of the derivatives \cite[Theorem 2-2]{Spivak65}:
\begin{theorem}[compose/``chain'' rule] \thmLabel{compose}
$$|der (g . f) a == der g (f a) . der f a|$$
\end{theorem}
If |f :: a -> b| and |g :: b -> c|\out{, and |a :: a|}, then |der f a :: a :-* b|, and |der g (f a) :: b :-* c|, so both sides of this equation have type |a :-* c|.\footnote{I adopt the common, if sometimes confusing, Haskell convention of sharing names between type and value variables, e.g., with |a| (a value variable) having type |a| (a type variable).
Haskell value and type variable names live in different name spaces and are distinguished by syntactic context.}

Strictly speaking, \thmRef{compose} is not a compositional recipe for differentiating sequential compositions, i.e., it is \emph{not} the case |der (g . f)| can be constructed solely from |der g| and |der f|.
Instead, it also needs |f| itself.
Fortunately, there is a simple way to restore compositionality.
Instead of constructing just the derivative of a function |f|, suppose we \emph{augment} |f| with its derivative:

%format ad = der QQ "^+\!"
%format ad0 = der QQ"_{\scriptscriptstyle 0}\!\!^+\!"

\begin{code}
ad0 :: (a -> b) -> ((a -> b) :* (a -> (a :-* b)))   -- first guess
ad0 f = (f, der f)
\end{code}
As desired, this altered specification is compositional:
\begin{code}
    ad0 (g . f)
==  (g . f, der (g . f))                   -- definition of |ad0|
==  (g . f, \ a -> der g (f a) . der f a)  -- \thmRef{compose}
\end{code}

Note that |ad0 (g . f)| is assembled entirely from components of |ad0 g| and |ad0 f|, which is to say from |g|, |der g|, |f|, and |der f|.
Writing out |g . f| as |\ a -> g (f a)| underscores that the two parts of |ad0 (g . f)| when applied to |a| both involve |f a|.
Computing these parts independently thus requires redundant work.
Moreover, the chain rule itself requires applying a function and its derivative (namely |f| and |der f|) to the same |a|.
Since the chain rule gets applied recursively to nested compositions, this redundant work multiplies greatly, resulting in an impractically expensive algorithm.

Fortunately, this efficiency problem is easily fixed.
Instead of pairing |f| and |der f|, \emph{combine} them\out{ into a single function}:\footnote{The precedence of ``|:*|'' is tighter than that of ``|->|'' and ``|:-*|'', so |a -> b :* (a :-* b)| is equivalent to |a -> (b :* (a :-* b))|.}
\begin{code}
ad :: (a -> b) -> (a -> b :* (a :-* b))   -- better!
ad f a = (f a, der f a)
\end{code}
Combining |f| and |der f| into a single function in this way enables us to eliminate the redundant computation of |f a| in |ad (g . f) a|, as follows:
\begin{corollary}[\provedIn{corollary:compose}]\corLabel{compose}
|ad| is (efficiently) compositional with respect to |(.)|. Specifically,
\begin{code}
ad (g . f) a == let { (b,f') = ad f a ; (c,g') = ad g b } in (c, g' . f')
\end{code}
\end{corollary}

\subsectionl{Parallel composition}

The chain rule, telling how to differentiate sequential compositions, gets a lot of attention in calculus classes and in automatic and symbolic differentiation.\out{\notefoot{To do: introduce AD and SD early.}}
There are other important ways to combine functions, however, and examining them yields additional helpful tools.
Another operation (pronounced ``cross'') combines two functions in \emph{parallel} \citep{Gibbons2002Calculating}:\footnote{By ``parallel'', I simply mean without data dependencies. Operationally, the two functions can be applied simultaneously or not.}
\begin{code}
(***) :: (a -> c) -> (b -> d) -> (a :* b -> c :* d)
f *** g = \ (a,b) -> (f a, g b)
\end{code}

While the derivative of the sequential composition is a sequential composition of derivatives, the derivative of a parallel composition is a parallel composition of the derivatives \citep[variant of Theorem 2-3 (3)]{Spivak65}:\notefoot{Is there a name for this rule? I've never seen it mentioned.}
\begin{theorem}[cross rule] \thmLabel{cross}
$$|der (f *** g) (a,b) == der f a *** der g b|$$
\end{theorem}
If |f :: a -> c| and |g :: b -> d|, then |der f a :: a :-* c| and |der g b :: b :-* d|, so both sides of this equation have type |a :* b :-* c :* d|.

\thmRef{cross} gives us what we need to construct |ad (f *** g)| compositionally:
\begin{corollary}[\provedIn{corollary:cross}] \corLabel{cross}
|ad| is compositional with respect to |(***)|. Specifically,
$$|ad (f *** g) (a,b) == let { (c,f') = ad f a ; (d,g') = ad g b } in ((c,d), f' *** g')|$$
\end{corollary}

An important point left implicit in the discussion above is that sequential and parallel composition preserve linearity.
This property is what makes it meaningful to use these forms to combine derivatives, i.e., linear maps, as we've done above.

\subsectionl{Linear functions}

A function |f :: u -> v| is said to be \emph{linear} when |f| distributes over (preserves the structure of) vector addition and scalar multiplication, i.e.,
\begin{code}
f (a + a')  == f a + f a'
f (s *^ a)  == s *^ f a
\end{code}
for all |a,a' :: u| and |s| taken from the scalar field underlying |u| and |v|.

In addition to \thmRefTwo{compose}{cross}, we will want one more broadly useful rule, namely that \emph{the derivative of every linear function is itself, everywhere} \citep[Theorem 2-3 (2)]{Spivak65}:
\begin{theorem}[linear rule] \thmLabel{linear}
For all linear functions |f|, |der f a == f|.
\end{theorem}
This statement may sound surprising at first, but less so when we recall that the |der f a| is a local linear approximation of |f| at |a|, so we're simply saying that linear functions are their own perfect linear approximations.

For example, consider the (linear) function |id = \ a -> a|.
The linearity rule says that |der id a == id|.
When expressed in terms of typical \emph{representations} of linear maps, this property may be expressed as saying that |der id a| is the number one or as an identity matrix (with ones on the diagonal and zeros elsewhere).

%% %format Rmn = R"^{m+n}"

As another example, consider the (linear) function |fst (a,b) = a|, for which the linearity rule says |der fst (a,b) == fst|.
This property, when expressed in terms of typical \emph{representations} of linear maps, would appear as saying that |der fst a| comprises the partial derivatives one and zero if |a, b :: R|.
More generally, if |a :: Rm| and |b :: Rn|, then the Jacobian matrix representation has shape |m :* (m+n)| (i.e., |m| rows and |m + n| columns) and is formed by the horizontal juxtaposition of an |m :* m| identity matrix on the left with an |m :* n| zero matrix on the right.
This |m :* (m+n)| matrix, however, represents |fst :: Rm :* Rn :-* Rm|.
Note how much simpler it is to say |der fst (a,b) == fst|, and with no loss of precision!

Given \thmRef{linear}, we can construct |ad f| for all linear |f|:
\begin{corollary} \corLabel{linear}
For all linear functions |f|, |ad f == \ a -> (f a, f)|.
(Proof: immediate from the |ad| definition and \thmRef{linear}.)
\end{corollary}

\sectionl{Putting the pieces together}

The definition of |ad| is a precise specification; but it is not an implementation, since |der| itself is not computable.
\corRefs{compose}{linear} provide insight into the compositional nature of |ad| in exactly the form we can now assemble into an efficient, correct-by-construction implementation.

Although differentiation is not computable when given just an arbitrary computable function, we can instead build up differentiable functions compositionally, using exactly the forms introduced above, (namely |(.)|, |(***)| and linear functions), together with various non-linear primitives having known derivatives.
Computations expressed in this vocabulary are differentiable by construction thanks to \corRefs{compose}{linear}.
The building blocks above are not just a random assortment, but rather a fundamental language of mathematics, logic, and computation, known as \emph{category theory} \citep{MacLane1998categories,Lawvere:2009:Conceptual,Awodey2006CT}.
While it would be unpleasant to program directly in such an austere language, its foundational nature enables instead an automatic conversion from conventionally written functional programs \citep{Lambek:1980:LambdaToCCC,Lambek:1985:CCC,Elliott-2017-compiling-to-categories}.

%format (arr c) = "\mathbin{\to_{"c"}}"

%format CU = "\mathcal{U}"
%format CV = "\mathcal{V}"
%format CF = "\mathcal{F}"

%format <- = `elem`

\subsectionl{Categories}

The central notion in category theory is that of a \emph{category}, comprising \emph{objects} (generalizing sets or types) and \emph{morphisms} (generalizing functions between sets or types).
For the purpose of this paper, we will take objects to be types in our program, and morphisms to be enhanced functions.
We will introduce morphisms using Haskell-style type signatures, such as ``|f :: a ~> b <- CU|'', where ``|~>|'' refers to the morphisms for a category |CU|, with |a| and |b| being the \emph{domain} and \emph{codomain} objects/types for |f|.
In most cases, we will omit the ``|<- CU|'', where choice of category is (hopefully) clear from context.
Each category |CU| has a distinguished \emph{identity} morphism |id :: a ~> a <- CU| for every object/type |a| in the category.
For any two morphisms |f :: a ~> b <- CU| and |g :: b ~> c <- CU| (note same category and matching types/objects |b|), there is also the composition |g . f :: a ~> c <- CU|.
The category laws state that
(a) |id| is the left and right identity for composition, and (b) composition is associative.
You are probably already familiar with at least one example of a category, namely functions, in which |id| and |(.)| are the identity function and function composition.

%% %format `k` = "\leadsto"
%% %format k = "(\leadsto)"

Although Haskell's type system cannot capture the category laws explicitly, we can express the two required operations as a Haskell type class, along with a familiar instance:\notefoot{To save space, present each category class with the |(->)| to the right, as in \citet{Elliott-2017-compiling-to-categories}.}\notefootsep{}\notefoot{Would the signatures in this paper be easier to read without using infix type variables?
For instance, ``|(.)  :: k b c -> k a b -> k a c|''.}
\\
\begin{minipage}[b]{0.48\textwidth}
\begin{code}
class Category k where
  id   :: a `k` a
  (.)  :: (b `k` c) -> (a `k` b) -> (a `k` c)
\end{code}
\end{minipage}
\begin{minipage}[b]{0ex}{\rule[1ex]{0.5pt}{0.5in}}\end{minipage}
\begin{minipage}[b]{0.45\textwidth} % \mathindent1em
\begin{code}
instance Category (->) where
  id = \ a -> a
  g . f = \ a -> g (f a)
\end{code}
\end{minipage}
\\
Another example is \emph{linear} functions, which we've written ``|a :-* b|'' above.
Still another example is \emph{differentiable} functions, which we can see by noting two facts:
\begin{itemize}
\item The identity function is differentiable, as witnessed by \thmRef{linear} and the linearity of |id|.
\item The composition of differentiable functions is differentiable, as \thmRef{compose} attests.
\end{itemize}
The category laws (identity and associativity) hold, because differentiable functions form a subset of all functions.\footnote{There are many examples of categories besides restricted forms of functions, including relations, logics, partial orders, and even matrices.}

%format --> = "\dashrightarrow"

Each category forms its own world, with morphisms relating objects within that category.
To bridge between these worlds, there are \emph{functors} that connect a category |CU| to a (possibly different) category |CV|.
Such a functor |F| maps objects in |CU| to objects in |CV|, \emph{and} morphisms in |CU| to morphisms in |CV|.
If |f :: u ~> v <- CU| is a morphism, then a \emph{functor} |F| from |CU| to |CV| transforms |f <- CU| to a morphism |F f :: F u --> F v <- CV|, i.e., the domain and codomain of the transformed morphism |F f <- CV| must be the transformed versions of the domain and codomain of |f <- CU|.
In this paper, the categories use types as objects, while the functors map these types to themselves.%
\footnote{In contrast, Haskell's functors stay within the same category and do change types.}
The functor must also preserve ``categorical'' structure:\footnote{Making the categories explicit, |F (id <- CU) == (id <- CV)| and |F (g . f <- CU) == (F g . F f <- CV)|.}
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

To make the new category more explicit, package the result type of |ad| in a new data type:\out{\notefoot{Maybe format |D a b| using an infix operator. Remember that I'll need another for generalized AD (|GD|).}}
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
\end{closerCodePars}%
\begin{code}
linearD :: (a -> b) -> D a b
linearD f = D (\ a -> (f a,f))

instance Category D where
  id == linearD id
  D g . D f == D (\ a -> let { (b,f') = f a ; (c,g') = g b } in (c, g' . f'))
\end{code}
Factoring out |linearD| will also tidy up treatment of other linear functions.

Before we get too pleased with this definition, let's remember that for |D| to be a category requires more than having definitions for |id| and |(.)|.
These definitions must also satisfy the identity and composition laws.
How might we go about proving that they do?
Perhaps the most obvious route is take those laws, substitute our definitions of |id| and |(.)|, and reason equationally toward the desired conclusion.
For instance, let's prove that |id . D f == D f| for all |D f :: D a b|:\footnote{Note that \emph{every} morphism in |D| has the form |D f| for some |f|, so it suffices to consider this form.}\notefootsep{}\notefoot{If pinched for space, remove this proof or move it to \out{\appref{Proofs}}Appendix \ref{sec:Proofs}.}
\begin{code}
    id . D f
==  D (\ b -> (b,id)) . D f                                            -- definition of |id| for |D|
==  D (\ a -> let { (b,f') = f a ; (c,g') = (b,id) } in (c, g' . f'))  -- definition of |(.)| for |D|
==  D (\ a -> let { (b,f') = f a } in (b, id . f'))                    -- substitute |b| for |c| and |id| for |g'|
==  D (\ a -> let { (b,f') = f a } in (b, f'))                         --  |id . f' == f'| (category law)
==  D (\ a -> f a)                                                     -- replace |(b,f')| by its definition
==  D f                                                                -- $\eta$-reduction
\end{code}

We can prove the other required properties similarly.
Fortunately, there is a way to bypass the need for these painstaking proofs, and instead rely \emph{only} on our original specification for this |Category| instance, namely that |ad| is a functor.
To buy this proof convenience, we have to make one concession, namely that we consider only morphisms in |D| that arise from |adf|, i.e., only |hat f :: D a b| such that |hat f = adf f| for some |f :: a -> b|.
We can ensure that indeed only such |hat f| do arise by making |D a b| an \emph{abstract} type, i.e., hiding its data |constructor|.\notefoot{%
For the |Category D| instance given above, the painstaking proofs appear to succeed even without this condition.
Am I missing something?}
The slightly more specialized requirement of our first identity property is then |id . adf f == adf f| for any |f :: a -> b|, which follows easily:
\begin{code}
    id . adf f
==  adf id . adf f  -- functor law for |id| (specification of |adf|)
==  adf (id . f)    -- functor law for |(.)|
==  adf f           -- category law
\end{code}
The other identity law is proved similarly.
Associativity has a similar flavor as well:
\begin{code}
    adf h . (adf g . adf f)
==  adf h . adf (g . f)      -- functor law for |(.)|
==  adf (h . (g . f))        -- functor law for |(.)|
==  adf ((h . g) . f)        -- category law
==  adf (h . g) . adf f      -- functor law for |(.)|
==  (adf h . adf g) . adf f  -- functor law for |(.)|
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
\\
\begin{minipage}[b]{0.59\textwidth}
\begin{code}
class Category k => MonoidalPCat k where
  (***) :: (a `k` c) -> (b `k` d) -> ((Prod k a b) `k` (Prod k c d))
\end{code}
\end{minipage}
\begin{minipage}[b]{0ex}{\rule[1ex]{0.5pt}{0.3in}}\end{minipage}
\begin{minipage}[b]{0.35\textwidth} % \mathindent1em
\begin{code}
instance MonoidalPCat (->) where
  f *** g = \ (a,b) -> (f a, g b)
\end{code}
\end{minipage}
\\
More generally, a category |k| can be monoidal over constructions other than products, but cartesian products (ordered pairs) suffice for this paper.

Two monoidal categories can be related by a \emph{monoidal functor}, which is a functor that also preserves the monoidal structure.
That is, a monoidal functor |F| from monoidal category |CU| to monoidal category |CV|, besides mapping objects and morphisms in |CU| to counterparts in |CV| while preserving the category structure (|id| and |(.)|), \emph{also} preserves the monoidal structure:
\begin{code}
F (f *** g) == F f *** F g
\end{code}
Just as \corRefTwo{compose}{linear} were key to deriving a correct-by-construction |Category| instance from the specification that |adf| is a functor, \corRef{cross} leads to a correct |MonoidalPCat| instance from the specification that |adf| is a monoidal functor, as we'll now see.

Let |F| be |adf| in the reversed form of the monoidal functor equation above, and expand |adf| to its definition as |D . ad|:
\begin{code}
D (ad f) *** D (ad g) == D (ad (f *** g))
\end{code}
By \corRef{cross},
\begin{code}
ad (f *** g) == \ (a,b) -> let { (c,f') = ad f a ; (d,g') = ad g b } in ((c,d), f' *** g')
\end{code}
Now substitute the left-hand side of this equation into the right-hand side of the of the monoidal functor property for |adf|, and \emph{strengthen} the condition by generalizing from |ad f| and |ad g|:
\begin{code}
D f *** D g == D (\ (a,b) -> let { (c,f') = f a ; (d,g') = g b } in ((c,d), f' *** g'))
\end{code}
This strengthened form of the specification can be converted directly to a sufficient definition:
\begin{code}
instance MonoidalPCat D where
  D f *** D g = D (\ (a,b) -> let { (c,f') = f a ; (d,g') = g b } in ((c,d), f' *** g'))

\end{code}

\subsectionl{Cartesian categories}

%format TerminalCat = Terminal
%format CoterminalCat = Initial

%format ProductCat = Cartesian
%format CoproductCat = Cocartesian
%format CoproductPCat = Cocartesian

%format (Coprod (k) a b) = a "+\scrk{-0.4ex}" b
%% %format (Exp (k) a b) = a "\Rightarrow\scrk{-0.2ex}" b

The |MonoidalPCat| abstraction gives a way to combine two functions but not separate them.
It also gives no way to duplicate or discard information.
These additional abilities require another algebraic abstraction, namely that of \emph{cartesian category}, adding operations for projection and duplication:
\\
\begin{minipage}[b]{0.5\textwidth}
\begin{code}
class MonoidalPCat k => ProductCat k where
  exl  :: (Prod k a b) `k` a
  exr  :: (Prod k a b) `k` b
  dup  :: a `k` (Prod k a a)
\end{code}
\end{minipage}
\begin{minipage}[b]{0ex}{\rule[1ex]{0.5pt}{0.67in}}\end{minipage}
\begin{minipage}[b]{0.48\textwidth} \mathindent2em
\begin{code}
instance ProductCat (->) where
  exl  = \ (a,b) -> a
  exr  = \ (a,b) -> b
  dup  = \ a -> (a,a)
\end{code}
\mathindent1em
\end{minipage}

\begin{closerCodePars}
Two cartesian categories can be related by a \emph{cartesian functor}, which is a functor that also preserves the cartesian structure.
That is, a cartesian functor |F| from cartesian category |CU| to cartesian category |CV|, besides mapping objects and morphisms in |CU| to counterparts in |CV| while preserving the category and monoidal structure (|id|, |(.)|, and |(***)|), \emph{also} preserves the cartesian structure:
\begin{code}
F exl  == exl
F exr  == exr
F dup  == dup
\end{code}
Just as \corRefs{compose}{linear} were key to deriving a correct-by-construction |Category| and |MonoidalPCat| instances from the specification that |adf| is a functor and a monoidal functor respectively, \corRef{linear} enables a correct-by-construction |ProductCat| instance from the specification that |adf| is a cartesian functor.
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

%% %format -+> = ->

Cartesian categories have a dual, known as \emph{cocartesian categories}, with each cartesian operation having a mirror image with morphisms reversed (swapping domain and codomain).
In general, each category can have its own notion of coproduct, e.g., sum (disjoint union) types for the |(->)| category.
In this paper, however, all coproducts will be pairs (cartesian \emph{products}, coinciding with the categorical products), i.e., we'll be using biproduct categories \citep{MacedoOliveira2013Typing}:
\\
%format zero = 0
%format ^+^ = +
\begin{minipage}[b]{0.52\textwidth}
\begin{code}
class Monoidal k => CoproductPCat k where
  inl  ::  Additive b => a `k` (Prod k a b)
  inr  ::  Additive a => b `k` (Prod k a b)
  jam  ::  Additive a => (Prod k a a) `k` a
\end{code}
\end{minipage}
\begin{minipage}[b]{0ex}{\rule[1ex]{0.5pt}{0.66in}}\end{minipage}
\begin{minipage}[b]{0.48\textwidth} \mathindent2em
\begin{code}
instance CoproductPCat (->) where
  inl  = \ a -> (a,zero)
  inr  = \ b -> (zero,b)
  jam  = \ (a,b) -> a ^+^ b
\end{code}
\end{minipage}
\\
Unlike |Category| and |ProductCat|, |CoproductPCat| introduces an additivity requirement (having a notion of addition and corresponding zero) to the types involved, in order to have an instance for functions.

Unsurprisingly, there is a notion of \emph{cocartesian functor}, saying that the cocartesian structure is preserved, i.e.,
\begin{closerCodePars}
\begin{code}
F inl  == inl
F inr  == inr
F jam  == jam
\end{code}
\end{closerCodePars}%
From the specification that |adf| is a cocartesian functor and the linearity of |inl|, |inr|, and |jam|, we can derive a correct-by-construction |CoproductPCat| instance for differentiable functions:%
\begin{code}
instance CoproductPCat D where
  inl  = linearD inl
  inr  = linearD inr
  jam  = linearD jam
\end{code}

The translation from Haskell to categorical form \citep{Elliott-2017-compiling-to-categories} does not use this |CoproductPCat| class.
In fact, |Additive| constraints in the |CoproductPCat| class above are only for concise presentation.
The actual implementation omits these constraints in the |CoproductPCat| class definition and has no |CoproductPCat (->)| instance.
Instead, there is a category |(-+>)| of additive functions, defined simply as a |newtype| wrapper around regular functions.
The |CoproductPCat (-+>)| instance is a wrapped version of the |CoproductPCat (->)| instance shown above.
The full |Category| class includes an associated constraint \citep{Bolingbroke2011CK} restricting the types involved in all categorical operations, and defines this constraint to be |Additive| for |(-+>)|.
As a reminder of this distinction, ``|(-+>)|'' is used below where regular functions are used to represent linear (and hence additive) functions.\notefoot{Reconsider this choice even for the conference version of this paper.
See how I'm doing on space.}

\subsectionl{Derived operations}

With |dup|, we can define an alternative to |(***)| that takes two morphisms sharing a domain:
\begin{code}
(&&&) :: Cartesian k => (a `k` c) -> (a `k` d) -> (a `k` (Prod k c d))
f &&& g = (f *** g) . dup
\end{code}
The |(&&&)| operation is particularly useful for translating the $\lambda$-calculus to categorical form \citep[Section 3]{Elliott-2017-compiling-to-categories}.

Dually, |jam| lets us define a second alternative to |(***)| for two morphisms sharing a \emph{codomain}:
\begin{code}
(|||) :: Cocartesian k => (c `k` a) -> (d `k` a) -> ((Prod k c d) `k` a)
f ||| g = jam . (f +++ g)
\end{code}
The |(&&&)| and |(###)| operations\out{ (sometimes called ``fork'' and ``join'')} are invertible in uncurried form \citep{Gibbons2002Calculating}:
\begin{code}
fork    :: Cartesian    k => (a `k` c) :* (a `k` d) -> (a `k` (Prod k c d))
unfork  :: Cartesian    k => (a `k` (Prod k c d)) -> (a `k` c) :* (a `k` d)

join    :: Cocartesian  k => (c `k` a) :* (d `k` a) -> ((Prod k c d) `k` a)
unjoin  :: Cocartesian  k => ((Prod k c d) `k` a) -> (c `k` a) :* (d `k` a)
\end{code}
where
\\[1.5ex]
\begin{minipage}[b]{0.38\textwidth} % \mathindent1em
\begin{code}
fork (f,g) = f &&& g
unfork h = (exl . h, exr . h)
\end{code}
\end{minipage}
\begin{minipage}[b]{0ex}{\rule[1ex]{0.5pt}{0.32in}}\end{minipage}
\begin{minipage}[b]{0.48\textwidth} \mathindent2.5em
\begin{code}
join (f,g) = f ||| g
unjoin h = (h . inl, h . inr)
\end{code}
\end{minipage}

\subsectionl{Numeric operations}

So far, the vocabulary we've considered comprises linear functions and combining forms (|(.)| and |(***)|) that preserve linearity.
To make differentiation interesting, we'll need some non-linear primitives as well.
Let's now add these primitives, while continuing to derive correct implementations from simple, regular specifications in terms of homomorphisms (structure-preserving transformations).
We'll define a collection of interfaces for numeric operations, roughly imitating Haskell's numeric type class hierarchy.

Haskell provides the following basic class:
\begin{code}
class Num a where
  negate :: a -> a
  (+), (*) :: a -> a -> a
  ...
\end{code}
Although this class can accommodate many different types of ``numbers'', the class operations are all committed to being functions.
A more flexible alternative allows operations to be non-functions as well:
\\
%format * = "\cdot"
\begin{minipage}[b]{0.4\textwidth}
\begin{code}
class NumCat k a where
  negateC :: a `k` a
  addC :: (a :* a) `k` a
  mulC :: (a :* a) `k` a
  ...
\end{code}
\end{minipage}
\begin{minipage}[b]{0ex}{\rule[1.4ex]{0.5pt}{0.83in}}\end{minipage}
\begin{minipage}[b]{0.48\textwidth} \mathindent2em
\begin{code}
instance Num a => NumCat (->) a where
  negateC = negate
  addC  = uncurry (+)
  mulC  = uncurry (*)
  ...
\end{code}
\end{minipage}
\\
Besides generalizing from |(->)| to |k|, we've also uncurried the operations, so as to demand less of supporting categories |k|.
There are similar classes for other operations, such as division, powers and roots, and transcendental functions (|sin|, |cos|, |exp| etc).
Note that the |(->)| instance uses the operations from the standard numeric classes (|Num| etc).

Differentiation rules for these operations are part of basic differential calculus:\footnote{The conventional differentiation rules shown here treat derivatives as numbers rather than linear maps.}
\begin{code}
der (negate u) == negate (der u)
der (u  +  v) == der u + der v
der (u * v) == u * der v + v * der u
\end{code}
This conventional form is unnecessarily complex, as each of these rules implicitly involves not just a numeric operation, but also an application of the chain rule.
This form is also imprecise about the nature of |u| and |v|.
If they are functions, then one needs to explain arithmetic on functions; and if they are not functions, then differentiation of non-functions needs explanation.\out{\footnote{Arithmetic on functions is usually defined pointwise, e.g., $u + v = \ t -> u t + v t$.}}

A precise and simpler presentation is to remove the arguments and talk about differentiating the primitive operations in isolation.
Since we have the chain rule to account for context, we do not need to involve it in every numeric operation.
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
Just as |Category|, |MonoidalPCat|, |Cartesian|, |Cocartesian|, |NumCat|, etc generalize operations beyond functions, it will also be handy to generalize scalar multiplication as well:
%format ScalarCat = Scalable
\\
\begin{minipage}[b]{0.35\textwidth} % \mathindent1em
\begin{code}
class ScalarCat k a where
  scale :: a -> (a `k` a)
\end{code}
\end{minipage}
\begin{minipage}[b]{0ex}{\rule[1ex]{0.5pt}{0.32in}}\end{minipage}
\begin{minipage}[b]{0.48\textwidth} \mathindent2em
\begin{code}
instance Num a => ScalarCat (->) a where
  scale a = \ da -> a * da
\end{code}
\end{minipage}
\\
Since uncurried multiplication is bilinear, its partial application as |scale a| (for functions) is linear for all |a|.
Now we can rephrase the product rule in terms of more general, linear language, using the derived |(###)| operation defined in \secref{Derived operations}:
\begin{code}
der mulC (a,b) = scale b ||| scale a
\end{code}

This product rule, along with the linearity of negation and uncurried addition, enables using the same style of derivation as with operations from |Category|, |MonoidalPCat|, |Cartesian|, and |Cocartesian| above.
As usual, specify the |NumCat| instance for differentiable functions by saying that |adf| preserves (|NumCat|) structure, i.e., |adf negateC == negateC|, |adf addC == addC|, and |adf mulC == mulC|.
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
Some remarks:
\begin{itemize}
\item The derivatives are (linear) functions, as depicted in boxes.
\item Work is shared between the a function's result (the ``primal'') and its derivative in \figref{cosSinProd-adf}
\item The graphs shown here are used \emph{solely} for visualizing functions before and after differentiation, playing no role in the programming interface or in the implementation of differentiation.
\end{itemize}

\sectionl{Programming as defining and solving algebra problems}

Stepping back to consider what we've done, a general recipe emerges:\out{\notefoot{Go over the wording of this section to make as clear as I can.}}
\begin{itemize}
\item Start with an expensive or even non-computable specification (here involving differentiation).
\item Build the desired result into the representation of a new data type (here as the combination of a function and its derivative).
\item Try to show that conversion from a simpler form (here regular functions) to the new data type---even if not computable---is \emph{compositional} with respect to a well-understood collection of algebraic abstractions (here |Category| etc).
\item If compositionality fails (as with |der|, unadorned differentiation, in \secref{Sequential composition}), examine the failure to find an augmented specification, iterating as needed until converging on a representation and corresponding specification that \emph{is} compositional.
\item Set up an algebra problem whose solution will be an instance of the well-understood algebraic abstraction for the chosen representation.
These algebra problems always have a particular stylized form, namely that the operation being solved for is a \emph{homomorphism} for the chosen abstractions (here including a category homomorphism, also called a ``functor'').
\item Solve the algebra problem by using the compositionality properties.
\item Rest assured that the solution satisfies the required laws, at least when the new data type is kept abstract, thanks to the homomorphic specification.
\end{itemize}
The result of this recipe is not quite an implementation of our homomorphic specification, which may after all be non-computable.
Rather, it gives a computable alternative that is nearly as useful: if the input to the specified conversion is expressed in vocabulary of the chosen algebraic abstraction, then a re-interpretation of that vocabulary in the new data type is the result of the (possibly non-computable) specification.
Furthermore, if we can \emph{automatically} convert conventionally written functional programs into the chosen algebraic vocabulary (as in \citep{Elliott-2017-compiling-to-categories}), then those programs can be re-interpreted to compute the desired specification.

\mynote{Relate to \citet{BirddeMoor96:Algebra} and maybe \citet{Elliott2009-type-class-morphisms-TR}.}

\sectionl{Generalizing automatic differentiation}

\corRefs{compose}{linear} all have the same form: an operation on |D| (differentiable functions) is defined entirely via the same operation on |(:-*)| (linear maps).
Specifically, the sequential and parallel composition of differentiable functions rely (respectively) on sequential and parallel composition of linear maps, and likewise for each other operation.
These corollaries follow closely from \thmRefs{compose}{linear}, which relate derivatives for these operations to the corresponding operations on linear maps.
These properties make for a pleasantly poetic theory, but they also have a powerful, tangible benefit, which is that we can replace linear maps by any of a much broader variety of underlying categories to arrive at a greatly generalized notion of AD.

%format (GD (k)) = D"_{"k"}"
%% %format GD (k) a b = a "\leadsto_{"k"}" b

A few small changes to the non-generalized definitions derived in \secref{Putting the pieces together} result in the generalized AD definitions shown in \figref{GAD}:
\begin{itemize}
\item The new category takes as parameter a category |k| that replaces |(:-*)| in |D|.
\item The |linearD| function takes two arrows, previously identified.\notefoot{Alternatively, posit an embedding function |lin :: (a :-* b) -> (a -> b)|, write \thmRef{linear} as |der (lin f) a = f|, and change to |linearD :: (a :-* b) -> D a b|.
Then retroactively make |lin| a method of a new class.
Could incremental computation implement |lin|?}
\item The functionality needed of the underlying category becomes explicit.
\end{itemize}
\begin{figure}
\begin{center}
\begin{code}
newtype GD k a b = D (a -> b :* (a `k` b))

linearD :: (a -> b) -> (a `k` b) -> GD k a b
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

instance ScalarCat k s => NumCat (GD k) s where
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
There are three (non-exclusive) possibilities for a nonempty matrix |W|:
\begin{itemize}
\item |width W == height W == 1|;
\item |W| is the horizontal juxtaposition of two matrices |U| and |V| with |height W == height U == height V|, and |width W = width U + width V|; or
\item |W| is the vertical juxtaposition of two matrices |U| and |V| with |width W == width U == width V|, and |height W = height U + height V|.
\end{itemize}
These three shape constraints establish and preserve rectangularity.

The vocabulary we have needed from generalized linear maps so far is exactly that of |Category|, |Cartesian|, |Cocartesian|, and |ScalarCat|.
Let's now extract just three operations from this vocabulary:
\begin{closerCodePars}
\begin{code}
  scale  :: a -> (a `k` a)

  (|||)  :: (a `k` c) -> (b `k` c) -> ((a :* b) `k` c)

  (&&&)  :: (a `k` c) -> (a `k` d) -> (a `k` (c :* d))
\end{code}
\end{closerCodePars}%
These operations exactly correspond to the three possibilities above for a nonempty matrix |W|, with the width and height constraints captured neatly by types.
When matrices are used to represent linear maps, the domain and codomain types for the corresponding linear map are determined by the width and height of the matrix, respectively (assuming the convention of matrix on the left multiplied by a column vector on the right), together with the type of the matrix elements.

\out{\mynote{Maybe say something about block matrices and their use in efficient matrix computations, citing \citet{MacedoOliveira2013Typing}.}}

\sectionl{Extracting a data representation}

%format R2
%format R3
The generalized form of AD in \secref{Generalizing automatic differentiation} allows for different representations of linear maps (as well as alternatives to linear maps).
One simple choice is to use functions, as in \figreftwo{magSqr-adf}{cosSinProd-adf}.
Although this choice is simple and reliable, sometimes we need a \emph{data} representation.
For instance,
\begin{itemize}
\item Gradient-based optimization (including in machine learning) works by searching for local minima in the domain of a differentiable function |f :: a -> s|, where |a| is a vector space over the scalar field |s|.
      Each step in the search is in the direction opposite of the gradient of |f|, which is a vector form of |der f|.
\item Computer graphics shading models rely on normal vectors.
      For surfaces represented in parametric form, i.e., as |f :: R2 -> R3|, normal vectors are calculated from the partial derivatives of |f| as vectors, which are the rows of the $3 \times 2$ Jacobian matrix that represents the derivative of |f| at any given point |p :: R2|.
\end{itemize}

Given a linear map |f' :: U :-* V| represented as a function, it is possible to extract a Jacobian matrix (including the special case of a gradient vector) by applying |f'| to every vector in a basis of |U|.
A particularly convenient basis is the sequence of column vectors of an identity matrix, where the |ith| such vector has a one in the |ith| position and zeros elsewhere.
If |U| has dimension |m| (e.g., |U = Rm|), this sampling requires |m| passes.
\out{Considering the nature of the sparse vectors used as arguments, each pass likely computes inefficiently.
Alternatively, the computations can be done using a sparse vector representation, but such an implementation involves considerable complexity and poses difficulties for efficient, massively parallel, SIMD implementations, such as graphics processors\needcite.}
If |U| has very low dimension, then this method of extracting a Jacobian is tolerably efficient, but as dimension grows, it becomes quite expensive.
In particular, many useful problems involve gradient-based optimization over very high-dimensional spaces, which is the worst case for this technique.

\sectionl{Generalized matrices}

Rather than representing derivatives as functions and then extracting a (Jacobian) matrix, a more conventional alternative is to construct and combine matrices in the first place.
These matrices are usually rectangular arrays, representing |Rm :-* Rn|, which interferes with the composability we get from  organizing around binary cartesian products, as in the |MonoidalPCat|, |Cartesian|, and |Cocartesian| categorical interfaces.

There is, however, an especially convenient perspective on linear algebra, known as \emph{free vector spaces}\needcite\out{FreeVectorSpaceOverASet}.
Given a scalar field |s|, any free vector space has the form |p -> s| for some |p|, where the cardinality of |p| is the dimension of the vector space (and only finitely many |p| values can have non-zero images).
Scaling a vector |v :: p -> s| or adding two such vectors is defined in the usual way for functions.
Rather than using functions directly as a representation, one can instead use any representation isomorphic to such a function.
In particular, we can represent vector spaces over a given field as a \emph{representable functor}, i.e., a functor |F| such that $\exists p \, \forall s$ |F s =~= p -> s| (where ``|=~=|'' denotes isomorphism)\out{\notefoot{Relate this notion of \emph{functor} to the one used for specifying |adf|.}}
This method is convenient in a richly typed functional language like Haskell, which comes with libraries of functor-level building blocks.
Four such building blocks are functor product, functor composition, and their corresponding identities, which are the unit functor (containing no elements) and the identity functor (containing one element) \citep{Magalhaes:2010,HaskellWikiGhcGenerics}.
\begin{code}
data     (f  :*:  g)  a = f a :*: g a               -- product
newtype  (g  :.:  f)  a = Comp1 (g (f a)) NOP       -- composition

newtype  U1           a = U1                        -- unit
newtype  Par1         a = Par1 a                    -- identity
\end{code}
Use of these functors gives data representation of functions that saves recomputation over a native function representation, as a form of functional memoization \cite{Hinze00memofunctions}.
They also provide a composable, type-safe alternative to the more commonly used multi-dimensional arrays (often called ``tensors'') in machine learning libraries

%format toV = to QQ"_"V
%format unV = un QQ"_"V

One way to relate these representable functors to the types that appear in our categorical operations is to use associated types, associating a functor representation to various types \citep{Chakravarty05AssociatedSynonyms}.
Given a scalar field |s| and type |a| of values, presumably built up from a scalar type |s|, the associated |Vec s a| is a functor such that |Vec s a s =~= a|.
In other words, the type |a| is modeled as a structure of |s| values, where the structure is given by the associated functor |Vec s a|.
A ``generalized matrix'' for the linear map type |a :-* b| is the composition of two functors---an outer functor for |b| and an inner functor for |a|, together containing elements from the underlying scalar field |s|:
%format (LC (s)) = L"_{"s"}"
\begin{code}
newtype LC s a b = L (Vec s b (Vec s a s))
\end{code}
For a given type |t|, in addition to the choice of functor |Vec s t|, there must be functions to convert from |t| to |Vec s t s| and back:
\begin{code}
class HasV s t where
  type Vec s t :: Type -> Type
  toV  :: t -> Vec s t s
  unV  :: Vec s t s -> t
\end{code}
%format Double = R
Some instances are shown in \figref{HasV instances}.
Note that products are represented as functor products, and uses of existing functors such as length-typed vectors \citep{vector-sized} are represented by functor compositions.
\begin{figure}
\begin{minipage}[b]{0.28\textwidth} \mathindent0.25em
\begin{code}
instance HasV s () where
  type Vec s () = U1
  toV () = U1
  unV U1 = ()

instance HasV Double Double where
  type Vec Double Double = Par1
  toV x = Par1 x
  unV (Par1 x) = x
\end{code}
\end{minipage}
\begin{minipage}[b]{0ex}{\rule[1.1ex]{0.5pt}{1.5in}}\end{minipage}
\begin{minipage}[b]{0.65\textwidth} \mathindent1em
\begin{code}
instance (HasV s a, HasV s b) => HasV s (a :* b) where
  type Vec s (a :* b) = Vec s a :*: Vec s b
  toV (a,b) = toV a :*: toV b
  unV (f :*: g) = (unV f,unV g)

instance (HasV s b, KnownNat n) => HasV s (Vector n b) where
  type Vec s (Vector n b) = Vector n :.: Vec s b
  toV bs = Comp1 (fmap toV bs)
  unV (Comp1 vs) = fmap unV vs
\end{code}
\end{minipage}
\caption{Some ``vector'' representations}
\figlabel{HasV instances}
\end{figure}
Finally, one must define the standard functionality for linear maps in the form of instances of |Category|, |MonoidalPCat|, |ProductCat|, |CoproductPCat|, and |ScalarCat|.
Details are spelled out elsewhere \citep[Section 7.4 and Appendix A]{Elliott-2017-compiling-to-categories}.\notefoot{Maybe remove most of the detail from this section in favor of this citation.}

\sectionl{Efficiency of composition}

With the function representation of linear maps, composition is simple and efficient, but extracting a matrix can be quite expensive, as described in \secref{Extracting a data representation}.
The generalized matrix representation of \secref{Generalized matrices} eliminates the need for this expensive extraction step but at the cost of more expensive construction operations used throughout.

One particularly important efficiency concern is that of (generalized) matrix multiplication.
Although matrix multiplication is associative (because it correctly implements composition of linear maps represented as matrices), different associations can result in very different computational cost.
The problem of optimally associating a chain of matrix multiplications can be solved via dynamic programming in $O(n^3)$ time \citep[Section 15.2]{CLRS} or in $O(n \log n)$ time with a more subtle algorithm \citep{Hu:Shing:1981}.
Solving this problem requires knowing only the sizes (heights and widths) of the matrices involved, and those sizes depend only on the types involved for a strongly typed linear map representation like |LC s a b| above.
One can thus choose an optimal association at compile time rather than waiting for run-time and then solving the problem repeatedly.
A more sophisticated version of this question is known as the ``optimal Jacobian accumulation'' problem and is NP-complete \citep {Naumann2008OptimalJA}.

Alternatively, for some kinds of problems we might want to choose a particular association for sequential composition.
For instance, gradient-based optimization (including its use in machine learning) uses ``reverse-mode'' automatic differentiation (RAD), which is to say fully left-associated compositions.\notefoot{Is RAD always optimal for gradient problems?}
(Dually, ``foward-mode'' AD fully right-associates.)
Reverse mode (including its specialization, backpropagation) is much more efficient for these problems, but is also typically given much more complicated explanations and implementations, involving mutation, graph construction, and ``tapes''\needcite.
One the main purposes of this paper is to demonstrate that RAD can be implemented quite simply, with performance equal to or better than conventionally complex implementations.

\sectionl{Reverse-mode AD}

The AD algorithm derived in \secref{Putting the pieces together} and generalized in \figref{GAD} can be thought of as a family of algorithms.
For fully right-associated compositions, it becomes forward mode AD; for fully left-associated compositions, reverse-mode AD; and for all other associations, various mixed modes.

Let's now look at how to separate the associations used in formulating a differentiable function from the associations used to compose its derivatives.
A practical reason for making this separation is that we want to do gradient-based optimization (calling for left association), while modular program organization results in a mixture of compositions.
Fortunately, a fairly simple technique removes the tension between efficient execution and modular program organization.

%format (rcomp f) = (. SPC f)

Given any category |U|, we can represent its morphisms by the intent to left-compose with some to-be-given morphism |h|.
That is, represent |f :: a `k` b| by the function |(rcomp f) :: (b `k` r) -> (a `k` r)|, where |r| is any object in |U|.\footnote{Following Haskell notation for \emph{right sections}, ``|rcomp f|'' is shorthand for |\ h -> h . f|.}
The morphism |h| will be a \emph{continuation}, finishing the journey from |f| all the way to the codomain of the overall function being assembled.
Building a category around this idea results in converting \emph{all} composition patterns into fully left-associated form.
This trick is as with conversion to continuation-passing style \citep{Appel2007CC,Kennedy2007ContCont}.
%% Give each computation a continuation saying how the result will ultimately be consumed.
Compositions in the computation become compositions in the continuation\out{ which is post-/left-composed with the main computation}.
For instance, |g . f| with a continuation |k| (i.e., |k . (g . f)|) becomes |f| with a continuation |k . g| (i.e., |(k . g) . f|).
The initial continuation is |id| (because |id . f == f|).

%format ContC (k) = Cont"_{"k"}"
%format (ContC (k) (r)) = Cont"_{"k"}^{"r"}"
Package up the continuation representation as a transformation from category |k| and codomain |r| to a new category, |ContC k r|:
\begin{code}
newtype ContC k r a b = Cont ((b `k` r) -> (a `k` r))

cont :: Category k => (a `k` b) -> ContC k r a b
cont f = Cont (rcomp f)
\end{code}
As usual, we can derive instances for our new category by homomorphic specification:
\begin{theorem}[\provedIn{theorem:cont}]\thmLabel{cont}
Given the definitions in \figref{cont}, |cont| is a homomorphism with respect to each instantiated class.
\end{theorem}
Note the pleasant symmetries in \figref{cont}.
Each |ProductCat| or |CoproductPCat| operation on |ContC k r| is defined via the dual |CoproductPCat| or |ProductCat| operation, together with the |join|/|unjoin| isomorphism.

\begin{figure}
\begin{center}
\begin{code}
newtype ContC k r a b = Cont ((b `k` r) -> (a `k` r))

instance Category k => Category (ContC k r) where
  id = Cont id
  Cont g . Cont f = Cont (f . g)

instance MonoidalPCat k => MonoidalPCat (ContC k r) where
  Cont f *** Cont g = Cont (join . (f *** g) . unjoin)

instance ProductCat k => ProductCat (ContC k r) where
  exl  = Cont (join . inl)
  exr  = Cont (join . inr)
  dup  = Cont (jamP . unjoin)

instance CoproductCat k => CoproductCat (ContC k r) where
  inl  = Cont (exl . unjoin)
  inr  = Cont (exr . unjoin)
  jam  = Cont (join . dup)

instance ScalarCat k a => ScalarCat (ContC k r) a where
   scale s = Cont (scale s)
\end{code}
\caption{Continuation category transformer (specified by functoriality of |cont|)}
\figlabel{cont}
\end{center}
\end{figure}

\out{\mynote{Mention Cayley's Theorem: that any monoid is equivalent to a monoid of functions under composition.
I think |ContC k r| is a generalization from |Monoid| to |Category|.
Also generalizes to the contravariant Yoneda lemma.}}

The instances for |ContC k r| constitute a simple algorithm for reverse-mode AD.
\out{\mynote{Contrast with other presentations.}}
%format adr = adf
\figreftwo{magSqr-adr}{cosSinProd-adr} show the results of |ContC k r| corresponding to \figreftwo{magSqr}{cosSinProd} and \figreftwo{magSqr-adf}{cosSinProd-adf}.
\figp{
\figoneW{0.40}{magSqr-adr}{|magSqr| in |GD (ContC ((-+>)) R)|}}{
\figoneW{0.57}{cosSinProd-adr}{|cosSinProd| in |GD (ContC ((-+>)) R)|}}
The derivatives are represented as (linear) functions again, but reversed (mapping from codomain to domain).

\sectionl{Gradients and duality}

As a special case of reverse-mode automatic differentiation, let's consider its use to compute \emph{gradients}, i.e., derivatives of functions with a scalar codomain (e.g., for optimization).
%% This case is very important for gradient-based optimization.

Given a vector space |A| over a scalar field |s|, the \emph{dual} of |A| is |A :-* s|, i.e., the linear maps to the underlying field \citep[]{Lang1987LinearAlgebra}.\footnote{These linear maps are variously known as ``linear functionals'', ``linear forms'', ``one-forms'', and ``covectors''.}
This dual space is also a vector space, and when |A| has finite dimension, it is isomorphic to its dual.
In particular, every linear map in |A :-* s| has the form |dot u| for some |u :: A|, where |dot| is the curried dot product:\notefoot{Maybe I don't need this isomorphism, and it suffices to consider those linear maps that do correspond to |dot u| for some |u|.}
\begin{code}
class HasDot s u where dot :: u -> (u :-* s)

instance HasDot R R where dot = scale

instance (HasDot s a, HasDot s b) => HasDot s (a :* b) where dot (u,v) = dot u ||| dot v
\end{code}

The |ContC k r| construction from \secref{Reverse-mode AD} works for \emph{any} type/object |r|, so let's take |r| to be the scalar field |s|.
The internal representation of |ContC ((:-*)) s a b| is |(b :-* s) -> (a :-* s)|, which is isomorphic to |b -> a|.
Call this representation the \emph{dual} (or ``opposite'') of |k|:
%% %format Dual = Op
%format (DualC k) = Dual"_{"k"}"
\begin{code}
newtype DualC k a b = Dual (b `k` a)
\end{code}
To construct dual representations of (generalized) linear maps, it suffices to convert from |ContC k s| to |DualC k| by a functor we will now derive.
Composing this new functor with |cont :: (a `k` b) -> ContC k s a b| will give us a functor from |k| to |DualC k|.
The new to-be-derived functor:
\begin{code}
asDual :: (HasDot s a, HasDot s b) => ContC k s a b -> DualC k a b
asDual (Cont f) = Dual (onDot f)
\end{code}
where |onDot| uses both halves of the isomorphism between |a :-* s| and |a|:\out{\notefoot{Maybe drop |onDot| in favor of its definition.}}
%format unDot = dot"^{-1}"
%% %format unDot = dot"^{\scriptscriptstyle -\!1}"
\begin{code}
onDot :: (HasDot s a, HasDot s b) => ((b :-* s) -> (a :-* s)) -> (b :-* a)
onDot f = unDot . f . dot
\end{code}

As usual, we can derive instances for our new category by homomorphic specification:
\begin{theorem}[\provedIn{theorem:asDual}]\thmLabel{asDual}
Given the definitions in \figref{asDual}, |asDual| is a homomorphism with respect to each instantiated class.
\end{theorem}
\begin{figure}
\begin{center}
\begin{code}
instance Category k => Category (DualC k) where
   id = Dual id
   Dual g . Dual f = Dual (f . g)

instance MonoidalPCat k => MonoidalPCat (DualC k) where
   Dual f *** Dual g = Dual (f *** g)

instance ProductCat k => ProductCat (DualC k) where
   exl  = Dual inlP
   exr  = Dual inrP
   dup  = Dual jamP

instance CoproductCat k => CoproductCat (DualC k) where
   inlP  = Dual exl
   inrP  = Dual exr
   jamP  = Dual dup

instance ScalarCat k => ScalarCat (DualC k) where
   scale s = Dual (scale s)
\end{code}
\caption{Dual category transformer (specified by functoriality of |asDual|)}
\figlabel{asDual}
\end{center}
\end{figure}

Note that the instances in \figref{asDual} exactly dualize a computation, reversing sequential compositions and swapping corresponding |ProductCat| and |CoproductCat| operations.
The derived operations are also dualized:
\begin{corollary}[\provedIn{corollary:dual-derived}]\corLabel{dual-derived}
%% |Dual f &&& Dual g == Dual (f ### g)|, and |Dual f ### Dual g == Dual (f &&& g)|.
The |(&&&)| and |(###)| operations mutually dualize:
$$|Dual f &&& Dual g == Dual (f ### g)|$$
$$|Dual f ### Dual g == Dual (f &&& g)|$$
\end{corollary}
Recall from \secref{Matrices}, that |scale| forms $1 \times 1$ matrices, while |(###)| and |(&&&)| correspond to horizontal and vertical juxtaposition, respectively.
Thus, from a matrix perspective, duality is \emph{transposition}, turning an $m \times n$ matrix into an $n \times m$ matrix.
Note, however, that |DualC k| involves no actual matrix computations unless |k| does.
In particular, we can simply use the category of linear functions |(-+>)|.

\figreftwo{magSqr-gradr}{cos-xpytz-gradr} show the results of reverse-mode AD via |GD (DualC (-+>))|.
Compare \figref{magSqr-gradr} with\out{ the same example in} \figreftwo{magSqr-adf}{magSqr-adr}.
%% \figp{
%% \figoneW{0.40}{magSqr-gradr}{|magSqr :: GD (DualC (-+>)) R2 R|}}{
%% \figoneW{0.56}{cos-xpytz-gradr}{|\ ((x,y),z) -> cos (x + y * z) :: GD (DualC (-+>)) R3 R|}}
\figp{
\figoneW{0.40}{magSqr-gradr}{|magSqr| in |GD (DualC (-+>))|}}{
\figoneW{0.56}{cos-xpytz-gradr}{|\ ((x,y),z) -> cos (x + y * z)| in |GD (DualC (-+>))|}}

\sectionl{Forward-mode AD}

It may be interesting to note that we can turn the |Cont| and |Dual| techniques around to yield category transformers that perform full \emph{right-} instead of left-association, converting the general, mode-independent algorithm into forward mode, thus yielding an algorithm preferable for low-dimensional domains (rather than codomains):
%format (BeginC (k) (r)) = Begin"_{"k"}^{"r"}"
%format (lcomp f) = (f SPC .)
\begin{code}
newtype BeginC k r a b = Begin ((r `k` a) -> (r `k` b))

begin :: Category k => (a `k` b) -> BeginC k r a b
begin f = Begin (lcomp f)
\end{code}
As usual, we can derive instances for our new category by homomorphic specification (for |begin|).
Then choose |r| to be the scalar field |s|, as in \secref{Gradients and duality}, noting that |(s :-* a) =~= a|.

%if indexed

\sectionl{Scaling up}

\mynote{Writing in progress. When finished, add to the contributions and maybe abstract.}

So far, we have considered binary products.
Practical applications, including machine learning and other optimization problems, often involve very high-dimensional spaces.
While those spaces can be encoded as nested binary products, doing so would result in unwieldy representations and prohibitively long compilation and execution times.
A practical alternative is to consider $n$-ary products, for which we can again use representable functors.
To construct and consume these ``indexed'' (bi)products, we'll need an indexed variant of |Monoidal|, replacing the two arguments to |(***)| by a (representable) functor |h| of morphisms:
%format IxMonoidalPCat = MonoidalI
%format IxProductCat = CartesianI
%format IxCoproductPCat = CocartesianI
%format crossF = crossI
%format plusPF = crossI
%format exF = exI
%format replF = replI
%format inPF = inI
%format jamPF = jamI
%format forkF = forkI
%format joinPF = joinI
\\
\begin{minipage}[b]{0.49\textwidth} % \mathindent1em
\begin{code}
class Category k => IxMonoidalPCat k h where
  crossF :: h (a `k` b) -> (h a `k` h b)
\end{code}
\end{minipage}
\begin{minipage}[b]{0ex}{\rule[1ex]{0.5pt}{0.3in}}\end{minipage}
\begin{minipage}[b]{0.48\textwidth} % \mathindent1em
\begin{code}
instance Zip h => IxMonoidalPCat (->) h where
  crossF = zipWith id
\end{code}
\end{minipage}
\\
Note that the collected morphisms must all agree in domain and codomain.
While not required for the general categorical notion of products, this restriction accommodates Haskell's type system and seems adequate in practice so far.

%let cartLong = True

Where the |ProductCat| class has two projection methods and a duplication method, the indexed counterpart has a collection of projections and one replication method.\footnote{Assume the following interface for representable functors \citep{Kmett2011Adj}:
\begin{code}
class Distributive f => Representable f where
  type Rep f :: Type
  tabulate  :: (Rep f -> a) -> f a
  index     :: f a -> (Rep f -> a)
\end{code}
}
%if cartLong
\begin{code}
class IxMonoidalPCat k h => IxProductCat k h where
  exF    :: h (h a `k` a)
  replF  :: a `k` h a

instance (Representable h, Zip h, Pointed h) => IxProductCat (->) h where
  exF    = tabulate (flip index)
  replF  = point
\end{code}
%else
\\
\begin{minipage}[b]{0.44\textwidth} % \mathindent1em
\begin{code}
class  IxMonoidalPCat k h =>
       IxProductCat k h where
  exF    :: h (h a `k` a)
  replF  :: a `k` h a
\end{code}
\end{minipage}
\begin{minipage}[b]{0ex}{\rule[1ex]{0.5pt}{0.67in}}\end{minipage}
\begin{minipage}[b]{0.48\textwidth} \mathindent2em
\begin{code}
instance  (Representable h, Zip h, Pointed h) =>
          IxProductCat (->) h where
  exF    = tabulate (flip index)
  replF  = point
\end{code}
\end{minipage}
\\
%endif
Dually, where the |CoproductCat| class has two injection methods and a binary combination method, the indexed counterpart has a collection of injections and one collection-combining method:
%format sumA = sum
%format Summable = Foldable
%if cartLong
\begin{code}
class IxMonoidalPCat k h => IxCoproductPCat k h where
  inPF   :: Additive a => h (a `k` h a)
  jamPF  :: Additive a => h a `k` a

instance Summable h => IxCoproductPCat (->) h where
  inPF      = tabulate (\ i a -> tabulate (\ j -> if i == j then a else zero))
  jamPF     = sumA
\end{code}
%else
\\
\begin{minipage}[b]{0.44\textwidth} % \mathindent1em
\begin{code}
class  IxMonoidalPCat k h =>
       IxCoproductPCat k h where
  inPF   :: Additive a => h (a `k` h a)
  NOP
  jamPF  :: Additive a => h a `k` a
\end{code}
\end{minipage}
\begin{minipage}[b]{0ex}{\rule[1ex]{0.5pt}{0.84in}}\end{minipage}
\begin{minipage}[b]{0.48\textwidth} \mathindent2em
\begin{code}
instance  Summable h =>
          IxCoproductPCat (->) h where
  inPF      =  tabulate $ \ i a -> tabulate $ \ j ->
                 if i == j then a else zero
  jamPF     = sumA
\end{code}
\end{minipage}
%endif

\noindent
There are also indexed variants of the derived operations |(&&&)| and |(###)| from \secref{Derived operations}:
\begin{code}
forkF :: IxProductCat k h => h (a `k` b) -> (a `k` h b)
forkF fs = crossF fs . replF

joinPF :: (IxProductCat k h, Additive a) => h (b `k` a) -> (h b `k` a)
joinPF fs = jamPF . plusPF fs
\end{code}
As usual, we can derive instances by homomorphic specification:
\begin{theorem}[\provedIn{theorem:indexed}]\thmLabel{indexed}
Given the definitions in \figref{indexed}, |adf| is a homomorphism with respect to each instantiated class.
\end{theorem}
%% \figref{indexed} assumes the following definitions:
%% \begin{code}
%% \end{code}
\begin{figure}
\begin{center}
\begin{code}
instance (IxMonoidalPCat k h, Zip h) => IxMonoidalPCat (GD k) h where
  crossF fs = D (second crossF . unzip . crossF (fmap unD fs))

instance (IxProductCat (->) h, IxProductCat k h, Zip h) => IxProductCat (GD k) h where
  exF = linearD exF exF
  replF = zipWith linearD replF replF

instance (IxCoproductPCat k h, Zip h) => IxCoproductPCat (GD k) h where
  inF = zipWith linearD inF inF
  jamPF = linearD jamPF jamPF

-- Auxiliary definitions:

unD :: D a b -> (a -> (b :* (a :-* b)))
unD (D f) = f

unzip :: Functor h => h (a :* b) -> h a :* h b
unzip = fmap exl &&& fmap exr

second :: MonoidalPCat k => (b `k` d) -> ((a :* b) `k` (a :* d))
second g = id *** g

class Zip h where zipWith :: (a -> b -> c) -> h a -> h b -> h c
\end{code}
\caption{AD for indexed biproducts}
\figlabel{indexed}
\end{center}
\end{figure}

These indexed operations are useful in themselves but can be used to derive other operations.
For instance, note the similarity between the types of |crossF| and |fmap|:
\begin{code}
crossF  :: MonoidalPCat  h => h  (a -> b) -> (h a -> h b)
fmap    :: Functor       h =>    (a -> b) -> (h a -> h b)
\end{code}
In fact, the following relationship holds: |fmap == crossF . replF|.
This equation, together with the differentiation rules for |crossF|, |replF|, and |(.)| determines differentiation for |fmap|.

As with \figref{GAD}, the operations defined in \figref{indexed} rely on corresponding operations for the category parameter |k|.
Fortunately, all of those operations are linear or preserve linearity, so they can all be defined on the various representations of derivatives (linear maps) used for AD in this paper, including |ContC k r| and |DualC s|.

\mynote{Discuss other bulk operations? |zipWith|, etc.}

%endif

\sectionl{Related work}

The literature on automatic differentiation is vast, beginning with forward mode \citep{Wengert64} and later reverse mode \citep{Speelpenning:1980:CFP,Rall1981Automatic}, with many developments since \citep{Griewank89onAD,GriewankWalther2008EvalDerivs}.
While most techniques and uses of AD have been directed at imperative programming, there are also variations for functional programs \citep{Karczmarczuk1999FunCoding,Karczmarczuk00adjointcodes,Karczmarczuk2001FunDif,Pearlmutter2007LMH,Pearlmutter2008RAF,Elliott2009-beautiful-differentiation}.
The work in this paper differs in being phrased at the level of functions/morphisms and specified by functoriality without any mention or manipulation of graphs or other syntactic representations.\footnote{Of course the Haskell compiler itself manipulates syntax trees, and the compiler plugin that converts Haskell code to categorical form helps do so, but both are entirely domain-independent, with no knowledge of or special support for differentiation or linear algebra \citep{Elliott-2017-compiling-to-categories}.}
Moreover, the specifications in this paper are simple enough that the various forms of AD presented can be calculated into being (easily)\notefoot{In the conference version, add a citation here to \appref{Proofs} in the extended version.}, and so are correct by construction.

\citet{Pearlmutter2008RAF} make the following observation:
\begin{quotation}\noindent
In this context, reverse-mode AD refers to a particular construction in which the primal data-flow graph is transformed to construct an adjoint graph that computes the sensitivity values. In the adjoint, the direction of the data-flow edges are reversed; addition nodes are replaced by fanout nodes; fanout nodes are replaced by addition nodes; and other nodes are replaced by multiplication by their linearizations. The main constructions of this paper can, in this context, be viewed as a method for constructing scaffolding that supports this adjoint computation.
\end{quotation}
The |Cont| and |Dual| category transformers described in \secreftwo{Reverse-mode AD}{Gradients and duality} (shown in \figreftwo{cont}{asDual}) above explain this ``adjoint graph'' construction without involving graphs.
Data-flow edge reversal corresponds to the reversal of |(.)| (from |Category|), while fanout and addition correspond to |dup| and |jam| (from |ProductCat| and |CoproductPCat| respectively), which are mutually dual.
\citet{Pearlmutter2008RAF} further remark:
\begin{quotation}\noindent
The main technical difficulty to be faced is that reverse-mode AD must convert fanout (multiple use of a variable) in the untransformed code into addition in the reverse phase of the transformed code. We address this by expressing all straight-line code segments in A-normal form, which makes fanout lexically apparent. 
\end{quotation}
The categorical approach in this paper also makes fanout easily apparent, as appearances of |dup|, which are produced during translation from Haskell to categorical form \citep{Elliott-2017-compiling-to-categories} (via |(&&&)| as defined in \secref{Derived operations} above).
This translation is specified and implemented independently of AD.

Closely related to our choice of derivatives as linear maps and their categorical generalizations is the work of \citet{MacedoOliveira2013Typing}, also based on biproducts (though not addressing differentiation).
That work uses natural numbers as categorical objects to capture the dimensions of vectors and matrices, while the current paper uses vector spaces themselves.
The difference is perhaps minor, however, since natural numbers can be thought of as representing finite sets (or corresponding cardinality), which are \emph{bases} of finite-dimensional free vector spaces (as in \secref{Generalized matrices}).
On the other hand, the duality-based gradient algorithm of \secref{Gradients and duality} involves no matrices at all in their traditional representation (arrays of numbers) or generalized sense of \secref{Generalized matrices} (representable functors).

Also sharing a categorical style is the work of \citet{Fong2017BackpropAF}, formulating the backpropropagation algorithm as a functor.
That work, which also uses biproducts (in monoidal but not cartesian form), does not appear to be separable from the application to machine learning, and so would seem to complement this paper.
Backpropagation is a specialization of reverse-mode AD to the context of machine learning, discovered by \citet{Linnainmaa1970MS} and famous by \citet{Rumelhart1988backprop}.

The continuation transformation of \secref{Reverse-mode AD} was inspired by Mitch Wand's work on continuation-based program transformation \citep{Wand80continuation-basedprogram}.
He derived a variety of algorithms based on a single elegant technique: transform a simple recursive program into continuation-passing form, examine the continuations that arise, and find a data (rather than function) representation for them.
Each such representation is a monoid, with its identity and associative operation corresponding to identity and composition of the continuations.
Monoids are categories with only one object, but the technique extends to general categories.
Cayley's theorem for groups (or monoids) captures this same insight and is a corollary (in retrospect) of the Yoneda lemma \cite[Section 2.2]{Riehl2016category}.
The idea of using data representations for functions (``defunctionalization'') was pioneered by \citep{Reynolds72definitionalinterpreters} and further explored by \citep{Danvy2001DW}.

The notion of derivatives as linear maps is the basis of calculus on manifolds \cite{Spivak65} and was also used for AD by \citet{Elliott2009-beautiful-differentiation}.
The latter addressed only forward-mode AD but also included all orders of derivatives.

While there are many forward-mode AD libraries for Haskell, reverse mode (RAD) has been much more difficult.
The most successful implementation appears to be in the \emph{ad} library \citep{Kmett2010AD}.
One RAD implementation in that library uses stable names \citep{PeytonJones99Stretching} and reification \citep{Gill2009TOS} to recover sharing information.
Another maintains a Wengert list (or ``tape'') with the help of a reflection library \citep{Kiselyov2004FPI}.
Both implementations rely on hidden, carefully crafted use of side effects.

Chris \citet{Olah2015NNTFP} shared a vision for ``differentiable functional programming'' similar to that in \secref{Introduction}.
He pointed out that most of the patterns now used in machine learning are already found in functional programming:
\begin{quotation}
These neural network patterns are just higher order functions---that is, functions which take functions as arguments. Things like that have been studied extensively in functional programming. In fact, many of these network patterns correspond to extremely common functions, like fold. The only unusual thing is that, instead of receiving normal functions as arguments, they receive chunks of neural network.
\end{quotation}
The current paper carries this perspective further, suggesting that the that the essence is \emph{differentiable functions}, with ``networks'' (graphs) being an unnecessary (and unwise) operational choice.

This paper builds on a compiler plugin that translates Haskell programs into categorical form to be specialized to various specific categories, including differentiable functions \citep{Elliott-2017-compiling-to-categories}.
(The plugin knows nothing about any specific category, including differentiable functions.)
Another instance of generalized AD given there is automatic incremental evaluation of functional programs.
Relative to that work, the new contributions are the |ContC k r| and |DualC k| categories, their use to succinctly implement reverse-mode AD (by instantiating the generalized differentiation category), the precise specification of instances for |D|, |ContC k r|, and |DualC k| via functoriality, and the calculation of implementations from these specifications.

The implementations in this paper are quite simple and would appear to be efficient as well.
For instance, the duality-based version (\secref{Gradients and duality}) involves no matrices.
Moreover, typical reverse-mode AD (RAD) implementations use mutation to incrementally update derivative contributions from each \emph{use} of a variable or intermediate computation, holding onto all of these accumulators until the very end of the derivative computation.
For this reason, such implementations tend to use considerable memory\needcite.
In contrast, the implementations in this paper (\secreftwo{Reverse-mode AD}{Gradients and duality}) are free of mutation and can easily free (reuse) memory as they go, keeping memory use low.
Given the prominent use of AD, particularly with large data, performance is crucial, so will be worthwhile to examine and compare time and space use in detail.

\mynote{Maybe relate the methodology of \secref{Programming as defining and solving algebra problems} to \citet{BirddeMoor96:Algebra} and \citet{Elliott2009-type-class-morphisms-TR}.}

%if False
\mynote{
Perhaps more about the following:
\begin{itemize}
\item AD work by Barak Pearlmutter and coauthors.
\item \emph{Kan Extensions for Program Optimisation} \citep{Hinze2012KanEF}.
\item \emph{Beautiful differentiation} \citep{Elliott2009-beautiful-differentiation}
\item Denotational design \citep{Elliott2009-type-class-morphisms-TR} (similar methodologies).
\item \emph{Algebra of programming} \citep{BirddeMoor96:Algebra}.
\end{itemize}
}
%endif

\sectionl{Conclusions}

This paper develops a simple, mode-independent algorithm for automatic differentiation (AD) (\secref{Putting the pieces together}), calculated from a simple, natural specification in terms of elementary category theory (functoriality).
It then generalizes the algorithm, replacing linear maps (as derivatives) by an arbitrary biproduct category (\figref{GAD}).
Specializing this general algorithm to two well-known categorical constructions (\figreftwo{cont}{asDual})---also calculated---yields reverse-mode AD (RAD) for general derivatives and for gradients.
These RAD implementations are far simpler than previously known.
In contrast to common approaches to AD, the algorithms described here involve no graphs, tapes, variables, partial derivatives, or mutation, and are usable directly from an existing programming language with no need for new data types or programming style (thanks to use of an AD-agnostic compiler plugin).
Only the simple essence remains.

AD is typically said to be about the chain rule for sequential composition (\thmRef{compose})\needcite.
This paper rounds out the story with two more rules: one for parallel composition and one for all linear operations (\thmRefTwo{cross}{linear}).
Parallel composition is usually left implicit in the special-case treatment of a collection of non-unary operations, such as addition, multiplication, division, and dot products.
With explicit, general support for parallel composition, all operations come to be on equal footing, regardless of arity (as illustrated in \figref{GAD}).

AD is also typically presented in opposition to symbolic differentiation (SD), which is described as applying differentiation rules symbolically.
The main criticism of SD is that it can blow up expressions, resulting a great deal of redundant work\needcite.
Secondly, SD requires implementation of symbolic manipulation as in a computer algebra system.
In contrast, AD is described as a numeric method and can retain the complexity of the original function (within a small constant factor) if carefully implemented, as in reverse mode.
The approach explored in this paper suggests a different perspective: automatic differentiation \emph{is} symbolic differentiation done by a compiler.
Compilers already work symbolically and already take care to preserve sharing in computations.

The specification and implementation of AD in a simple, efficient, and correct-by-construction manner, together with its use from a typed functional language (here via a compiler plugin), make a step toward the vision of differentiable functional programming for machine learning and other uses, as outlined in \secref{Introduction}.
Programmers then define their functions just as they are accustomed, differentiating where desired, without the intrusion of operational notions such as graphs with questionably defined, extralinguistic semantics.

%if False

\sectionl{Acknowledgments}

Putonlalla (IRC)

%endif

%if extended

\appendix

\vspace{2ex}

\mynote{The appendices that follow appear in the extended version of this paper and replaced by citations from the shorter, conference version.}

\sectionl{Terminal and initial objects}

In the biproduct setting of this paper, terminal and initial objects coincide and may be taken to be any singleton type.
We may as well choose the unit type, having exactly one element, representing a canonical zero-dimensional vector space, and written ``|()|'' in Haskell:\footnote{In a more general categorical setting, terminal and initial objects need not coincide and are defined per category.}\footnote{As with |CoproductPCat|,  in the actual implementation, the |CoterminalCat| definition has no |Additive| constraint or |CoterminalCat (->)| instance, and instead has a |CoterminalCat| instance for additive functions.}
\begin{code}
class TerminalCat k    where it ::                 a `k` ()
class CoterminalCat k  where ti :: Additive  a =>  () `k` a

instance TerminalCat (->)    where it = \ _ -> ()
instance CoterminalCat (->)  where ti = \ () -> zero
\end{code}
Differentiation is trivial, since |it| and |ti| on functions are both linear.

\sectionl{Abelian categories}

Another perspective on the operations we've considered is that morphisms sharing any particular domain and codomain (i.e., hom-sets) form an abelian group.
The zero for |a `k` b| results from the composition of initial and terminal morphisms:
\begin{code}
instance (ProductCat k, CoproductPCat k, TerminalCat k, InitialCat k) => Additive (a `k` b) where
  zero = ti . it
  f ^+^ g = jamP . (f *** g) . dup -- | == jamP . (f &&& g) == (f ### g) . dup|.
\end{code}
%% TODO: replace uses of |zero| and |(^+^)| by |zero| and |(+)|
The following identities hold (with ``|.|'' binding more tightly than ``|+|'') \cite[Equations 16 and 17]{MacedoOliveira2013Typing}:
\begin{code}
u &&& v == u . exl ^+^ v . exr
u ||| v == inl . u ^+^ inr . v
\end{code}
In particular,
\begin{code}
u &&& zero == u . exl
zero &&& v == v . exr

u ||| zero == inl . u
zero ||| v == inr . v
\end{code}

\sectionl{Proofs}

\subsection{\corRef{compose}}\proofLabel{corollary:compose}
\begin{code}
    ad (g . f) a
==  ((g . f) a, der (g . f) a)                                 -- definition of |ad|
==  (g (f a), der (g . f) a)                                   -- definition of |(.)|
==  (g (f a), der g (f a) . der f a)                           -- \thmRef{compose}
==  let b = f a in (g b, der g b . der f a)                    -- refactoring to share |f a|
==  let { (b,f') = ad f a ; (c,g') = ad g b } in (c, g' . f')  -- refactoring to show compositionality
\end{code}

\subsection{\corRef{cross}}\proofLabel{corollary:cross}
\begin{code}
    ad (f *** g) (a,b)
==  ((f *** g) (a,b), der (f *** g) (a,b))                                           -- definition of |ad|
==  ((f a, g b), der (f *** g) (a,b))                                                -- definition of |(***)|
==  ((f a, g b), der f a *** der g b)                                                -- \thmRef{cross}
==  let { (c,f') = (f a, der f a) ; (d,g') = (g b, der g b) } in ((c,d), f' *** g')  -- refactoring
==  let { (c,f') = ad f a ; (d,g') = ad g b } in ((c,d), f' *** g')                  -- definition of |ad|
\end{code}

\subsection{\thmRef{cont}}\proofLabel{theorem:cont}

Recall the definition of |cont|:
\begin{code}
cont :: Category k => (a `k` b) -> ContC k r a b
cont f = Cont (rcomp f)
\end{code}
To say that |cont| is a functor (|Category| homomorphism) is equivalent to the following two equalities:
\begin{closerCodePars}
\begin{code}
cont id == id

cont (g . f) == cont g . cont f
\end{code}
\end{closerCodePars}%
Simplify the first homomorphism equation:
\begin{code}
    cont id
==  Cont (rcomp id)       -- definition of |cont|
==  Cont (\ h -> h . id)  -- definition of right section
==  Cont (\ h -> h)       -- category law
==  Cont id               -- definition of |id| for functions
\end{code}
The first homomorphism equation is thus equivalent to |id == Cont id|, which is in solved form.
For the second homomorphism equation, simplify both sides:
\begin{code}
    cont g . cont f
==  Cont (rcomp g) . Cont (rcomp f)        -- definition of |cont|
    
    cont (g . f)
==  cont (rcomp (g . f))                   -- definition of |cont|
==  cont (\ h -> h . (g . f))              -- definition of right section
==  cont (\ h -> (h . g) . f)              -- category law
==  cont (\ h -> (rcomp f) ((rcomp g) h))  -- definition of right section
==  Cont (rcomp f . rcomp g)               -- definition of |(.)|
\end{code}
The simplified requirement:
\begin{code}
Cont (rcomp g) . Cont (rcomp f) == Cont ((rcomp f) . (rcomp g))
\end{code}
Generalize to a stronger condition, replacing |(rcomp g)| and |(rcomp f)| with |g| and |f| (appropriately re-typed):
\begin{code}
Cont g . Cont f == Cont (f . g)
\end{code}
This strengthened condition is also in solved form.
Notice the reversal of composition (and, more subtly, of |id|).

The monoidal functor (i.e., a |MonoidalPCat| homomorphism) property:
\begin{code}
cont (f *** g) == cont f *** cont g
\end{code}
Simplify both sides:
%format ha = h"_{"a"}"
%format hb = h"_{"b"}"
\begin{code}
    cont f *** cont g
==  Cont (rcomp f) *** Cont (rcomp g)                                  -- definition of |cont|
    
    cont (f *** g)
==  Cont (rcomp (f *** g))                                             -- definition of |cont|
==  Cont (\ h -> h . (f *** g))                                        -- definition of right section
==  Cont (\ h -> join (unjoin h) . (f *** g))                          -- |join . unjoin == id|
==  Cont (\ h -> let (ha,hb) = unjoin h in join (ha,hb) . (f *** g))   -- refactor
==  Cont (\ h -> let ... in (ha ||| hb) . (f *** g))                   -- definition of |join|
==  Cont (\ h -> let ... in (ha . f ||| hb . g))                       -- \citep[Section 1.5.2]{Gibbons2002Calculating}
==  Cont (\ h -> let ... in ((rcomp f) ha ||| (rcomp g) hb))           -- definition of right section
==  Cont (\ h -> let ... in join ((rcomp f) ha , (rcomp g) hb))        -- definition of |join|
==  Cont (\ h -> let ... in join (((rcomp f) *** (rcomp g)) (ha,hb)))  -- definition of |(***)|
==  Cont (\ h -> join (((rcomp f) *** (rcomp g)) (unjoin h)))          -- eliminate |let|
==  Cont (join . ((rcomp f) *** (rcomp g)) . unjoin)                   -- definition of |(.)|
\end{code}
The crucial trick here was to note that |h :: (a :* b) `k` r| can be split into two continuations |ha :: a `k` r| and |hb :: b `k` r| thanks to |join|/|unjoin| isomorphism from \secref{Derived operations}.\notefoot{In general, this splitting can lose efficiency, since |ha| and |hb| could duplicate work that was shared in |h|. Investigate this concern.}
Now, strengthen the massaged specification, generalizing from |rcomp f| and |rcomp g| as usual, resulting in a sufficient condition in solved form:
\begin{code}
Cont f *** Cont g == Cont (join . (f *** g) . unjoin)
\end{code}

Next, derive |ProductCat| and |CoproductPCat| instances from the specification that |cont| is a cartesian functor and a cocartesian functor (i.e., |ProductCat| and |CoproductPCat| homomorphisms), i.e.,\\
{\mathindent2.5em
\begin{minipage}[b]{0.30\textwidth}
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
Reversing each of these equations puts them in solved form, so they can be used directly as definitions.
\out{\\
\begin{minipage}[b]{0.45\textwidth}
\begin{code}
instance  ProductCat k =>
          ProductCat (ContC k r) where
  exl  = cont exl
  exr  = cont exr
  dup  = cont dup
\end{code}
\end{minipage}
\begin{minipage}[b]{0ex}{\rule[2.5ex]{0.5pt}{0.8in}}\end{minipage}
\begin{minipage}[b]{0.0\textwidth}
\begin{code}
instance  CoproductCat k =>
          CoproductCat (ContC k r) where
  inl  = cont inl
  inr  = cont inr
  jam  = cont jam
\end{code}
\end{minipage}
}
While these definitions are correct, they can be made more efficient.
For instance,
\begin{code}
    cont exl
==  Cont (\ h -> h . exl)        -- definition of |cont|
==  Cont (\ h -> h ||| zero)     -- \appref{Abelian categories}
==  Cont (\ h -> join (h,zero))  -- definition of |join|
==  Cont (\ h -> join (inl h))   -- definition of |inl| for functions
==  Cont (join . inl)            -- definition of |(.)| for functions
\end{code}
Similarly, |cont exr == Cont (join . inr)|.
For |dup :: a `k` (a :* a)|, we'll have |h :: (a :* a) ~> r|, so we can split |h| with |unjoin|:
\begin{code}
    cont dup
==  Cont (\ h -> h . dup)                                      -- definition of |cont|
==  Cont (\ h -> join (unjoin h) . dup)                        -- |join . unjoin == id|
==  Cont (\ h -> let (ha,hb) = unjoin h in (ha ||| hb) . dup)  -- refactor; definition of |join|
==  Cont (\ h -> let (ha,hb) = unjoin h in ha ^+^ hb)          -- \appref{Abelian categories}
==  Cont (\ h -> let (ha,hb) = unjoin h in jam (ha,hb))        -- definition of |jamP| for functions
==  Cont (\ h -> jam (unjoin h))                               -- eliminate the |let|
==  Cont (jam . unjoin)                                        -- definition of |(.)| on functions
\end{code}

For |CoproductCat|, we reason dually:
\begin{code}
    cont inl
==  Cont (\ h -> h . inl)                                      -- definition of |inl|
==  Cont (\ h -> join (unjoin h) . inl)                        -- |join . unjoin == id|
==  Cont (\ h -> let (ha,hb) = unjoin h in (ha ||| hb) . inl)  -- definition of |join|
==  Cont (\ h -> let (ha,hb) = unjoin h in ha)                 -- \citep[Section 1.5.2]{Gibbons2002Calculating}
==  Cont (\ h -> exl (unjoin h))                               -- definition of |exl| for functions
==  Cont (exl . unjoin)                                        -- definition of |(.)| for functions
\end{code}
Similarly, |cont inr == Cont (exr . unjoin)|.
Next,
\begin{code}
    cont jam
==  Cont (\ h -> h . jam)            -- definition of |cont|
==  Cont (\ h -> h . (id ||| id))    -- a law for |jam| and |(###)|
==  Cont (\ h -> h . id ||| h . id)  -- \citep[Section 1.5.2]{Gibbons2002Calculating}
==  Cont (\ h -> h ||| h)            -- category law
==  Cont (\ h -> join (h,h))         -- definition of |join|
==  Cont (join . dup)                -- definition of |dup| on functions
\end{code}

The final element of our linear vocabulary is scalar multiplication:\notefoot{Is there a more general argument to make? I haven't wanted to say that |h| is linear.}
\begin{code}
    cont (scale s)
==  Cont (\ h -> h . scale s)  -- definition of |cont|
==  Cont (\ h -> scale s . h)  -- linearity of |h|
==  Cont (\ h -> scale s h)    -- definition of |scale| for functions/maps
==  Cont (scale s)             -- $\eta$-reduction
\end{code}
These optimized solved forms match the definitions in \figref{cont}.

\subsection{\thmRef{asDual}}\proofLabel{theorem:asDual}

\nc\lemDot[1]{\lemRef{dot-properties}, part \ref{#1}}
\nc\lemDotTwo[2]{Lemma \ref{lemma:dot-properties}, parts \ref{#1} \& \ref{#2}}

To derive instances for |DualC k|, we'll need some properties.
\begin{lemma} \lemLabel{dot-properties}
The following properties hold:
% https://tex.stackexchange.com/questions/38260/non-italic-text-in-theorems-definitions-examples
\normalfont
\begin{enumerate}
\item |dot| is linear. \label{dot-linear}
\item |unDot| is linear. \label{unDot-linear}
\item |unjoin . dot == dot *** dot| \label{unjoin-dot}
\item |unDot . join == unDot *** unDot| \label{unDot-join}
\item |dot u ### dot v == dot (u,v)| \label{dot-dot-join}
\item |dot zeroV == zero| (zero vector vs zero morphism) \label{dot-zeroV}
\end{enumerate}
\end{lemma}
\emph{Proof:}
\begin{enumerate}
\item Follows from the bilinearity of uncurried dot product:\notefoot{I'm treating linear maps here as functions. Revisit.}
\begin{code}
    dot (u + v)
==  \ w -> dot (u + v) w      -- $\eta$-expansion
==  \ w -> dot u w + dot v w  -- bilinearity of uncurried dot product
==  dot u + dot v             -- definition of |(+)| of functions

    dot (s *^ u)
==  \ w -> dot (s *^ u) w     -- $\eta$-expansion
==  \ w -> s *^ dot u w       -- bilinearity of uncurried dot product
==  s *^ dot u                -- definition of |(*^)| on functions
\end{code}
\item Invertible linear functions have linear inverses. In particular,
\begin{code}
    unDot (u + v)
==  unDot (dot (unDot u) + dot (unDot v))  -- |dot . unDot == id|
==  unDot (dot (unDot u + unDot v))        -- linearity of |dot|
==  unDot u + unDot v                      -- |unDot . dot == id|

    unDot (s *^ u)                         
==  unDot (s *^ dot (unDot u))             -- |dot . unDot == id|
==  unDot (dot (s *^ unDot u))             -- linearity of |dot| 
==  s *^ unDot u                           -- |unDot . dot == id|
\end{code}
\item Noting that the argument of both sides is a pair,
\begin{code}
    unjoin . dot
==  \ (u,v) -> unjoin (dot (u,v))                                      -- $\eta$-expansion
==  \ (u,v) -> (dot (u,v) . inlP, dot (u,v) . inrP)                    -- definition of |unjoin|
==  \ (u,v) -> (\ x -> dot (u,v) (inlP x), \ y -> dot (u,v) (inrP y))  -- def'n of |(.)| for |(->)|
==  \ (u,v) -> (\ x -> dot (u,v) (x,0), \ y -> dot (u,v) (0,y))        -- def'n of |inlP| for |(:-*)|
==  \ (u,v) -> (\ x -> dot u x + dot v 0, \ y -> dot u 0 + dot v y)    -- def'n of |dot| for pairs
==  \ (u,v) -> (\ x -> dot u x, \ y -> dot v y)                        -- linearity of |dot|
==  \ (u,v) -> (dot u, dot v)                                          -- $\eta$-reduction
==  dot *** dot                                                        -- def'n of |(***)| for |(->)|
\end{code}
\item Follows from inverting each side of part \ref{unjoin-dot}.
\item Noting again that the argument of both sides is a pair,
\begin{code}
    dot u ||| dot v
==  jamP . (dot u *** dot v)                   -- definition of |(###)|
==  \ (x,y) -> jamP ((dot u *** dot v) (x,y))  -- definition of |(.)| for functions
==  \ (x,y) -> jamP (dot u x, dot v y)         -- definition of |(***)| for functions
==  \ (x,y) -> dot u x + dot v y               -- definition of |jamP| for functions
==  \ (x,y) -> dot (u,v) (x,y)                 -- definition of |dot| for pairs
==  dot (u,v)                                  -- $\eta$-reduction
\end{code}
\item Immediate from linearity and the definition of |zero| for functions.
\end{enumerate}
\emph{End of proof of \lemRef{dot-properties}}.\\

Recall the definition of |asDual| from \secref{Gradients and duality}:
\begin{code}
asDual :: (HasDot s a, HasDot s b) => ContC k s a b -> DualC k a b
asDual (Cont f) = Dual (onDot f)
\end{code}
where
\begin{code}
onDot :: (HasDot s a, HasDot s b) => ((b :-* s) -> (a :-* s)) -> (b :-* a)
onDot f = unDot . f . dot
\end{code}
For the |Category| instance of |DualC k|, we'll need that |id == asDual id|.
Simplifying the RHS,
\begin{code}
    asDual id
==  asDual (Cont id)         -- definition of |id| for |ContC k r| (\figref{cont})
==  Dual (unDot . id . dot)  -- definition of |asDual|
==  Dual (unDot . dot)       -- |Category| law for |id|/|(.)|
==  Dual id                  -- |unDot . dot == id|
\end{code}
We also need |asDual (g . f) == asDual g . asDual f|, or (without loss of generality)
\begin{code}
asDual (Cont g . Cont f) == asDual (Cont g) . asDual (Cont f)
\end{code}
Simplifying both sides,
\begin{code}
    asDual (Cont g . Cont f)
==  asDual (Cont (f . g))                     -- definition of |(.)| for |ContC k r|
==  Dual (unDot . f . g . dot)                -- definition of |asDual|
==  Dual (unDot . f . dot . unDot . g . dot)  -- |dot . unDot == id|
==  Dual (onDot f . onDot g)                  -- definition of |onDot|
    
    asDual (Cont g) . asDual (Cont f)
==  Dual (onDot g) . asDual (onDot f)         -- definition of |asDual|
\end{code}
As usual, strengthen this equality by replacing |onDot g| and |onDot f| by re-typed |g| and |f|, and read off a sufficient definition:
\begin{code}
Dual (f . g) == Dual g . asDual f
\end{code}

For |MonoidalPCat|, the homomorphism condition is |asDual (f *** g) == asDual f *** asDual g|.
Simplify both sides:
\begin{code}
    asDual (Cont f) *** asDual (Cont g)
==  Dual (onDot f) *** Dual (onDot g)                     -- definition of |asDual|
    
    asDual (Cont f *** Cont g)
==  asDual (Cont (join . (f *** g) . unjoin))             -- definition of |(***)| on |Cont|
==  Dual (onDot (join . (f *** g) . unjoin))              -- definition of |asDual|
==  Dual (unDot . join . (f *** g) . unjoin . dot)        -- definition of |onDot|
==  Dual ((unDot *** unDot) . (f *** g) . (dot *** dot))  -- \lemDotTwo{unjoin-dot}{unDot-join} 
==  Dual (unDot . f . dot *** unDot . g . unDot)          -- law about |(***)|/|(.)|
==  Dual (onDot f *** onDot g)                            -- definition of |onDot|
\end{code}
Strengthening from |onDot f| and |onDot g| gives a simple sufficient condition:
\begin{code}
Dual f *** Dual g == Dual (f *** g)
\end{code}

For |ProductCat|,
\begin{code}
    exl
==  asDual exl                                  -- specification
==  asDual (Cont (join . inl))                  -- definition of |exl| for |ContC k r|
==  Dual (onDot (join . inl))                   -- definition of |asDual|
==  Dual (unDot . join . inl . dot)             -- definition of |onDot|, and associativity of |(.)|
==  Dual (\ u -> unDot (join (inl (dot u))))    -- definition of |(.)| for functions
==  Dual (\ u -> unDot (join (dot u, zero)))    -- definition of |inl| for functions
==  Dual (\ u -> unDot (dot u ||| zero))        -- definition of |join|
==  Dual (\ u -> unDot (dot u ||| dot zeroV))   -- \lemDot{dot-zeroV}
==  Dual (\ u -> unDot (dot (u,zeroV)))         -- \lemDot{dot-dot-join}
==  Dual (\ u -> (u,zeroV))                     -- |unDot . dot == id|
==  Dual (\ u -> inl u)                         -- definition of |inl| for functions
==  Dual inl                                    -- $\eta$-reduction
    
    exrP
==  Dual inr                                    -- as with |exlP|
    
    dup
==  asDual dup                                           -- specification
==  asDual (Cont (jamP . unjoin))                        -- definition of |dup| for |ContC k r|
==  Dual (onDot (jamP . unjoin))                         -- definition of |asDual|
==  Dual (unDot . jamP . unjoin . dot)                   -- definition of |onDot|
==  Dual (\ (u,v) -> unDot (jamP (unjoin (dot (u,v)))))  -- definition of |(.)| for functions
==  Dual (\ (u,v) -> unDot (jamP (dot u, dot v)))        -- \lemDot{unjoin-dot}
==  Dual (\ (u,v) -> unDot (dot u + dot v))              -- definition of |jamP| for functions
==  Dual (\ (u,v) -> unDot (dot u) + unDot (dot v))      -- \lemDot{unDot-linear}
==  Dual (\ (u,v) -> u + v)                              -- |unDot . dot == id|
==  Dual jamP                                            -- definition of |jamP| for functions
\end{code}
The |CoproductPCat| instance comes out similarly:
\begin{code}
    inlP
==  asDual inlP                                          -- specification
==  asDual (Cont (exl . unjoin))                         -- definition of |inlP| for |ContC k r|
==  Dual (onDot (exl . unjoin))                          -- definition of |asDual|
==  Dual (unDot . exl . unjoin . dot)                    -- definition of |onDot|
==  Dual (\ (u,v) -> unDot (exl (unjoin (dot (u,v)))))   -- definition of |(.)| for functions
==  Dual (\ (u,v) -> unDot (exl (dot u, dot v)))         -- \lemDot{unjoin-dot}
==  Dual (\ (u,v) -> unDot (dot u))                      -- definition of |exl| on functions
==  Dual (\ (u,v) -> u)                                  -- |unDot . dot == id|
==  Dual exl                                             -- definition of |exl| for functions
    
    inrP
==  Dual exr                                             -- \ldots{} as with |inlP| \ldots
    
    jam
==  asDual jam                                  -- specification
==  asDual (Cont (join . dup))                  -- definition of |jam| on |Cont|
==  Dual (onDot (join . dup))                   -- definition of |asDual|
==  Dual (unDot . join . dup . dot)             -- definition of |onDot|
==  Dual (\ u -> unDot (join (dup (dot u))))    -- definition of |(.)| on functions
==  Dual (\ u -> unDot (join (dot u, dot u)))   -- definition of |dup| for functions
==  Dual (\ u -> unDot (dot u ||| dot u))       -- definition of |join|
==  Dual (\ u -> unDot (dot (u,u)))             -- \lemDot{dot-dot-join}
==  Dual (\ u -> (u,u))                         -- |unDot . dot == id|
==  Dual (\ u -> dup u)                         -- definition of |dup| on functions
==  Dual dup                                    -- $\eta$-reduction
\end{code}

Finally, scaling:
\begin{code}
    scale s
==  asDual (scale s)              -- specification
==  asDual (Cont (scale s))       -- definition of |scale| for |ContC k r|
==  Dual (onDot (scale s))        -- definition of |asDual|
==  Dual (unDot . scale s . dot)  -- definition of |onDot|
==  Dual (scale s . unDot . dot)  -- \lemDot{unDot-linear}
==  Dual (scale s)                -- |unDot . dot == id|
\end{code}

\subsection{\corRef{dual-derived}}\proofLabel{corollary:dual-derived}
Given the definitions in \figref{asDual},
\begin{code}
    Dual f &&& Dual g
==  (Dual f *** Dual g) . dup   -- definition of |(&&&)|
==  Dual (f *** g) . dup        -- definition of |(***)| for |DualC k|
==  Dual (f *** g) . Dual jamP  -- definition of |dup| for |DualC k|
==  Dual (jamP . (f *** g))     -- definition of |(.)| for |DualC k|
==  Dual (f ||| g)              -- definition of |(###)|
    
    Dual f ||| Dual g
==  jamP . (Dual f *** Dual g)  -- definition of |(###)|
==  jamP . Dual (f *** g)       -- definition of |(***)| for |DualC k|
==  Dual dup . Dual (f *** g)   -- definition of |jamP| for |DualC k|
==  Dual ((f *** g) . dup)      -- definition of |(.)| for |DualC k|
==  Dual (f &&& g)              -- definition of |(&&&)|
\end{code}

%if indexed

\subsection{\thmRef{indexed}}\proofLabel{theorem:indexed}

%% Given the definitions in \figref{indexed},

We will need an indexed counterpart to \thmRef{cross}, which says
$$|der (f *** g) (a,b) == der f a *** der g b|$$
Letting |cross = uncurry (***)|, we can rephrase this theorem:
\begin{code}
    der (f *** g)
==  \ (a,b) -> der f a *** der g b              -- \thmRef{cross}
==  \ (a,b) -> cross (der f a, der g b)         -- definition of |cross|
==  \ (a,b) -> cross ((der f *** der g) (a,b))  -- definition of |(***)| on functions
==  cross . (der f *** der g)                   -- definition of |(.)| on functions
\end{code}
Likewise, extend from binary to $n$-ary:
\begin{theorem}[indexed cross rule] \thmLabel{crossF}
$$|der (crossF fs) == crossF . crossF (fmap der fs)|$$
\end{theorem}
If |fs :: h (a -> b)|, then both sides of this equation have type |h a -> (h a :-* h b)|.
The proof is similar to \thmRef{cross} \citep[variant of Theorem 2-3 (3)]{Spivak65}.

\thmRef{crossF} gives us what we need to construct |ad (crossF fs)| compositionally:
\begin{corollary} \corLabel{crossF}
|ad| is compositional with respect to |crossF|. Specifically,
$$|ad (crossF fs) == second crossF . unzip . crossF (fmap ad fs)|$$
\end{corollary}
The proof is analogous to that of \corRef{cross}:
\begin{code}
    ad (crossF fs) as
==  (crossF fs as, der (crossF fs) as)                                  -- definition of |ad|
==  (crossF fs as, crossF (crossF (fmap der fs) as))                    -- \thmRef{crossF}
==  second crossF (crossF fs as, crossF (fmap der fs) as)               -- def'n of |second| (\figref{indexed})
==  second crossF ((crossF fs &&& crossF (fmap der fs)) as)             -- def'n of |(&&&)| on functions
==  second crossF (unzip (crossF (zipWith (&&&) fs (fmap der fs)) as))  -- \lemRef{crossZip} below
==  (second crossF . unzip . crossF (fmap ad fs)) as                    -- definition of |(.)| on |(->)|
\end{code}
For the second-to-last step,
\begin{lemma}\lemLabel{crossZip}
|crossF fs &&& crossF gs == unzip (crossF (zipWith (&&&) fs gs))|.
\end{lemma}
For now, let's prove just the binary version of this lemma, namely
$$ |(f *** f') &&& (g *** g') == transpose ((f &&& g) *** (g' &&& g'))| $$
where
\begin{code}
transpose :: ((a :* b) :* (c :* d)) -> ((a :* c) :* (b :* d))
transpose ((a,b),(c,d)) = ((a,c),(b,d))
\end{code}
\out{For general cartesian categories, |transpose = (exl.exl &&& exl.exr) &&& (exr.exl &&& exr.exr)|.}
Proof:
\begin{code}
    (f *** f') &&& (g *** g')
==  (inl . f ||| inr . f') &&& (inl . g ||| inr . g')               -- \citep[Equation (17)]{MacedoOliveira2013Typing}
==  (inl . f &&& inl . g) ||| (inr . f' &&& inr . g')               -- exchange law \citep[Section 1.5.4]{Gibbons2002Calculating}
==  transpose . inl . (f &&& g) ||| transpose . inr . (f' &&& g')   -- \lemRef{inlFork} below
==  transpose . (f *** g) ||| transpose . (f' *** g')               -- \citep[Equation (17)]{MacedoOliveira2013Typing}
==  transpose ((f *** g) ||| (f' *** g'))                           -- \citep[Section 1.5.2]{Gibbons2002Calculating}
\end{code}

For the third step, we need two more properties.
\begin{lemma}\lemLabel{inlFork}
$$ |inl . f &&& inl . g == transpose . inl . (f &&& g)| $$
$$ |inr . f &&& inr . g == transpose . inr . (f &&& g)| $$
\end{lemma}
Below is a proof in the |(->)| category, which suffice for our purpose.
(To do: does the property hold for general biproduct categories?)
\begin{code}
    inl . f &&& inl . g
==  \ a -> (inl . f &&& inl . g) a               -- $\eta$-expand
==  \ a -> (inl (f a), inl (g a))                -- definition of |(&&&)| for functions
==  \ a -> ((f a, zero), (g a, zero))            -- definition of |inl| for functions
==  \ a -> transpose ((f a, g a), (zero, zero))  -- definition of |transpose|
==  \ a -> transpose ((f a, g a), zero)          -- definition of |zero| for pairs
==  \ a -> transpose (inl (f a, g a))            -- definition of |inl| for functions
==  transpose . inl . (f &&& g)                  -- definition of |(.)| for functions
\end{code}
Similarly for the second property (with |inr|).

%endif

%endif %% appendices

\bibliography{bib}

%if draft

\sectionl{To do}
\begin{itemize}
\item Possibly replace most of \secref{Generalized matrices} by a reference \citep{Elliott-2017-compiling-to-categories}, saving nearly a page.
      If so, merge the decimated section into the previous one (``Extracting a data representation'').
\item Maybe define and use |(-+>)|.
\item Return to comparison with TensorFlow etc in the related work and/or conclusions section.
\item Maybe more future work
\item Nested AD. I think the categorical approach in this paper can correctly handle nesting with ease and that the nesting problem indicates an unfortunate choice of abstraction together with non-rigorous specification and development.
\item Resolve possible title or subtitle: ``Differentiable functional programming made easy''.
Note the two meanings: easy to implement correctly and efficiently, and easy to use.
Perhaps save that title for another talk and paper.
A quick web search turns up a few uses of ``differentiable functional programming''.
\item Sub-differentiation. 
\item Probably remove the |Additive| constraints in |Cocartesian|, along with the |Cocartesian (->)| instance.
      Otherwise, mention that the implementation does so.
      |InitialCat (->)| isn't what we need.
\item Consider moving the current examples into a single section after gradients and duality.
      For each example, show the function, |andDerivF|, |andDerivR|, and |andGradR|.
\item Mention graph optimization and maybe show one or more un-optimized graphs.
\item Examples with generalized matrices.
\item Mention flaw in the customary chain rule: the decomposed pieces may not be differentiable.
\item Fix separating vertical bars in the extended version.
%% \item |ConstCat| for |DualC k| and for linear arrows in general.
\item What is ``generalized AD''?
      Is it AD at all or something else?
\end{itemize}

%endif

\end{document}
