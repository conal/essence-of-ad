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

\nc\R{\mathbb{R}}
%format R="\R"
\nc\C{\mathbb{C}}
%format C="\C"

\nc\eps\epsilon
\rnc\eps\varepsilon

Since automatic differentiation (AD) has to do with computing derivatives, let's begin by considering what derivatives are.
If your introductory calculus class was like mine, you learned that the derivative $f'(x)$ (or |f' x|) of a function $f : \R \to \R$ at a point $x$ (in the domain of $f$) is a \emph{number}, defined as follows;
$$ f'(x) = \lim_{\eps \to 0}\frac{f(x+\eps) - f(x)}{\eps} $$
That is, $f'(x)$ tells us how fast $f$ is scaling input changes at $x$.

How well does this definition hold up beyond functions of type $\R \to \R$?
It will do fine with complex numbers ($\C \to \C$), where division is also defined.
Extending to $\R \to \R^n$ also works if we interpret the ratio as dividing a vector by a scalar in the usual way.
When we extend to $\R^m \to \R^n$, however, this definition no longer makes sense, as it would rely on dividing \emph{by} a vector.
Fortunately, a more abstract version of the $f'$ above does generalize.

First, change the $f'$ definition above ...: unique scalar $s$ such that

\end{document}

 \[ |lim(epsilon -> 0)(frac(f (x+epsilon) - f(x)) epsilon) - s == 0| \]

\pause
Equivalently,

 \[ |lim(epsilon -> 0)(frac(f (x+epsilon) - f(x) - s *^ epsilon) epsilon) == 0| \]
or
 \[ |lim(epsilon -> 0)(frac(f (x+epsilon) - (f(x) + s *^ epsilon)) epsilon) == 0| \]

}

\frame{\frametitle{What's a derivative -- really?}
 \[ |lim(epsilon -> 0)(frac(f (x+epsilon) - (f(x) + s *^ epsilon)) epsilon) == 0| \]

\pause

\ 

Now generalize: unique \wow{linear map} $T$ such that:

$$|lim(epsilon -> 0)(frac(abs (f (x+epsilon) - (f(x) + T epsilon)))(abs epsilon)) == 0|$$

\ 


\framet{What's a derivative?}{
\begin{itemize}\itemsep4ex
\pitem Number
\pitem Vector
\pitem Covector
\pitem Matrix
\pitem Higher derivatives
\end{itemize}

\vspace{2ex}\pause
Chain rule for each.
}

