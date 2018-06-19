---
title: ICFP 2018 paper still to do
...

## To do

*   Update abstract and paper [on arXiv](https://arxiv.org/abs/1804.00746).

*   Update the abstract on the submission page.
*   Acknowledgments.
*   Non-anonymous references to extended version from ICFP version.
*   From reviews:
    *   I find the paper difficult to read front to back. The description which starts the conclusions would help as a forward overview of the achievement.
        I think it could help to split the unusual high number of sections into a bit more hierarchy with subsections?

    *   144: "The derivative of a function" => "The derivative $\mathcal{D}$ of a function" 
    *   267: "The definition of $\mathcal{D}^+$ ...": Perhaps give a number to the definition at lines 197-198 (like the key theorems and corollaries) to facilitate referenencing and make it precise.
    *   311: Why the dashed arrow in the type of "F f"? It's also a morphism?
    *   1309-1316: I found the writing here a bit unclear.

    *   527: it would be helpful to mention the associated constraint for
        `Category` earlier, or at least mention that you're going to add
        more to the class.  I was confused for a while reading Fig. 1 (top
        of p11), since the `Category` instance for `→⁺` had an additional
        member that didn't appear in the class definition.
    *   In a few places I think the exposition could benefit from additional
        refinement.  For example, I like the idea of section 6 (stepping
        back to find a general recipe).  These kinds of recipes are core to
        functional programming, and it's nice to see this one made explicit.
        However, in its current form it's quite difficult to see the essence
        of the recipe.  I think it might be clearer to split the
        presentation into three parts
        *   the general recipe itself, presented very succinctly;  
        *   the particular instance of the recipe here, differentiation  
        *   commentary on the recipe  

      (The first two points might go in side-by-side columns, much as
      you've presented classes and instances in earlier sections.)  And if
      it's possible to reduce the number of steps, all the better.

    * Section 7: these changes are not especially well motivated.  You say
      what properties are needed to support the derivatives seen so far,
      and then go on to say *what* you're going to change but not really
      *why* these are the right changes to make.  ("A few small changes
      ... result in the generalized AD definitions")

    * Several diagrams (e.g. Figs. 4, 5, 9) are quite small and hard to
      read.  Furthermore, the layout doesn't seem optimal – e.g. in Fig. 4
      could the top two boxes in the second column be swapped to avoid
      several crossings?

    * The claims about time & space efficiency are plausible but
      unsubstantiated; you say that detailed performance evaluation is
      left for future work.  I don't think there's a need for a full
      quantitative evaluation in the present paper, but it would be
      helpful to have at least some preliminary figures (e.g. for
      such-and-such an application, maximum memory usage drops from X MB
      to Y MB) to support your claims.

    * I think some of the discussion in the conclusion (e.g. the
      comparisons between SD and AD, could profitably be moved to the
      introduction.  I would have found it helpful to know more of the
      story up front.

    * Is it possible to more directly compare the RAD implementation in
      this paper with more conventional approaches (tapes, graphs, etc.)?
      You've commented on various differences, but what is the
      relationship between the approaches?

*   Left/right-reverse the homomorphism equations, adding a remark about fitting the derivations.
*   In the last paragraph of the Related Work section, with efficiency remarks, maybe mention parallel evaluation again.
    Instead or in addition, discuss in the conclusions as well.
*   `CoproductPCat` for `GD k`.
    The "cocartesian rule", a peer to the chain and cartesian rules.

## Did

*   Even out the font sizes in the side-by-side figures 11 & 12. [done]
*   Abstract: "can be specialized" --> "is then specialized". [done]
*   As required by my reviews, tone down efficiency claims, including the following:
    *   "The choice of dualized linear functions for gradient computations is particularly compelling in simplicity and efficiency. It requires no matrix-level representations or computation and is suitable for gradient-based optimization, e.g., for machine learning."
        ["It also appears to be quite efficient---requiring no matrix-level representations or computations---and is suitable for gradient-based optimization, e.g., for machine learning. In contrast to conventional reverse-mode AD algorithms, all algorithms in this paper are free of mutation and hence naturally parallel."]
    *   "Corollaries 1.1 through 3.1 provide insight into the compositional nature of |ad| in exactly the form we can now assemble into an efficient, correct-by-construction implementation."
        [Replace "an efficient, correct-by-construction" by "a correct-by-construction".]
    *   "The specification and implementation of AD in a simple, efficient, and correct-by-construction manner, ..."
        ["The specification and implementation of AD in a simple, correct-by-construction, and apparently efficient manner, ...".]
    *   "One of the main purposes of this paper is to demonstrate that RAD can be implemented quite simply, with performance equal to or better than conventionally complex implementations."
        ["One of the main purposes of this paper is to demonstrate that these complications are inessential and that RAD can instead be specified and implemented quite simply."]
    *   "Then transform a correct, naive representation into subtler, high-performance representations."
        [Replace "high-performance" by "more efficient".]

*   From reviews:
    *   1244: duplicate: "that the that the"
        [done]
    *   910: sentence lacking a verb. "This trick akin ..."
        [done]
    *   159: Why the emphasis on the Hessian matrix here?  You don't seem to
        use it again.
        [Emphasis unintentional. Changed "In particular" to "For instance".]

    Minor:

    *   411: "the nature `D`" ~> "the nature of `D`" [done]
    *   910: "This trick akin" [done]
    *   945: this sentence would benefit from reworking -- e.g. what does
        "(e.g., for optimization)" belong to?
        [I replaced "(e.g., for optimization)" by "as with gradient-based optimization"]

*   `AddFun`
*   Abstract
*   Use infix `(~>)` in place of `k`.
*   Rework `Incremental`.
*   Resolve whether to give the talk at PEPM.
    [Yes](https://popl18.sigplan.org/track/PEPM-2018#Invited-Talks).
*   AD example `cosSinProd`
*   Extracting a data representation: if we use `AddFun`, we have an expensive extraction step e.g., to get the gradient.
*   RAD examples with derivatives as dual functions and as dual vectors: `add`, `fst`, `sqr`, `magSqr`, `cosSinProd`.
*   Get the talk scheduled at work. Wed 3pm.
*   Add `Incremental` to talk.
*   Remarks on symbolic vs automatic differentiation
*   Maybe show CPS-like category (`ConCat.Continuation`, in progress).
*   Conclusions: FAD & RAD ...
*   Linear maps via representable functors.
*   Perhaps material from second set of notes
