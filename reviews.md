---
title: Reviews for "The simple essence of AD", ICFP 2018
...

ICFP '18 Paper #18 Reviews and Comments
===========================================================================
Paper #18 The simple essence of automatic differentiation (Differentiable
functional programming made easy)


Review #18A
===========================================================================

**Overall merit**:
A. Good paper, I will champion it

**Reviewer expertise**:
Y. I am knowledgeable in this area, but not an expert

Paper summary
-------------
The main contribution of this paper is to derive novel and interesting (potentially useful and simpler) algorithms for automatic differentiation (AD) in reverse mode (RAD) using an abstract approach based on category theory. This contribution was hinted as future work in prior work on beautiful differentiation (Elliot 2009) and compiling to categories (Elliot 2017). This paper uses the machinery of compiling to categories so that differentiable functions can be treated as first-class "function-like things".

Section 2 introduces differentiation.

Section 3 presents the rules for differentiation.

Section 4 "putting the pieces together" develops the first version of AD based on linear maps and category theory.

Section 5 "examples" shows an example of AD in action, compiling to graphs for clarity.

Section 6 "programming as defining and solving algebra problems" states a key principle of this paper.

Section 7 "Generalizing Automatic Differentation" generalizes the approach so far so that linear maps can be replaced by a broader variety of categories.

Section 8 "Matrices" motivates the next section. I think Sections 8-10 could be merged?

Section 9 "Extracting a Data Representation" discusses why functions might not always be the best intermediate representation. It motivates the following section.

Section 10 "Generalized Matrices" considers computing matrices during composition rather than extracting them after the fact, which is more costly.

Section 11 "Efficienty of Composition" ponders on how the representations affect what is easy to do, and what is costly to do.

Section 12 derives RAD by finding a representation that is optimized for left composition, via continuations.

Section 13 "Gradients and duality" considers the special case of RAD for gradients and their optimizations.

(Tiny) Section 14 "Forward-mode AD" shows how to turn the derivation around for forward mode.

Section 15 "Scaling up" shows how to generalize to n-products, which is important for optimization problems involving high-dimensional spaces.

Rating rationale and comments for author
----------------------------------------
The paper is gentle in that it starts with the high school definition of a derivative. I like that it uses linear maps by citing Spivak's Calculus on Manifolds, which notably has better notation than usual calculus textbooks. The notation inspired the functional notation in the executable textbook Structure and Interpretation of Classical Mechanics by Sussman and Wisdom, and so it's nice to see that kind of notation here too. This realization is not new, going back to Elliot 2009.

The development of the category theory is also rather self-contained. However, I had to consult Elliot (2017) to understand the compilation to categories, which is left as magic here and also concludes the paper like a teaser.

I like the first principle set in the conclusion: "focus on abstract notions (specified denotationally and/or axiomatically) rather than particular representations. Then transform a correct, naive representation into subtler, high-performance representations." The short section 6 sort of instantiate this principle with more details.

The paper leaves for future work any form of performance evaluation or evaluation of an implementation based on the techniques presented. This could be OK as I value the conceptual contribution, but it's also a shame not to know where the derived technique stands.

I find the paper difficult to read front to back. The description which starts the conclusions would help as a forward overview of the achievement.

I think it could help to split the unusual high number of sections into a bit more hierarchy with subsections?

Minor:

*  line 1244: duplicate: "that the that the"
*  line 910: sentence lacking a verb. "This trick akin ..."



Review #18B
===========================================================================

**Overall merit**:
A. Good paper, I will champion it

**Reviewer expertise**:
Y. I am knowledgeable in this area, but not an expert

Paper summary
-------------
This paper provides a beautiful, general presentation of automatic
differentiation (AD), expressed in categorical terms. Specialization then
yields concrete AD algorithms, including novel reverse mode AD (RAD)
algorithms that are much simpler than previously known ones, and has much
greater potential for being implemented efficiently on parallel hardware as
they, unlike existing ones, do not rely on imperative side effects. A
particular important application could be for implementing optimization by
gradient decent, of key imporatnce e.g. in modern multi-layer neural networks.

Rating rationale and comments for author
----------------------------------------
Very well written paper, and then in particular the first part, which is
exceptional. The paper thus makes contributions to functional programming,
as it shows how to go about deriving complex algorithms functionally
in a way that is correct by construction, and to the field of AD as it
presents novel, simple RAD algorithms that can be implemented very efficiently.

That said, the lack of a performance evaluation, or at least comments on
performance, is a bit disappointing given that is one of the hypothesised
advantages of the novel RAD algorithms.

Some larger examples/applications, or at least a discussion of such
(perhaps one involving high-dimensional gradient decent optimisation, or
at least one illustrating differentiation of functions on more than a
handful of dimensions) would also have been good. 

Nevertheless, a clear accept in my opinion.

Minor presentational points
---------------------------

Line 144: "The derivative of a function" => 
          "The derivative $\mathcal{D}$ of a function" 

Line 267: "The definition of $\mathcal{D}^+$ ...": Perhaps give a number
          to the definition at lines 197-198 (like the key theorems and
          corollaries) to facilitate referenencing and make it precise.

Line 311: Why the dashed arrow in the type of "F f"? It's also a morphism?

Lines 1309-1316: I found thenwriting here a bit unclear.



Review #18C
===========================================================================

**Overall merit**:
A. Good paper, I will champion it

**Reviewer expertise**:
Y. I am knowledgeable in this area, but not an expert

Paper summary
-------------
The paper presents a new formulation of reverse-mode automatic
differentiation in categorically-flavoured functional terms.  Previous
work, such as Elliott's "Beautiful Differentiation", has focused on
forward-mode differentiation, which is easier to define, but often too
inefficient for important applications, such as machine learning,
where the domain of the differentiated functions is high-dimensional.
This paper presents a uniform treatment of the forward and reverse
modes; furthermore, the reverse mode is developed without any of the
usual awkward tapes and graphs that are used in most existing
implementations.

Rating rationale and comments for author
----------------------------------------
I am strongly in favour of accepting this paper.  Although the
abstraction level is quite high (perhaps endangering accessibility),
the abstraction pays off both in succinctness of definitions, and in
the ease of replacing the various components with alternative
instances as the development proceeds.  The various structures
defined, especially the homomorphic instances, often have a canonical
feel.

The work builds on several strands of FP research: categorical
structuring of programs, CPS translation, representable functors,
etc. to address an practical problem in an elegant way.  It also seems
well-informed respecting previous work on automatic differentiation,
although I am much less able to judge here.

### Comments & questions

*   159: Why the emphasis on the Hessian matrix here?  You don't seem to
    use it again.
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
    - the general recipe itself, presented very succinctly;  
    - the particular instance of the recipe here, differentiation  
    - commentary on the recipe  

    (The first two points might go in side-by-side columns, much as
    you've presented classes and instances in earlier sections.)  And if
    it's possible to reduce the number of steps, all the better.
*   Section 7: these changes are not especially well motivated.  You say
    what properties are needed to support the derivatives seen so far,
    and then go on to say *what* you're going to change but not really
    *why* these are the right changes to make.  ("A few small changes
    ... result in the generalized AD definitions")
*   Several diagrams (e.g. Figs. 4, 5, 9) are quite small and hard to
    read.  Furthermore, the layout doesn't seem optimal – e.g. in Fig. 4
    could the top two boxes in the second column be swapped to avoid
    several crossings?
*   The claims about time & space efficiency are plausible but
    unsubstantiated; you say that detailed performance evaluation is
    left for future work.  I don't think there's a need for a full
    quantitative evaluation in the present paper, but it would be
    helpful to have at least some preliminary figures (e.g. for
    such-and-such an application, maximum memory usage drops from X MB
    to Y MB) to support your claims.
*   I think some of the discussion in the conclusion (e.g. the
    comparisons between SD and AD, could profitably be moved to the
    introduction.  I would have found it helpful to know more of the
    story up front.
*   Is it possible to more directly compare the RAD implementation in
    this paper with more conventional approaches (tapes, graphs, etc.)?
    You've commented on various differences, but what is the
    relationship between the approaches?

Minor:

*   411: "the nature `D`" ~> "the nature of `D`"
*   910: "This trick akin"
*   945: this sentence would benefit from reworking -- e.g. what does
    "(e.g., for optimization)" belong to?



Review #18D
===========================================================================

**Overall merit**:
A. Good paper, I will champion it

**Reviewer expertise**:
Y. I am knowledgeable in this area, but not an expert

Paper summary
-------------
The paper develops a Haskell compiler plugin to allow programmers to write differentiable functions without explicitly manipulating computation graph by using code transformation technique.

Rating rationale and comments for author
----------------------------------------
Automatic differentiation (AD) plays a central role in many numerical software. Besides its well-known use case in optimisation, the derivative information is also used in many others ways such as sampling, hyper-parameter tuning, and etc. 

Forward AD is ideal for the functions of small input and large output. The implementation is straightforward and dual space is often used. Reverse AD is ideal for large input but small output (e.g. neural networks), and it is implementation requires constructing and manipulating computation graphs therefore it is much more complicated than Forward AD. Depending on when a graph is constructed (compilation phase or runtime), AD tools can be further categorised as dynamic graph and static graph (as the one proposed in this paper).

The authors in this paper develops a compiler plugin which transforms programmers' source code into differentiable functions during compilation phase. By so doing, the construction and maintenance overhead can be avoided during the runtime. Moreover, as claimed by the authors, the memory footprint is smaller with the proposed technique. However, it is not clear to me how much pressure GC will undertake especially when large tensor is involved in the computation.

While dynamic graph provides great flexibility, static graph has bigger optimisation potential. However Google's TensorFlow requires explicit construction especially for branching/controlling node (if-else, for loops), and etc. It is also not clear how this kind of control structure can be introduced in the proposed solution.

The plugin generalises the AD operation and is mode independent, i.e. it can be applied to both Forward and Reverse mode (via Dual and Cont respectively). More specifically, the paper views AD operations as linear mapping, then uses functor to transform the original computation graph into its corresponding adjoint one in order to calculate the derivative. The support of higher-order derivatives is achieved by composability.

Overall, I quite enjoy reading this paper despite of some minor technical things remain unclear to me. The proposed method appears valid with justified motivation. The problem formulation is well structured and solution is clear.


Comment @A1
---------------------------------------------------------------------------
Conditionally Accepted with Mandatory Revisions
-----------------------------------------------

Your paper has been conditionally accepted to ICFP 2018, pending revisions as listed below.

The deadline for completing and submitting the revised version of your paper is June 22. In a brief second phase of reviewing, the program committee will determine whether the mandatory revisions have been applied, and the paper will be conclusively accepted or rejected. The intent and expectation is that mandatory revisions can be performed within the five-week period between notification and June 22 — and that conditionally accepted papers will, in general, be accepted in the second phase.

Your revised submission *must* be accompanied by a cover letter mapping each mandatory revision to specific parts of the paper. The cover letter should facilitate a quick second review, allowing for confirmation of final acceptance within two weeks. Both the final version and the cover letter must be uploaded to <https://icfp18.hotcrp.com>.

 Of course, you are also encouraged to make other changes to your paper in response reviews and comments. You must also ensure that the revised version of your paper meets the formatting requirements specified in the call for papers, including both the document style and a specific form and use of citations, which are necessary for inclusion in PACM PL.

You will receive information from Conference Publishing about *separately* submitting a camera-ready version of your paper by July 7, pending acceptance in the second round of review. The caremera-ready version is likely the same as the final version that you submit by June 22, but there will be a little time for small changes if needed.

Please contact the program chair, Matthew Flatt <mflatt@cs.utah.edu>, if you have any questions about the process.

Mandatory Revisions
-------------------

Generally, tone down the efficiency claims to ensure that any claims regarding the performance
benefits are carefully justified.

For example, the paper says:

 >
One the main purposes of this paper is to demonstrate that RAD can be implemented
quite simply, with performance equal to or better than conventionally complex
implementations.

The paper does not, however, actually compare the performance with that of "conventionally complex implementations".

(Moreover, such a comparison would have to be very carefully designed to capture the differences due to to the AD algorithm as such, as opposed to differences due to other factors such as the efficiency of underlying tensor operations.)
