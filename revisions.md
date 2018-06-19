# Response to Mandatory Revisions

There was only one mandatory revision in my reviews:

 <blockquote>

Generally, tone down the efficiency claims to ensure that any claims regarding the performance
benefits are carefully justified.

For example, the paper says:

 >
One the main purposes of this paper is to demonstrate that RAD can be implemented
quite simply, with performance equal to or better than conventionally complex
implementations.

The paper does not, however, actually compare the performance with that of "conventionally complex implementations".

(Moreover, such a comparison would have to be very carefully designed to capture the differences due to to the AD algorithm as such, as opposed to differences due to other factors such as the efficiency of underlying tensor operations.)

 </blockquote>

I addressed this issue by making the following changes.
Line numbers refer to the originally submitted paper.
Changes are shown in square brackets.

*   95: "It requires no matrix-level representations or computation and is suitable for gradient-based optimization, e.g., for machine learning."
    ["It also appears to be quite efficient---requiring no matrix-level representations or computations---and is suitable for gradient-based optimization, e.g., for machine learning. In contrast to conventional reverse-mode AD algorithms, all algorithms in this paper are free of mutation and hence naturally parallel."]
*   268: "Corollaries 1.1 through 3.1 provide insight into the compositional nature of |ad| in exactly the form we can now assemble into an efficient, correct-by-construction implementation."
    [Replaced "an efficient, correct-by-construction" by "a correct-by-construction".]
*   1295: "The specification and implementation of AD in a simple, efficient, and correct-by-construction manner, ..."
    ["The specification and implementation of AD in a simple, correct-by-construction, and apparently efficient manner, ...".]
*   863: "One of the main purposes of this paper is to demonstrate that RAD can be implemented quite simply, with performance equal to or better than conventionally complex implementations."
    ["One of the main purposes of this paper is to demonstrate that these complications are inessential and that RAD can instead be specified and implemented quite simply."]
*   1304: "Then transform a correct, naive representation into subtler, high-performance representations."
    [Replaced "high-performance" by "more efficient", referring to reverse-mode AD relative to other modes for gradient computation.]
