# NumPy reference provenance

Copied from this student's Assignment 1 `Part_2/modules.py` and
`Part_2/mlp_numpy.py`. Only the MLP import was changed to a relative import.
No numerical operations were changed. Assignment 1 itself remains untouched.

The reference CrossEntropy backward already differentiates with respect to
logits, so SoftMax.backward deliberately passes its input through. This is a
coupled implementation, not a general softmax Jacobian. Probability clipping
at 1e-12 means the forward loss and analytical gradient can disagree at extreme
logits. The comparison records the minimum true-class probability so that this
limitation is visible.
