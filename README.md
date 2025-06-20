# Control dataset for Color Equivariant Convolutional Networks

*Written by: Alexandru-Sebastian Nechita (5553946)*


**Hypothesis: Color-equivariant convolutional networks can handle datasets with color-class imbalance more effectively than conventional CNNs.[1]**

Dataset could be found at: https://github.com/SebiNechita/ColorDigit/tree/master/longtailed_colormnist

## Motivation

In many real‐world computer vision applications, two fundamental challenges frequently co‐occur: class imbalance and spurious color correlations. For instance, in medical diagnostics, rare pathologies may be underrepresented in training data, while staining procedures introduce color variations that conventional convolutional neural networks (CNNs) often mistake for meaningful features. Similarly, in wildlife monitoring, sightings of endangered species are scarce, and lighting conditions can bias color distributions. If a model learns to associate a rare class with a particular hue—rather than the object’s true shape—it will struggle to generalize when colors shift.

To address this, Lengyel et al. (2023) propose Color Equivariant Convolutional Networks, arguing that embedding color‐equivariance directly into the architecture improves robustness when color and class distributions are skewed.

To rigorously test how architectural color‐equivariance combats class imbalance, we construct Long‐Tail ColorMNIST, a synthetic dataset with the following properties:

1. *Semantic Classes* **(10)**: The ten digit identities from MNIST (0–9).
2. *Color Variants* **(3)**: Three base hues—Red, Green, Blue—applied orthogonally to digit shape.
3. *Total Classes* **(30)**: One class for each digit–color combination
4. *Balanced Test Set*: Exactly 100 images per class, ensuring evaluation is not confounded by frequency.
5. *Long-Tail Training Split*: The training data follows a long-tailed distribution across the 30 classes using a geometric decay (approximating a Pareto 80/20 rule). This means that a few classes (e.g., red 0, green 1) have thousands of samples, while most others have very few.
6. *Hold-Out Set (Zero-Shot Combinations)*: Some digit–color combinations are completely excluded from the training set {(9, Blue), (8, Green), (7, Red)}


By holding out specific digit–color combinations, we introduce a scenario in which traditional CNNs which conflate color with class are expected to struggle. Standard models may be unable to classify a red 7 if that color-digit pair was never observed during training. In contrast, a color-equivariant model should, in principle, generalize from having seen other red digits and other examples of the digit 7.

The design of Long-Tail ColorMNIST allows us to dissect two specific effects:

* The spurious correlation effect: Does the model overfit to hue when it correlates with class?
* The long-tail effect: How well can the model recognize rare or underrepresented classes?

A well-structured evaluation would involve training a standard CNN and a CE-CNN on the same long-tailed training set and then comparing their per-class accuracy. Of particular interest are:

* The bottom 20% of classes in terms of training frequency (to measure resilience to class imbalance).
* The three zero-shot classes (to measure generalization to unseen color–digit combinations).

### References

[1] Lengyel, A., Strafforello, O., Bruintjes, R.-J., Gielisse, A., & van Gemert, J. (2023). Color Equivariant Convolutional Networks. Advances in Neural Information Processing Systems (NeurIPS). https://arxiv.org/abs/2310.19368
