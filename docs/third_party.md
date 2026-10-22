# Third-party notices

The MAE model and training utilities derive from [Meta's MAE implementation](https://github.com/facebookresearch/mae). The original Meta notices remain in the source files. Philipp Allgeuer and Jan-Gerrit Habekost adapted the model, dataset loader, training entry point, training engine, and checkpoint utilities for AnomNOVIC. Philipp Allgeuer also updated the position embedding utility for NumPy compatibility. The learning rate scheduler remains unchanged from Meta's implementation. These adaptations are distributed under the same [CC BY-NC 4.0 license](../LICENSE) as the source project.

The synthetic object cutouts in the training archives are derived from the [YCB Object and Model Set](https://ycb-benchmarks.s3.amazonaws.com/index.html), provided by Berk Calli, Arjun Singh, Aaron Walsman, Siddhartha Srinivasa, Pieter Abbeel, and Aaron M. Dollar under CC BY 4.0. YCB attribution and its original license continue to apply to those cutouts.

The pipeline diagram and Wild prediction image in the README are reproduced from the [AnomNOVIC arXiv submission](https://arxiv.org/abs/2606.26829) by Philipp Allgeuer, Jan-Gerrit Habekost, and Stefan Wermter. These figures retain the paper's [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/).

NOVIC and optional SAM2 are external dependencies. Their source code is not bundled here. Use their respective repositories and licenses.
