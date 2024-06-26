This is a fork of the [SimKGC: Simple Contrastive Knowledge Graph Completion with Pre-trained Language Models](https://github.com/intfloat/SimKGC) repository, described in the ACL 2022 [paper](https://aclanthology.org/2022.acl-long.295.pdf of the same name by Wang et al.

The primary change in the codebase is to support pre-processing and training for KGX-based knowledge graphs at [kghub.org](https://kghub.org). This currently requires hard-coding the KG .tar.gz URL in the `scripts/preprocess_kghub.sh`; after installing dependencies with `poetry install`, edit and then run this, then run `scripts/train_kghub.sh` to train the model. Hyperparams may be edited in the train script (also see the Makefile).

I've also added an option to support the Apple [MPS](https://pytorch.org/docs/stable/notes/mps.html) device backend. However, training times are actually worse than CPU on my M3 Max, so perhaps this isn't configured correctly.

Many thanks to the SimKGC team for this great and extensible work!