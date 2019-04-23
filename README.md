# Structured Prediction with Deep Value Networks
**Implementation in python with PyTorch.
by Philippe Beardsell and Chih-Chao Hsu**

### Models to implement ###
* Structured Prediction Energy Networks (SPEN) (Belanger & McCallum 2015) <br />
(Implementation from David Belanger in Lua at https://github.com/davidBelanger/SPEN)
* Deep Value Networks (Gygli & al. 2017)<br />
(Based on the TensorFlow implementation from the authors at https://github.com/gyglim/dvn)

### Future Goals ###
* Compare and analyze these models on standard multi-label datasets like Bibtex and Bookmarks.
* Use these models on more complex tasks like image segmentation.

### Implemented ###
* Feature network (SPEN): <br /> Multi-layer perceptron that computes a feature representation
of the inputs. Also can be used as a baseline model
* SPEN for multi-label classification/ image segmentation and image tagging. 
* Deep Value Networks for multi-label classification/ image segmentation and image tagging. 
  

### Reproducibility ###
* F1 Score (%) on Bibtex dataset:
| Model        | Ours | Paper  |
| ------- | ------ | ----- |
| MLP | 38.9 | 38.9 |
| SPEN | 41.6 | 42.2 |
| DVN + Ground Truth | 42.9 |  N/A |
| DVN + Adversarial | 44.9 | 44.7 |
We could easily reproduce the authors' results with the DVN on Bibtex (F1 of 44.91% on the test set). Conversely,
for the SPEN model, we achieved a F1 Score of 41.07% on the test set, compared to 42.2% for the authors. We could have probaly done some extra hyper-parameter search to reach it though. <br /> 
<img src="figures/bibtex_dvn_comparisons.png" width="80%">
