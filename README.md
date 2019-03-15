# Structured Prediction with Deep Value Networks
**Implementation in python with PyTorch.**

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
* SPEN for multi-label classification 
* Deep Value Networks: Almost completed. <br /> Missing:
  * Parallel inference. 
  * Reproduction of the authors' results on the Weizmann horses dataset.
  

### Reproducibility ###
We could easily reproduce the authors' results with the DVN on Bibtex (F1 of 44.07% on the test set). Conversely,
we had trouble reproducing the SPEN's results and we only achieved a F1 Score of 40% on the test set, compared to 42.2 for the authors. We could have probaly done some extra hyper-parameter search to reach it though. <br /> 
<img src="figures/bibtex_dvn_comparisons.png" width="80%">
