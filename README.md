# Experiments in Structured Prediction 
**Implementation in python with PyTorch.**

### Models to implement ###
* Structured Prediction Energy Networks (SPEN) (Belanger & McCallum 2015) <br />
(Implementation from David Belanger in Lua at https://github.com/davidBelanger/SPEN)
* Inference Networks (INFNET) (Tu & Gimelp 2018) <br />
(Implementation from the authors in Theano at https://github.com/lifu-tu/INFNET) <br />
(Implementation in TensorFlow can also be found at https://github.com/TheShadow29/infnet-spen)
* Deep Value Networks (Gygli & al. 2017)<br />
(Based on the TensorFlow implementation from the authors at https://github.com/gyglim/dvn)

### Future Goals ###
* Compare and analyze these models on standard multi-label datasets like Bibtex and Bookmarks.
* Use these models on more complex tasks like image segmentation.

### Implemented ###
* Feature network (INFNET & SPEN): <br /> Multi-layer perceptron that computes a feature representation
of the inputs. Also can be used as a baseline model
* Deep Value Networks: Almost completed. <br /> Missing:
  * Parallel inference. 
  * Reproduction of the authors' results on the Weizmann horses dataset.
  

### Reproducibility ###
We could easily reproduce the authors' results with the DVN on Bibtex (F1 of 44.07% on the test set) :
<img src="figures/bibtex_dvn_comparisons.png" width="80%">
