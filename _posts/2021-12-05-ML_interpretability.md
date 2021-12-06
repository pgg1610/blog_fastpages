---
toc: true
# sticky_rank: 1
layout: post
description: Curated list of resources for data, science, and navigating life
categories: [exploratory-data-analysis, machine-learning, resources, chemical-science]
title: On machine learning model interpretability 
---

Process mindset vs outcome mindset argument -- understanding right decision and right outcomes.

Helpful [notebook](https://patwalters.github.io/practicalcheminformatics/jupyter/ml/interpretability/2021/06/03/interpretable.html) on simple and useful tips on model interpretations in chemical science from ever-amazing Patrick Walters. 

## Future Reading / Additional Links

* Online Jupyter book on Interpretable Machine Learning(https://christophm.github.io/interpretable-ml-book/index.html) -- __highly recommended__ 

Fantastic book commenting on the mathematics and idea for different method for ML model interpretatability. 

* [Explainable AI Cheetsheet](https://ex.pegg.io/Explainable-AI-cheat-sheet-v0.2.1080.png). [Video](https://www.youtube.com/watch?v=Yg3q5x7yDeM) discussion on the general idea by Jay Alammar. 

* [Gradient Blog Commentary on ML interpretability](https://thegradient.pub/interpretability-in-ml-a-broad-overview/)

* [Zoom in on circuits of a Neural Network](https://distill.pub/2020/circuits/zoom-in/)

* [Neptune Blog article on interpretable models](https://neptune.ai/blog/explainability-auditability-ml-definitions-techniques-tools)


* [Module for incorporating model interpretatbility for PyTorch models](https://github.com/pytorch/captum)

## Few key work horses: 

### 1. SHAP

SHAP (SHapley Additive exPlanations) is a game theoretic approach to explain the output of any machine learning model.[Github](https://github.com/slundberg/shap)


Considering cooperative prediction - the value added by the feature contribution to the final output and compare it to its individual output if that feature was active. 

Given the current set of feature values, the contribution of a feature value to the difference between the actual prediction and the mean prediction is the estimated Shapley value.

### 2. Counterfactuals 

> If X had not occured, would have Y occured? 

A counterfactual is a idea of relating an action to a consequence. 

"Would I have got a cold, if I had not eaten the ice-cream?"

Usually a model agnostic approach is implemented, wherein the input(s) of the model is varied and its effect on the prediction is analyzed. Mind here  that we dont really care __how__ the model predicts tthe output but __just__ if the output changes by changing the input. 

The idea echoes with the concept of [**degree of rate control**](https://pubs.acs.org/doi/10.1021/acscatal.7b00115) first proposed by Charles Campbell to propose kinetic pathways and intermediates which have most impact on the final chemical reaction rate. 

Counterfactual have an important drawback - they suffer from the possibility of multiple truths. 
Explain on why that molecule: Andrew White Lab.
[Github](https://ur-whitelab.github.io/exmol/)

[Pen's blog on implementation of Exmol](https://iwatobipen.wordpress.com/2021/09/18/try-to-use-exmol-to-explain-why-the-model-predicts-it-chemoinfomratics-rdkit-exmol/?utm_source=pocket_mylist)


[ExplainerDashboard python package](https://towardsdatascience.com/understanding-machine-learning-models-better-with-explainable-ai-bea6d28f5094)

[Automate model design and NN architecture search?](https://arxiv.org/abs/1802.03268)

### 3. Canonical Correlation Analysis

[Video](https://www.youtube.com/watch?v=u7Dvb_a1D-0) from Jay Alammar.  
