---
toc: true
layout: post
description: List of (fairly) recent articles, resources, and blogs that I've found useful to learn about Cheminformatics
categories: [chemistry, machine-learning, resources]
title: Cheminformatics Literature and Resources
---

## Noteworthy blogs to follow:

1. [Patrick Walters Blog on Cheminformatics](https://practicalcheminformatics.blogspot.com/2021/01/ai-in-drug-discovery-2020-highly.html)
    * [Pat Walter's Cheminformatics Resources list](https://github.com/PatWalters/resources/blob/main/cheminformatics_resources.md)
    * [Cheminformatics Hands-on workshop](https://github.com/PatWalters/workshop)

2. [Is Life Worth Living](https://iwatobipen.wordpress.com/)
    * [Very helpful cookbook on Python for Cheminformatics](https://github.com/iwatobipen/py4chemoinformatics)

3. [Andrew White's ML for Molecules and Materials Online Book](https://whitead.github.io/dmol-book/intro.html)

4. [Cheminformia](http://www.cheminformania.com)

5. [Depth-First](https://depth-first.com)

## Reviews:

1. [Navigating through the Maze of Homogeneous Catalyst Design with Machine Learning](https://chemrxiv.org/articles/preprint/Navigating_through_the_Maze_of_Homogeneous_Catalyst_Design_with_Machine_Learning/12786722/1)

2. [Coley, C. W. Defining and Exploring Chemical Spaces. Trends in Chemistry 2020](https://doi.org/10.1016/j.trechm.2020.11.004)

3. [Applications of Deep learning in molecular generation and molecular property prediction](https://pubs.acs.org/doi/abs/10.1021/acs.accounts.0c00699)

4. [Bayer's ADMET platform review](https://www.sciencedirect.com/science/article/pii/S1359644620302609)

5. [Utilising Graph Machine Learning within Drug Discovery and Development](https://arxiv.org/pdf/2012.05716.pdf)

6. [Machine Learning in Chemistry - Jon Paul Janet & Heather Kulik](https://books.google.com/books?hl=en&lr=&id=JObqDwAAQBAJ&oi=fnd&pg=PA1902&dq=info:lWNtugbBwAsJ:scholar.google.com&ots=cbiwVybtQW&sig=MTKJOvhlbt2sDujAZ5XECERmNl8)

## Special Journal Issues: 

1. [Nice collection of recent papers in Nature Communications on ML application and modeling](https://www.nature.com/collections/gcijejjahe)

2. [Journal of Medicinal Chemistry compendium of AI in Drug discovery issue](https://pubs.acs.org/doi/full/10.1021/acs.jmedchem.0c01077)

## Specific Articles 

Few key papers which I have found useful when learning more about the state-of-the-art in Cheminformatics. I've tried to categorize them roughly based on their area of application: 

### Representation:

* [Representation of Molecular in NN: Molecular representation in AI-driven drug discovery: review and guide](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-020-00460-5)

* [Screening of energetic molecules -- comparing different representations](https://www.nature.com/articles/s41598-018-27344-x)

* [M. Krenn, F. Hase, A. Nigam, P. Friederich, and A. Aspuru-Guzik, “Self-Referencing Embedded Strings (SELFIES): A 100% robust molecular string representation,” Mach. Learn. Sci. Technol., pp. 1–9, 2020](https://arxiv.org/abs/1905.13741)



### Uncertainty quantification:

* [Alan Aspuru-Guzik perspective on uncertainty and confidence](https://arxiv.org/pdf/2102.11439.pdf)

* [Uncertainty Quantification Using Neural Networks for Molecular Property Prediction. J. Chem. Inf. Model. (2020) Hirschfeld, L., Swanson, K., Yang, K., Barzilay, R. & Coley, C. W.](10.1021/acs.jcim.0c00502)

Benchmark different models and uncertainty metrics for molecular property prediction. 

* [Evidential Deep learning for guided molecular property prediction and disocovery Ava Soleimany, Conor Coley, et. al.](https://arxiv.org/abs/1910.02600). [Slides](https://slideslive.com/38942396/evidential-deep-learning-for-guided-molecular-property-prediction-and-discovery)

Train network to output the parameters of an evidential distribution. One forward-pass to find the uncertainty as opposed to dropout or ensemble - principled incorporation of uncertainties

* [Differentiable sampling of molecular geometries with uncertainty-based adversarial attacks](https://arxiv.org/pdf/2101.11588.pdf)

### Active Learning 

Active learning provides strategies for efficient screening of subsets of the library. In many cases, we can identify a large portion of the most promising molecules with a fraction of the compute cost.

* [Reker, D. Practical Considerations for Active Machine Learning in Drug Discovery. Drug Discov. Today Technol. 2020](https://doi.org/10.1016/j.ddtec.2020.06.001)

* [B. J. Shields et al., “Bayesian reaction optimization as a tool for chemical synthesis,” Nature, vol. 590, no. June 2020, p. 89, 2021](https://www.nature.com/articles/s41586-021-03213-y). [Github](https://github.com/b-shields/edbo)

Experimental design using Bayesian Optimization. 

### Transfer Learning  

* [Approaching coupled cluster accuracy with a general-purpose neural network potential through transfer learning](https://www.nature.com/articles/s41467-019-10827-4)
Transfer learning by training a network to DFT data and then retrain on a dataset of gold standard QM calculations (CCSD(T)/CBS) that optimally spans chemical space. The resulting potential is broadly applicable to materials science, biology, and chemistry, and billions of times faster than CCSD(T)/CBS calculations.

* [Improving the generative performance of chemical autoencoders through transfer learning](https://iopscience.iop.org/article/10.1088/2632-2153/abae75/meta)


### Generative models:

[B. Sanchez-Lengeling and A. Aspuru-Guzik, “Inverse molecular design using machine learning: Generative models for matter engineering,” Science (80-. )., vol. 361, no. 6400, pp. 360–365, Jul. 2018](https://science.sciencemag.org/content/361/6400/360)

- Research Articles:

* [R. Gómez-Bombarelli et al., “Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules,” ACS Cent. Sci., vol. 4, no. 2, pp. 268–276, 2018](https://pubs.acs.org/doi/10.1021/acscentsci.7b00572)

One of the first implementation of a variation auto-encoder for molecule generation

* [Penalized Variational Autoencoder](https://s3-eu-west-1.amazonaws.com/itempdf74155353254prod/7977131/Penalized_Variational_Autoencoder_for_Molecular_Design_v2.pdf)

* [SELFIES and generative models using STONED](https://chemrxiv.org/articles/preprint/Beyond_Generative_Models_Superfast_Traversal_Optimization_Novelty_Exploration_and_Discovery_STONED_Algorithm_for_Molecules_using_SELFIES/13383266)

Representation using SELFIES proposed to make it much more powerful

* [W. Jin, R. Barzilay, and T. Jaakkola, “Junction tree variational autoencoder for molecular graph generation,” 35th Int. Conf. Mach. Learn. ICML 2018, vol. 5, pp. 3632–3648, 2018](https://arxiv.org/abs/1802.04364)

Junction tree based decoding. Define a grammar for the small molecule and find sub-units based on that grammar to construct a molecule

* [N. De Cao and T. Kipf, “MolGAN: An implicit generative model for small molecular graphs,” 2018](https://arxiv.org/abs/1805.11973)

Generative adversarial network for finding small molecules using graph networks, quite interesting

* [Message passing graph networks for molecular generation](https://iopscience.iop.org/article/10.1088/2632-2153/abf5b7/pdf)

**Language models:**

* [LSTM based (RNN) approaches to small molecule generation](https://s3-eu-west-1.amazonaws.com/itempdf74155353254prod/10119299/Generating_Customized_Compound_Libraries_for_Drug_Discovery_with_Machine_Intelligence_v1.pdf). [Github](https://github.com/ETHmodlab/BIMODAL)

* [Chithrananda, S.; Grand, G.; Ramsundar, B. ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction. arXiv [cs.LG], 2020](https://arxiv.org/abs/2010.09885).

* [SMILES-based deep generative scaffold decorator for de-novo drug design](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-020-00441-8#availability-of-data-and-materials). [Github](https://github.com/undeadpixel/reinvent-randomized)

**Synthesizability Criteria into Generative Models:**

* [Gao, W.; Coley, C. W. The Synthesizability of Molecules Proposed by Generative Models. J. Chem. Inf. Model. 2020](https://doi.org/10.1021/acs.jcim.0c00174)
Paper looks at different ways of integrating synthesizability criteria into generative models.


### Reaction Network Predictions: 

* [Prediction of Organic Reaction Outcomes Using Machine Learning, ACS Cent. Sci. 2017](10.1021/acscentsci.7b00064)

* [Molecular Transformer: A Model for Uncertainty-Calibrated Chemical Reaction Prediction](https://pubs.acs.org/doi/abs/10.1021/acscentsci.9b00576)
    * Follow-up: [Quantitative interpretation explains machine learning models for chemical reaction prediction and uncovers bias](https://www.nature.com/articles/s41467-021-21895-w)

* [Automatic discovery of chemical reactions using imposed activation](https://chemrxiv.org/articles/preprint/Automatic_discovery_of_chemical_reactions_using_imposed_activation/13008500/1)

* [Machine learning in chemical reaction space](https://www.nature.com/articles/s41467-020-19267-x)

## Code / Packages:

* [Schnet by Jacobsen et. al. (Neural message passing)](https://arxiv.org/abs/1806.03146). [Github](https://github.com/atomistic-machine-learning/G-SchNet). [Tutorial](https://schnetpack.readthedocs.io/en/stable/tutorials/tutorial_03_force_models.html)

* [OpenChem](https://chemrxiv.org/articles/OpenChem_A_Deep_Learning_Toolkit_for_Computational_Chemistry_and_Drug_Design/12691943/1). [Github](https://github.com/Mariewelt/OpenChem)

* [DeepChem](https://github.com/deepchem/deepchem)

* [DimeNet++  -- extension of Directional message pasing working (DimeNet)](https://arxiv.org/abs/2003.03123). [Github](https://github.com/klicperajo/dimenet)

* [PhysNet](https://arxiv.org/pdf/1902.08408.pdf)

* [RNN based encoder software](https://github.com/ETHmodlab/BIMODAL)

* [AutodE](https://duartegroup.github.io/autodE/)

* [DScribe](https://singroup.github.io/dscribe/latest/)

## Helpful utilities:

* [RD-Kit](https://github.com/rdkit/rdkit)
    * [Get Atom Indices in the SMILE:](https://colab.research.google.com/drive/16T6ko0YE5WqIRzL4pwW_nufTDn7F3adw)
    * [Datamol for manipulating RDKit molecules](https://github.com/datamol-org/datamol)


* [Papers with code benchmark for QM9 energy predictions](https://paperswithcode.com/sota/formation-energy-on-qm9)

* [Molecular generation models benchmark](https://github.com/molecularsets/moses)

## Molecules datasets:

* [GDB Dataset](http://www.gdb.unibe.ch/downloads/)

* [Quantum Machine: Website listing useful datasets including QM9s and MD trajectory](http://quantum-machine.org/datasets/)

* [Github repository listing databases for Drug Discovery](https://github.com/LeeJunHyun/The-Databases-for-Drug-Discovery)
