---
toc: true
sticky_rank: 3
layout: post
description: List of (fairly) recent articles, resources, and blogs that I've found useful to learn about Cheminformatics
categories: [chemical-science, machine-learning, resources]
title: Cheminformatics Literature and Resources
---

Last update: 3rd July 2021

## Noteworthy blogs to follow:

1. [Patrick Walters Blog on Cheminformatics](https://practicalcheminformatics.blogspot.com/2021/01/ai-in-drug-discovery-2020-highly.html)
    * [Pat Walter's Cheminformatics Resources list](https://github.com/PatWalters/resources/blob/main/cheminformatics_resources.md)
    
2. [Is Life Worth Living](https://iwatobipen.wordpress.com/)

3. [Andrew White's ML for Molecules and Materials Online Book](https://whitead.github.io/dmol-book/intro.html)

4. [Cheminformia](http://www.cheminformania.com)

5. [Depth-First](https://depth-first.com)

6. [DrugDiscovery.NET - Andreas Bender](http://www.drugdiscovery.net/2019/07/25/new-post1/)

## Online resources 

* [Pen's Python cookbook for Cheminformatics](https://github.com/iwatobipen/py4chemoinformatics)

* [Patrick Walter's Cheminformatics Hands-on workshop](https://github.com/PatWalters/workshop)

* [Andrea Volkmer, TeachOpenCADD: a teaching platform for computer-aided drug design (CADD)](https://github.com/volkamerlab/TeachOpenCADD)

* [Chem LibreText collection from ACS Division of Chemical Education](https://bit.ly/2SxItoc)

## Reviews:

* [F. Strieth-Kalthoff, F. Sandfort, M. H. S. Segler, and F. Glorius, Machine learning the ropes: principles, applications and directions in synthetic chemistry, Chem. Soc. Rev](https://pubs.rsc.org/en/content/articlelanding/2020/CS/C9CS00786E#fn1)

Pedagogical account of various machine learning techniques, models, representation schemes from perspective of synthetic chemistry. Covers different applications of machine learning in synthesis planning, property prediction, molecular design, and reactivity prediction

* [Mariia Matveieva & Pavel Polishchuk. Benchmarks for interpretation of QSAR models](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00519-x). [Github](https://github.com/ci-lab-cz/ibenchmark). [Patrick Walter's blog on the paper](https://patwalters.github.io/practicalcheminformatics/jupyter/ml/interpretability/2021/06/03/interpretable.html).

Paper outlining good practices for interpretating QSAR (Quantative Structure-Property Prediction) models. Good set of heuristics and comparison in the paper in terms of model interpretability. Create 6 synthetic datasets with varying complexity for QSAR tasks. The authors compare interpretability of graph-based methods to conventional QSAR methods. In regards to performance graph-based models show low interpretation compared to conventional QSAR method. 

* [W. Patrick Walters & Regina Barzilay. Applications of Deep Learning in Molecule Generation and Molecular Property Prediction](https://pubs.acs.org/doi/full/10.1021/acs.accounts.0c00699)

Recent review summarising the state of the molecular property prediction and structure generation research. In spite of exciting recent advances in the modeling efforts,  there is a need to generate better (realistic)  training data, assess model prediction confidence, and metrics to quantify molecular generation performance. 

* [Navigating through the Maze of Homogeneous Catalyst Design with Machine Learning](https://chemrxiv.org/articles/preprint/Navigating_through_the_Maze_of_Homogeneous_Catalyst_Design_with_Machine_Learning/12786722/1)

* [Coley, C. W. Defining and Exploring Chemical Spaces. Trends in Chemistry 2020](https://doi.org/10.1016/j.trechm.2020.11.004)

* [Applications of Deep learning in molecular generation and molecular property prediction](https://pubs.acs.org/doi/abs/10.1021/acs.accounts.0c00699)


* [Utilising Graph Machine Learning within Drug Discovery and Development](https://arxiv.org/pdf/2012.05716.pdf)


## Industry-focused drug discovery reviews 

* [A. Bender and I. Cortés-Ciriano, “Artificial intelligence in drug discovery: what is realistic, what are illusions? Part 1: Ways to make an impact, and why we are not there yet,” Drug Discov. Today, vol. 26, no. 2, pp. 511–524, 2021](https://www.sciencedirect.com/science/article/pii/S1359644620305274)

* [A. H. Göller et al., “Bayer’s in silico ADMET platform: a journey of machine learning over the past two decades,” Drug Discov. Today, vol. 25, no. 9, pp. 1702–1709, 2020.](https://www.sciencedirect.com/science/article/pii/S1359644620302609)

* [J. Shen and C. A. Nicolaou, “Molecular property prediction: recent trends in the era of artificial intelligence,” Drug Discov. Today Technol., vol. 32–33, no. xx, pp. 29–36, 2019.](https://www.sciencedirect.com/science/article/abs/pii/S1740674920300032)

## Special Journal Issues: 

1. [Nice collection of recent papers in Nature Communications on ML application and modeling](https://www.nature.com/collections/gcijejjahe)

2. [Journal of Medicinal Chemistry compendium of AI in Drug discovery issue](https://pubs.acs.org/doi/full/10.1021/acs.jmedchem.0c01077)

3. [Account of Chemical Research Special Issue on advances in data-driven chemistry research](https://pubs.acs.org/page/achre4/data-science-meets-chemistry)

## Specific Articles 

Few key papers which I have found useful when learning more about the state-of-the-art in Cheminformatics. I've tried to categorize them roughly based on their area of application: 

### Representation:

* [Representation of Molecules in NN: Molecular representation in AI-driven drug discovery: review and guide](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-020-00460-5)

* [Screening of energetic molecules -- comparing different representations](https://www.nature.com/articles/s41598-018-27344-x)

* [M. Krenn, F. Hase, A. Nigam, P. Friederich, and A. Aspuru-Guzik, “Self-Referencing Embedded Strings (SELFIES): A 100% robust molecular string representation,” Mach. Learn. Sci. Technol., pp. 1–9, 2020](https://arxiv.org/abs/1905.13741)

* [Could graph neural networks learn better molecular representation for drug discovery? A comparison study of descriptor-based and graph-based models](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-020-00479-8)

Comparative study of descriptor-based and graph-based models using public data set. Used descriptor-based models (XGBoost, RF, SVM, using ECFP) and compared them to graph-based models (GCN, GAT, AttentiveFP, MPNN). They show descriptor-based models outperform the graph-based models in terms of prediction accuracy and computational efficiency with SVM having best predictions. Graph-based methods are good for multi-task learning. 

* [Matthew Clark, et. al. DNA-encoded small-molecule libraries (DEL)](https://www.nature.com/articles/nchembio.211). [C&EN article on the topic](https://cen.acs.org/articles/95/i25/DNA-encoded-libraries-revolutionizing-drug.html)

New form of storing huge amounts of molecule related data using DNA. Made partially possible by low cost of DNA sequencing. Each molecule in the storage is attached with a DNA strand which encode information about its recipe. 

* Follow up to the work with Machine Learning for hit finding. [(Link)](https://pubs.acs.org/doi/abs/10.1021/acs.jmedchem.0c00452)

DNA encodings for discovery of novel small-molecule protein inhibitors. Outline a process for building a ML model using DEL. Compare graph convolutions to random forest for classification tasks with application to protein target binding. Graph models seemed to achieve high hit rate comapred to random forest. Apply diversity, logistical, structural filtering to search for novel candidates. 

### Uncertainty quantification:

* [Alan Aspuru-Guzik perspective on uncertainty and confidence](https://arxiv.org/pdf/2102.11439.pdf)

* [Uncertainty Quantification Using Neural Networks for Molecular Property Prediction. J. Chem. Inf. Model. (2020) Hirschfeld, L., Swanson, K., Yang, K., Barzilay, R. & Coley, C. W.](10.1021/acs.jcim.0c00502)

Benchmark different models and uncertainty metrics for molecular property prediction. 

* [Evidential Deep learning for guided molecular property prediction and disocovery Ava Soleimany, Conor Coley, et. al.](https://arxiv.org/abs/1910.02600). [Slides](https://slideslive.com/38942396/evidential-deep-learning-for-guided-molecular-property-prediction-and-discovery)

Train network to output the parameters of an evidential distribution. One forward-pass to find the uncertainty as opposed to dropout or ensemble - principled incorporation of uncertainties

* [Differentiable sampling of molecular geometries with uncertainty-based adversarial attacks](https://arxiv.org/pdf/2101.11588.pdf)

* [J. P. Janet, S. Ramesh, C. Duan, H. J. Kulik, ACS Cent. Sci. 2020](https://pubs.acs.org/doi/10.1021/acscentsci.0c00026)

Conduct a global multi-objective optimization with expected improvement criterion. Find transition metal complex redox couples for Redox flow batteries that address stability, solubility, and redox potential metric. Use distance of a point from a training data in latent space as a metric to quantify uncertainty. 

* [J. P. Janet, C. Duan, T. Yang, A. Nandy, H. J. Kulik, Chem. Sci. 2019, 10, 7913–7922](https://pubs.rsc.org/en/content/articlelanding/2019/sc/c9sc02298h#!divAbstract)

Distance from available data in NN latent space is used as a variable for low-cost, quantitative uncertainty metric that works for both inorganic and organic chemistry. Introduce a technique to calibrate latent distances enabling conversion of distance-based metric to error estimates in units of predicted property 


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

**Reviews** 

* [Mouchlis VD, Afantitis A, Serra A, et al. Advances in de Novo Drug Design: From Conventional to Machine Learning Methods. Int J Mol Sci. 2021;22(4):1676. Published 2021 Feb 7. doi:10.3390/ijms22041676](https://pubmed.ncbi.nlm.nih.gov/33562347/)

* [B. Sanchez-Lengeling and A. Aspuru-Guzik, “Inverse molecular design using machine learning: Generative models for matter engineering,” Science (80)., vol. 361, no. 6400, pp. 360–365, Jul. 2018](https://science.sciencemag.org/content/361/6400/360)

**Benchmarks**

* [MOSES - Benchmarking platform for generative models](https://arxiv.org/abs/1811.12823).

Propose a platform to deploy and compare state-of-the-art generative models for exploring molecular space on same dataset. In addition the authors also propose list of metrics  to evaluate the quality and diversity of the generated structures.  

* [J. Zhang, R. Mercado, O. Engkvist, and H. Chen, “Comparative Study of Deep Generative Models on Chemical Space Coverage,” J. Chem. Inf. Model., vol. 61, no. 6, pp. 2572–2581, Jun. 2021.](https://pubs.acs.org/doi/10.1021/acs.jcim.0c01328)

Interesting analysis from team at AstraZeneca R&D. They look at the chemical space coverage accounted by the SOTA generative models. Proposes a metric for evaluating space coverage, and thereby comparing different SOTA models, using a reference data (GDB-13 in this case). The new metric computes how much of the GDB-13 dataset can be recovered by a model that is trained on small GDB subset.  Generative models were trained on same 1M data points and 1B molecules were then sampled from each model. It was seen that at most 39% of the molecules in the parent dataset were sampled / generated by the model. Most models sampled the same compounds atleast twice. It was observed that graph-based model sampled much diverse molecules than string-based methods. Besides, the coverage of GAN-based models was worse compared to Language and Graph models. 

* [Gao, W.; Coley, C. W. The Synthesizability of Molecules Proposed by Generative Models. J. Chem. Inf. Model. 2020](https://doi.org/10.1021/acs.jcim.0c00174)

This paper looks at different ways of integrating synthesizability criteria into generative models. 


**Language models:**

* [R. Gómez-Bombarelli et al., “Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules,” ACS Cent. Sci., vol. 4, no. 2, pp. 268–276, 2018](https://pubs.acs.org/doi/10.1021/acscentsci.7b00572)

One of the first implementation of a variation auto-encoder for molecule generation

* [Penalized Variational Autoencoder](https://s3-eu-west-1.amazonaws.com/itempdf74155353254prod/7977131/Penalized_Variational_Autoencoder_for_Molecular_Design_v2.pdf)

* [SELFIES and generative models using STONED](https://chemrxiv.org/articles/preprint/Beyond_Generative_Models_Superfast_Traversal_Optimization_Novelty_Exploration_and_Discovery_STONED_Algorithm_for_Molecules_using_SELFIES/13383266)

Representation using SELFIES proposed to make it much more powerful

* [LSTM based (RNN) approaches to small molecule generation](https://s3-eu-west-1.amazonaws.com/itempdf74155353254prod/10119299/Generating_Customized_Compound_Libraries_for_Drug_Discovery_with_Machine_Intelligence_v1.pdf). [Github](https://github.com/ETHmodlab/BIMODAL)

* [Chithrananda, S.; Grand, G.; Ramsundar, B. ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction. arXiv [cs.LG], 2020](https://arxiv.org/abs/2010.09885).

* [SMILES-based deep generative scaffold decorator for de-novo drug design](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-020-00441-8#availability-of-data-and-materials). [Github](https://github.com/undeadpixel/reinvent-randomized)

SMILES-based language model that generates molecules from scaffolds and can be trained from any arbitrary molecular set. Uses randomized SMILES to improve final prediction validity. 

**Graph-based**

* [W. Jin, R. Barzilay, and T. Jaakkola, “Junction tree variational autoencoder for molecular graph generation,” 35th Int. Conf. Mach. Learn. ICML 2018, vol. 5, pp. 3632–3648, 2018](https://arxiv.org/abs/1802.04364)

Junction tree based decoding. Define a grammar for the small molecule and find sub-units based on that grammar to construct a molecule. The molecule is generated in two-steps: first being generating the scaffold or backbone of the molelcule, then the nodes  are added with molecular substructure as identified from the 'molecular grammar'. 

* [MPGVAE: Message passing graph networks for molecular generation, Daniel Flam-Shepherd et al 2021 Mach. Learn.: Sci. Technol.](https://iopscience.iop.org/article/10.1088/2632-2153/abf5b7/pdf)

Introduce a graph generation model by building a Message Passing Neural Network (MPNNs) into the encoder and decoder of a VAE (MPGVAE).

* [ConfVAE: End-to-end framework for molecular conformation generation via bilevel programming](https://arxiv.org/pdf/2105.07246.pdf)

Algorithm to predict 3D conforms from molecular graphs.


* [GraphINVENT: R. Mercado, T. Rastemo, E. Lindelöf, G. Klambauer and O. Engkvist, “Graph networks for molecular design,” Mach. Learn. Sci. Technol., vol. 2, no. 2, p. 25023, 2021](https://iopscience.iop.org/article/10.1088/2632-2153/abcf91/meta). [Github](https://github.com/MolecularAI/GraphINVENT). [Blogpost](https://www.cheminformania.com/using-graphinvent-to-generate-novel-drd2-actives/)

GraphINVENT uses a tiered deep neural network architecture to probabilistically generate new molecules a single bond at a time 

**GANs**

* [MolGAN: An implicit generative model for small molecular graphs, N. De Cao and T. Kipf, 2018](https://arxiv.org/abs/1805.11973)

Generative adversarial network for finding small molecules using graph networks, quite interesting. Avoids issues arising from node ordering that are associated with likelihood based methods by using an adversarial loss instead (GAN)

* [LatentGAN: A de novo molecular generation method using latent vector based generative adversarial network](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-019-0397-9)

Molecular generation strategy is described which combines an autoencoder and a GAN. Generator and discriminator network do not use SMILES strings as input, but instead n-dimensional vectors derived from the code-layer of an autoencoder trained as a SMILES heteroencoder that way syntax issues are expected to be addressed. 

### Reaction Network Predictions: 

Recent review on the matter from Reiher group: 

* [The Exploration of Chemical Reaction Networks](https://arxiv.org/pdf/1906.10223.pdf)

Perspective article summarising their position on the current state of research and future considerations on developing better reaction network models. Break down the analysis of reaction networks as into 3 classes (1) Front Open End: exploration of products from reactants (2) Backward Open Start: Know the product and explore potential reactants (3) Start to End: Product and reactant known, explore the likely intermediates. 

Nice summary of potential challenges in the field: 

- Validating exploration algorithms on a consistent set of reaction system.
Need to generate a comparative metric to benchmark different algorithms.  
- Considering effect of solvents and/or protein embeddings in the analysis

* Previous review article by same group: [Exploration of Reaction Pathways and Chemical Transformation Networks](https://pubs.acs.org/doi/pdf/10.1021/acs.jpca.8b10007)

Technical details of various algorithms being implemented for reaction mechanism discovery at the time of writing the review. 

Articles: 

* [M. Liu et al., “Reaction Mechanism Generator v3.0: Advances in Automatic Mechanism Generation,” J. Chem. Inf. Model., May 2021](https://pubs.acs.org/doi/10.1021/acs.jcim.0c01480)

Newest version of RMG (v3) is updated to Python v3. It has ability to generate heterogeneous catalyst models, uncertainty analysis to conduct first order sensitivity analysis. RMG dataset for the thermochemical and kinetic parameters have been expanded. 

* [More and Faster: Simultaneously Improving Reaction Coverage and Computational Cost in Automated Reaction Prediction Tasks](https://s3-eu-west-1.amazonaws.com/itempdf74155353254prod/13076087/More_and_Faster__Simultaneously_Improving_Reaction_Coverage_and_Computational_Cost_in_Automated_Reaction_Prediction_Task_v1.pdf)

Presents an algorithmic improvement to the reaction network prediction task through their YARP (Yet Another Reaction Program) methodology. Shown to reduce computational cost of optimization while improving the diversity of identified products and reaction pathways. 

* [Molecular Transformer: A Model for Uncertainty-Calibrated Chemical Reaction Prediction](https://pubs.acs.org/doi/abs/10.1021/acscentsci.9b00576)
    * Follow-up: [Quantitative interpretation explains machine learning models for chemical reaction prediction and uncovers bias](https://www.nature.com/articles/s41467-021-21895-w)

* [Automatic discovery of chemical reactions using imposed activation](https://chemrxiv.org/articles/preprint/Automatic_discovery_of_chemical_reactions_using_imposed_activation/13008500/1)

* [Machine learning in chemical reaction space](https://www.nature.com/articles/s41467-020-19267-x)

Look at exploration of reaction space rather than compound space. SOAP kernel for representing the moelcules. Estimate atomization energy for the molecules using ML. Calculate the d(AE) for different ML-estimated AEs. Reaction energies (RE) are estimated and uncertainty propogation is used to estimate the errors. Uncorrelated constant error propogation. 30,000 bond breaking reaction steps Rad-6-RE network used. RE prediction is not as good as AE. 

* [C. W. Coley et al., “A graph-convolutional neural network model for the prediction of chemical reactivity,” Chem. Sci., vol. 10, no. 2, pp. 370–377, 2019.](https://pubs.rsc.org/en/content/articlepdf/2019/sc/c8sc04228d)

* [Prediction of Organic Reaction Outcomes Using Machine Learning, ACS Cent. Sci. 2017](10.1021/acscentsci.7b00064)




## Code / Packages:

* [Molecular AI department at AstraZeneca R&D](https://github.com/MolecularAI)

* [GHOST: Generalized threshold shifting procedure](https://github.com/rinikerlab/GHOST). [Paper](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00160
)

Automates the selection of decision threshold for imbalanced classification task. The assumption for this method to work is the similar characteristics (like imbalance ratio) of training and test data. 

* [MOSES - Benchmarking platform for generative models (PyTorch Implementation)](https://arxiv.org/abs/1811.12823). [Github](https://github.com/molecularsets/moses)

Benchmarking platform to implement molecular generative models. It also provides a set of metrics to evaluate the quality and diversity of the generated molecules. A benchmark dataset (subset of ZINC) is provided for training the models. 

* [Reinvent 2.0 - an AI tool forr de novo drug design](https://chemrxiv.org/articles/preprint/REINVENT_2_0_an_AI_Tool_for_De_Novo_Drug_Design/12058026/1). [Github](https://github.com/MolecularAI/Reinvent)

Production-ready tool for de novo design from Astra Zeneca. It can be effectively applied on drug discovery projects that are striving to resolve either exploration or exploitation problems while navigating the chemical space. Language model with SMILE  output and trained by “randomizing” the SMILES representation of the input data. Implement reinforcement-leraning for directing the model towards relevant area of interest. 

* [Schnet by Jacobsen et. al. (Neural message passing)](https://arxiv.org/abs/1806.03146). [Github](https://github.com/atomistic-machine-learning/G-SchNet). [Tutorial](https://schnetpack.readthedocs.io/en/stable/tutorials/tutorial_03_force_models.html)

* [OpenChem](https://chemrxiv.org/articles/OpenChem_A_Deep_Learning_Toolkit_for_Computational_Chemistry_and_Drug_Design/12691943/1). [Github](https://github.com/Mariewelt/OpenChem)

* [DeepChem (Tensorflow)](https://github.com/deepchem/deepchem). [Website](https://deepchem.io) 

DeepChem aims to provide a high quality open-source toolchain that democratizes the use of deep-learning in drug discovery, materials science, quantum chemistry, and biology - from Github

* [ChemProp (Pytorch)](https://github.com/chemprop/chemprop)

Github repository for implmenting message passing neural networks for molecular property prediction as described in the paper [Analyzing Learned Molecular Representations for Property Prediction](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.9b00237) by Yang et. al. 

* [Chainer-Chemistry](https://github.com/chainer/chainer-chemistry)

"Chainer Chemistry is a deep learning framework (based on Chainer) with applications in Biology and Chemistry. It supports various state-of-the-art models (especially GCNN - Graph Convolutional Neural Network) for chemical property prediction" - from their Github repo introduction

* [FastJTNN - python 3 version of the JT-NN](https://github.com/Bibyutatsu/FastJTNNpy3)

* [DimeNet++  - extension of Directional message pasing working (DimeNet)](https://arxiv.org/abs/2003.03123). [Github](https://github.com/klicperajo/dimenet)

* [BondNet - Graph neural network model for predicting bond dissociation energies, considers both homolytic and heterolytic bond breaking]. [Github](https://github.com/mjwen/bondnet)

* [PhysNet](https://arxiv.org/pdf/1902.08408.pdf)

* [RNN based encoder software](https://github.com/ETHmodlab/BIMODAL)

* [AutodE](https://duartegroup.github.io/autodE/)

* [DScribe](https://singroup.github.io/dscribe/latest/)

* [RMG - Reaction Mechanism Generator](https://github.com/ReactionMechanismGenerator/RMG-Py)

Tool to generate chemical reaction networks. Includes Arkane, package for calculating thermodynamics from quantum mechanical calculations. 

## Helpful utilities:

* [RD-Kit](https://github.com/rdkit/rdkit)
    * [Get Atom Indices in the SMILE:](https://colab.research.google.com/drive/16T6ko0YE5WqIRzL4pwW_nufTDn7F3adw)
    * [Datamol for manipulating RDKit molecules](https://github.com/datamol-org/datamol)


* [Papers with code benchmark for QM9 energy predictions](https://paperswithcode.com/sota/formation-energy-on-qm9)

* [MOSES: Molecular generation models benchmark](https://github.com/molecularsets/moses)

* [Therapeutics Data Commons](https://tdcommons.ai)
"Therapeutics Data Commons is an open-science platform with AI/ML-ready datasets and learning tasks for therapeutics, spanning the discovery and development of safe and effective medicines. TDC also provides an ecosystem of tools, libraries, leaderboards, and community resources, including data functions, strategies for systematic model evaluation, meaningful data splits, data processors, and molecule generation oracles. All resources are integrated and accessible via an open Python library."


## Molecules datasets:

* [GDB Dataset](http://www.gdb.unibe.ch/downloads/)

* [Quantum Machine: Website listing useful datasets including QM9s and MD trajectory](http://quantum-machine.org/datasets/)

* [Github repository listing databases for Drug Discovery](https://github.com/LeeJunHyun/The-Databases-for-Drug-Discovery)
