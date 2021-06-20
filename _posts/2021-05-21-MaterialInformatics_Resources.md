---
toc: true
sticky_rank: 2
layout: post
description: List of resources and state-of-the-art for Material Informatics
categories: [chemical-science, machine-learning, resources]
title: Material-informatics Literature and Resources
---

Last update: 28th May 2021

Material Informatics is the solid-state, inorganic chemistry focused cousin to its organic chemistry contemporary: Cheminformatics. In spirit, the aim of Material Informatics is similar to Cheminformatics; it offers a promising avenue to augment traditional material R&D processes. Amplify the conventional material discovery task using data, analytics, and identify chemical spaces, and structure in the data, which are interesting and probe those rigorously using first-principles techniques and/or experimentation. 

The potential application of material informatics can be seen in:  Microelectronics, aerospace, and automotive to defense, clean energy, and health services, where ever there's a demand for new advanced materials at even greater rates and lower costs. 

**Application of material informatics in atomic-scale modeling:**

In case of molecular-level modeling of material properties, concepts developed in material informatics, statistics, and ML can be used for: 

1. Descriptor driven screening of computational models 

2. Discover new science and relations from large computational datasets 

3. Applying surrogate models to enable fast materials development

4. Undertake global optimization routines using surrogate models for composition and property predictions. 

Machine learning in atomic-scale modeling is often used to replace expensive ab initio methods with cheaper approximations. While certainly lucractive an additional consideration for ML use-case is its utility as a surrogate model to help researchers identify interesting regions in the material space. It also helps to decode the 'intuition' and serendipity involved in material development and hopefully provide a rigorous data driven basis for a design decision.

Below are few reviews, articles, and resources I've found that document the state-of-the-art for material informatics. It goes without saying that this is a highly biased and a non-exhaustive listing of articles covering only the ones I've read. The idea with this document is to provide a starting point in understanding the general status of the field. 


## Special Issues: 

* [Nature Materials collection of review articles discussing the role of computation for material design](https://www.nature.com/collections/dhcfgffecf)


* [Matter journal's Material prediction using data and ML prediction](https://www.cell.com/matter/collections/computation-data-and-machine-learning)

* [Nature Communications compendium on ML for material modelling](https://www.nature.com/collections/gcijejjahe)

## Reviews:

1. C. Chen, Y. Zuo, W. Ye, X. Li, Z. Deng, and S. P. Ong, “A Critical Review of Machine Learning of Energy Materials,” Adv. Energy Mater., vol. 1903242, p. 1903242, Jan. 2020.

2. J. Schmidt, M. R. G. Marques, S. Botti, and M. A. L. Marques, “Recent advances and applications of machine learning in solid-state materials science,” npj Comput. Mater., vol. 5, no. 1, p. 83, Dec. 2019.

3. J. Noh, G. H. Gu, S. Kim, and Y. Jung, “Machine-enabled inverse design of inorganic solid materials: promises and challenges,” Chem. Sci., vol. 11, no. 19, pp. 4871–4881, 2020.

4. S. M. Moosavi, K. M. Jablonka, and B. Smit, “The Role of Machine Learning in the Understanding and Design of Materials,” J. Am. Chem. Soc., no. Figure 1, p. jacs.0c09105, Nov. 2020.

5. F. Häse, L. M. Roch, P. Friederich, and A. Aspuru-Guzik, “Designing and understanding light-harvesting devices with machine learning,” Nat. Commun., vol. 11, no. 1, pp. 1–11, 2020.

6. M. Moliner, Y. Román-Leshkov, and A. Corma, “Machine Learning Applied to Zeolite Synthesis: The Missing Link for Realizing High-Throughput Discovery,” Acc. Chem. Res., vol. 52, no. 10, pp. 2971–2980, 2019.

### Best practices in material informatics:

A. Y. T. Wang et al., “Machine Learning for Materials Scientists: An Introductory Guide toward Best Practices,” Chem. Mater., vol. 32, no. 12, pp. 4954–4965, 2020.

## Featurizations possible:

Similar to other machine-learning development efforts -- featurization or descriptors used to convert material entries in machine-readable format is crucial for the eventual performance of any statistical model. Over the years there has been tremendous progress in describing the periodic solid crystal structures. Some of the key articles I've liked are mentioned below:

**Reviews:**

* A. P. Bartók, R. Kondor, and G. Csányi, “On representing chemical environments,” Phys. Rev. B - Condens. Matter Mater. Phys., vol. 87, no. 18, pp. 1–16, 2013.

* A. Seko, H. Hayashi, K. Nakayama, A. Takahashi, and I. Tanaka, “Representation of compounds for machine-learning prediction of physical properties,” Phys. Rev. B, vol. 95, no. 14, pp. 1–11, 2017.

* K. T. Schütt, H. Glawe, F. Brockherde, A. Sanna, K. R. Müller, and E. K. U. Gross, “How to represent crystal structures for machine learning: Towards fast prediction of electronic properties,” Phys. Rev. B - Condens. Matter Mater. Phys., vol. 89, no. 20, pp. 1–5, 2014.

**Articles:**

**1. Composition based:**

* [L. Ward et al., “Including crystal structure attributes in machine learning models of formation energies via Voronoi tessellations,” Phys. Rev. B, vol. 96, no. 2, 2017](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.96.024104)

Predicting properties of crystalline compounds using a representation consisting of attributes derived from the Voronoi tessellation of its structure and composition based features is both twice as accurate as existing methods and can scale to large training set sizes. Also the representations are  insensitive to changes in the volume of a crystal, which makes it possible to predict the properties of the crystal without needing to compute the DFT-relaxed geometry as input. Random forrest algorithm used for the prediction 

* [A. Wang, S. Kauwe, R. Murdock, and T. Sparks, “Compositionally-Restricted Attention-Based Network for Materials Property Prediction.” 20-Feb-2020.](https://chemrxiv.org/articles/preprint/Compositionally-Restricted_Attention-Based_Network_for_Materials_Property_Prediction/11869026/1)

Using attention-based graph networks on material composition to predict material properties. 

* [Goodall, R.E.A., Lee, A.A. Predicting materials properties without crystal structure: deep representation learning from stoichiometry. Nat Commun 11, 6280 (2020)](https://www.nature.com/articles/s41467-020-19964-7)

Similar to the previous article in spirit, here authors use material composition to generate weighted graphs and predict material properties 

**2. Structural based:**

* T. Xie and J. C. Grossman, “Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties,” Phys. Rev. Lett., vol. 120, no. 14, p. 145301, 2018.

## Material modeling benchmark studies:

* [Bartel, C.J., Trewartha, A., Wang, Q. et al. A critical examination of compound stability predictions from machine-learned formation energies. npj Comput Mater 6, 97 (2020)](https://www.nature.com/articles/s41524-020-00362-y)

Investigate if ML models can distinguish materials wrt thermodynamic stability and not just formation energies. Learning formation energy from composition alone is fine for MAE and RMSE representations. Propose that graph-based methods reduce the MAE by roughly 50% compared with the best performing compositional model. Show that including structural information is advantageous when predicting formation energies.


* [A. J. Chowdhury, W. Yang, E. Walker, O. Mamun, A. Heyden, and G. A. Terejanu, “Prediction of Adsorption Energies for Chemical Species on Metal Catalyst Surfaces Using Machine Learning,” J. Phys. Chem. C, vol. 122, no. 49, pp. 28142–28150, 2018](https://pubs.acs.org/doi/10.1021/acs.jpcc.8b09284)

Consider various encoding scheme and machine learning models to predict single adsorbate binding energy for carbon-based adsorabtes on transition metal surfaces. They show linear methods and scaling relationship hold well compared to ML methods. They found that for ML models to succeed, it is not necessary to use advanced (geometric) coordinate-based descriptors; simple descriptors, such as bond count, can provide satisfactory results. As many catalysis and materials science problems require significant time to generate each data point, in many cases the ML models would need to work with a relatively small-sized dataset

* [Rosen, Andrew; Iyer, Shaelyn; Ray, Debmalya; Yao, Zhenpeng; Aspuru-Guzik, Alan; Gagliardi, Laura; et al. (2020): Machine Learning the Quantum-Chemical Properties of Metal–Organic Frameworks for Accelerated Materials Discovery with a New Electronic Structure Database. ChemRxiv. Preprint](https://chemrxiv.org/articles/preprint/Machine_Learning_the_Quantum-Chemical_Properties_of_Metal_Organic_Frameworks_for_Accelerated_Materials_Discovery_with_a_New_Electronic_Structure_Database/13147616/1?file=25304507)

## Articles:

There is a rich history of using statistical model and data mining for predicting bulk inorganic crystal properties. The review articles mentioned  in the above section discuss those areas quite nicely. In this section particularly focusses on papers looking at apply informatics to encode surfaces for modeling heterogeneous catalyst surfaces, which is fairly new and very active research direction: 

* Ma, X., Li, Z., Achenie, L.E.K., and Xin, H. (2015). Machine-learning-augmented chemisorption model for CO2 electroreduction catalyst screening. J. Phys. Chem. Lett. 6, 3528–3533.

* F. Liu, S. Yang, and A. J. Medford, “Scalable approach to high coverages on oxides via iterative training of a machine-learning algorithm,” ChemCatChem, vol. 12, no. 17, pp. 4317–4330, 2020.

* C. S. Praveen and A. Comas-Vives, “Design of an Accurate Machine Learning Algorithm to Predict the Binding Energies of Several Adsorbates on Multiple Sites of Metal Surfaces,” ChemCatChem, vol. n/a, no. n/a, 2020.

* Z. Li, L. E. K. Achenie, and H. Xin, “An Adaptive Machine Learning Strategy for Accelerating Discovery of Perovskite Electrocatalysts,” ACS Catal., vol. 10, no. 7, pp. 4377–4384, 2020.

* R. García-Muelas and N. López, “Statistical learning goes beyond the d-band model providing the thermochemistry of adsorbates on transition metals,” Nat. Commun., vol. 10, no. 1, p. 4687, Dec. 2019.

* M. Rueck, B. Garlyyev, F. Mayr, A. S. Bandarenka, and A. Gagliardi, “Oxygen Reduction Activities of Strained Platinum Core-Shell Electrocatalysts Predicted by Machine Learning,” J. Phys. Chem. Lett., 2020.

* W. Xu, M. Andersen, and K. Reuter, “Data-Driven Descriptor Engineering and Refined Scaling Relations for Predicting Transition Metal Oxide Reactivity,” ACS Catal., vol. 11, no. 2, pp. 734–742, Jan. 2021.

* Liu, F., Yang, S. & Medford, A. J. Scalable approach to high coverages on oxides via iterative training of a machine-learning algorithm. ChemCatChem 12, 4317–4330 (2020).

Graph-network based approaches for encoding and predicting surface binding energies:

* Back, S. et al. Convolutional Neural Network of Atomic Surface Structures to Predict Binding Energies for High-Throughput Screening of Catalysts. J. Phys. Chem. Lett. 10, 4401–4408 (2019)

* Lym, J., Gu, G. H., Jung, Y. & Vlachos, D. G. Lattice convolutional neural network modeling of adsorbate coverage effects. J. Phys. Chem. C 123, 18951–18959 (2019).

Adsorbate binding predictions have been recently extended to cover high-entropy alloy surfaces as well: 

* T. A. A. Batchelor et al., “Complex solid solution electrocatalyst discovery by computational prediction and high‐throughput experimentation,” Angew. Chemie Int. Ed., p. anie.202014374, Dec. 2020.

* J. K. Pedersen, T. A. A. Batchelor, D. Yan, L. E. J. Skjegstad, and J. Rossmeisl, “Surface electrocatalysis on high-entropy alloys,” Curr. Opin. Electrochem., vol. 26, p. 100651, Apr. 2021.

* Z. Lu, Z. W. Chen, and C. V. Singh, “Neural Network-Assisted Development of High-Entropy Alloy Catalysts: Decoupling Ligand and Coordination Effects,” Matter, vol. 3, no. 4, pp. 1318–1333, 2020.

## Global optimization methods: 

* M. K. Bisbo and B. Hammer, “Efficient global structure optimization with a machine learned surrogate model,” Phys. Rev. Lett., vol. 124, no. 8, p. 86102, 2019.

* J. Dean, M. G. Taylor, and G. Mpourmpakis, “Unfolding adsorption on metal nanoparticles: Connecting stability with catalysis,” Sci. Adv., vol. 5, no. 9, 2019.


## Uncertainty quantification (UQ): 

* [A. Wang et al., “A Framework for Quantifying Uncertainty in DFT Energy Corrections.” 19-May-2021](https://chemrxiv.org/articles/preprint/A_Framework_for_Quantifying_Uncertainty_in_DFT_Energy_Corrections/14593476)

Method to comment on the uncertainty of DFT errors which accounts for both sources of uncertainty: experimental and model parameters. Fit energy corrections using a set of 222 binary and ternary compounds for which experimental and computed values are present. Quantifying this uncertainty can help reveal cases wherein empirically-corrected DFT calculations are limited to differentiate between stable and unstable phases. Validate  this approach on Sc-W-O phase diagram analysis. 

* [Feng, J., Lansford, J. L., Katsoulakis, M. A., & Vlachos, D. G. (2020). Explainable and trustworthy artificial intelligence for correctable modeling in chemical sciences. Science advances, 6(42)](https://advances.sciencemag.org/content/6/42/eabc3204)

Propose Bayesian networks, type of probabilistic graphical models, to integrate physics- and chemistry-based data and uncertainty. Demonstrate this framework in searching for the optimal reaction rate and oxygen binding energy for the oxygen reduction reaction (ORR) using the volcano model. Their model is able to comment on the source of uncertainty in the model. 

* [K. Tran, W. Neiswanger, J. Yoon, Q. Zhang, E. Xing, and Z. W. Ulissi, “Methods for comparing uncertainty quantifications for material property predictions,” pp. 1–29, Dec. 2019](https://arxiv.org/abs/1912.10066)

Helpful overview and benchmark of various model flavors and metrics to understand ways of reporting the confidence in model predictions for material properties. Interesting convolution-Fed Gaussian Process (CFGP) model framework looked into which is a combination of CGCNN and GP: pooled outputs of the convolutional layers of the network as features in a new GP. This was also their best model from the collection. Nice overview of different metrics used for comparing methods for UQ. 


## Active learning:

* A. Seko and S. Ishiwata, “Prediction of perovskite-related structures in ACuO3-x (A = Ca, Sr, Ba, Sc, Y, La) using density functional theory and Ba,” Phys. Rev. B, vol. 101, no. 13, p. 134101, Apr. 2020.

* K. Tran and Z. W. Ulissi, Active learning across intermetallics to guide discovery of electrocatalysts for CO2 reduction and H2 evolution, vol. 1, no. 9. Springer US, 2018.

* D. Xue, P. V. Balachandran, J. Hogden, J. Theiler, D. Xue, and T. Lookman, “Accelerated search for materials with targeted properties by adaptive design,” Nat. Commun., vol. 7, pp. 1–9, 2016.

## Surrogate optimizer and accelerating TS searches: 

* O.-P. Koistinen, F. B. Dagbjartsdóttir, V. Ásgeirsson, A. Vehtari, and H. Jónsson, “Nudged elastic band calculations accelerated with Gaussian process regression,” J. Chem. Phys., vol. 147, no. 15, p. 152720, Oct. 2017.

* J. A. Garrido Torres, P. C. Jennings, M. H. Hansen, J. R. Boes, and T. Bligaard, “Low-Scaling Algorithm for Nudged Elastic Band Calculations Using a Surrogate Machine Learning Model,” Phys. Rev. Lett., vol. 122, no. 15, pp. 1–6, 2019.

* E. Garijo del Río, J. J. Mortensen, and K. W. Jacobsen, “Local Bayesian optimizer for atomic structures,” Phys. Rev. B, vol. 100, no. 10, pp. 1–9, 2019.

## Combining experiments + theory:

* E. O. Ebikade, Y. Wang, N. Samulewicz, B. Hasa, and D. Vlachos, “Active learning-driven quantitative synthesis–structure–property relations for improving performance and revealing active sites of nitrogen-doped carbon for the hydrogen evolution reaction,” React. Chem. Eng., 2020.

* A. Smith, A. Keane, J. A. Dumesic, G. W. Huber, and V. M. Zavala, “A machine learning framework for the analysis and prediction of catalytic activity from experimental data,” Appl. Catal. B Environ., vol. 263, no. October 2019, p. 118257, 2020.

* M. Zhong et al., Accelerated discovery of CO2 electrocatalysts using active machine learning, vol. 581, no. 7807. 2020.

* A. J. Saadun et al., “Performance of Metal-Catalyzed Hydrodebromination of Dibromomethane Analyzed by Descriptors Derived from Statistical Learning,” ACS Catal., vol. 10, no. 11, pp. 6129–6143, Jun. 2020.

* [J. L. Lansford and D. G. Vlachos, “Infrared spectroscopy data- and physics-driven machine learning for characterizing surface microstructure of complex materials,” Nat. Commun., vol. 11, no. 1, p. 1513, Dec. 2020](https://github.com/JLans/jl_spectra_2_structure/blob/master/jl_spectra_2_structure/primary_data_creation/dft_2_data.py)

* [Accelerated discovery of metallic glasses through iteration of machine learning and high-throughput experiments](https://advances-sciencemag-org.ezproxy.lib.purdue.edu/content/advances/4/4/eaaq1566.full.pdf)

* [Materials genes of heterogeneous catalysis from clean experiments and artificial intelligence](https://arxiv.org/ftp/arxiv/papers/2102/2102.08269.pdf)

* N. Artrith, Z. Lin, and J. G. Chen, “Predicting the Activity and Selectivity of Bimetallic Metal Catalysts for Ethanol Reforming using Machine Learning,” ACS Catal., vol. 10, no. 16, pp. 9438–9444, Aug. 2020.

* S. Nellaiappan et al., “High-Entropy Alloys as Catalysts for the CO2 and CO Reduction Reactions: Experimental Realization,” ACS Catal., vol. 10, no. 6, pp. 3658–3663, 2020.

## Reaction Network Predictions: 

* [M. Liu et al., “Reaction Mechanism Generator v3.0: Advances in Automatic Mechanism Generation,” J. Chem. Inf. Model., May 2021.](https://pubs.acs.org/doi/10.1021/acs.jcim.0c01480)

Newest version of RMG (v3) is updated to Python v3. It has ability to generate heterogeneous catalyst models, uncertainty analysis to conduct first order sensitivity analysis. RMG dataset for the thermochemical and kinetic parameters have been expanded. 

* [A Chemically Consistent Graph Architecture for Massive Reaction Networks Applied to Solid-Electrolyte Interphase Formation. ChemRxiv. Blau, Samuel; Patel, Hetal; Spotte-Smith, Evan; Xie, Xiaowei; Dwaraknath, Shyam; Persson, Kristin (2020)](https://chemrxiv.org/articles/preprint/A_Chemically_Consistent_Graph_Architecture_for_Massive_Reaction_Networks_Applied_to_Solid-Electrolyte_Interphase_Formation/13028105/1)

Develop a multi-reactant representation scheme to look at arbitrary reactant product pairs. Apply this technique to understand electrochemical reaction network for Li-ion solid electrolyte interphase. 

* [Predicting chemical reaction pathways in solid state material synthesis](https://www.nature.com/articles/s41467-021-23339-x)

Chemical reaction network model to predict synthesis pathway for exotic oxides. Solid-state synthesis procedures for YMnO<sub>3</sub>, Y<sub>2</sub>Mn<sub>2</sub>O<sub>7</sub>, Fe<sub>2</sub>SiS<sub>4</sub>, and YBa<sub>2</sub>Cu<sub>3</sub>O<sub>6.5</sub> are proposed and compared to literature pathways. Finally apply the algorithm to search for a probable synthesis route to make MgMo<sub>3</sub>(PO<sub>4</sub>)<sub>3</sub>O, battery cathode material that has yet to be synthesized.

* [Discovering Competing Electrocatalytic Mechanisms and Their Overpotentials: Automated Enumeration of Oxygen Evolution Pathways, A. Govind Rajan and E. A. Carter, J. Phys. Chem. C, vol. 124, no. 45, pp. 24883–24898, Nov. 2020.](https://pubs.acs.org/doi/10.1021/acs.jpcc.0c08120)

* [Accurate Thermochemistry of Complex Lignin Structures via Density Functional Theory, Group Additivity, and Machine Learning, Q. Li, G. Wittreich, Y. Wang, H. Bhattacharjee, U. Gupta, and D. G. Vlachos, ACS Sustain. Chem. Eng., vol. 9, no. 8, pp. 3043–3049, Mar. 2021.](https://pubs.acs.org/doi/abs/10.1021/acssuschemeng.0c08856)

* [Ziyun Wang, Yuguang Li, Jacob Boes et al. CO<sub>2</sub> Electrocatalyst Design Using Graph Theory, 21 September 2020, Preprint](https://www.researchsquare.com/article/rs-66715/v1)

## Generative Models:

**Review:**

J. Noh et al., “Inverse Design of Solid-State Materials via a Continuous Representation,” Matter, vol. 1, no. 5, pp. 1370–1384, 2019.

**Articles:** 

* S. Kim, J. Noh, G. H. Gu, A. Aspuru-Guzik, and Y. Jung, “Generative Adversarial Networks for Crystal Structure Prediction,” pp. 1–37, 2020

* B. Kim, S. Lee, and J. Kim, “Inverse design of porous materials using artificial neural networks,” Sci. Adv., vol. 6, no. 1, 2020

* [Z. Yao et al., “Inverse Design of Nanoporous Crystalline Reticular Materials with Deep Generative Models,” 2020](https://chemrxiv.org/articles/Inverse_Design_of_Nanoporous_Crystalline_Reticular_Materials_with_Deep_Generative_Models/12186681/1?file=22407606)

Semantically constrained graph-based code for presenting a MOFs. Target property directed optimization. Encode MOFs as edges, vertices, topologies. Edges are molecular fragments with two connecting points, verticies contain node information, topologies indicate a definite framework. Supramolecular Variational Autoencoder (SmVAE) with several corresponding components that oversee encoding and decoding each part of the MOF: Map the frameworks with discrete representations (RFcodes) into continuous vectors (z) and then back.

* [Discovering Relationships between OSDAs and Zeolites through Data Mining and Generative Neural Networks](https://pubs.acs.org/doi/10.1021/acscentsci.1c00024)

## Datasets:

While we can attribute the recent interest in material informatics to democratization of data analytics and ML packages, growing set of benchmark datasets of materials from multiple research institution has been crucial for development of new methods, algorithms and providing a consistent set of comparison. 

* OC20 dataset (CMU + Facebook). [Paper](https://arxiv.org/pdf/2010.09435.pdf). [Github](https://github.com/Open-Catalyst-Project/ocp)

Dataset comprising of surface heterogeneous adsorbates.  

* Catalysis Hub from SUNCAT. [Website](https://www.catalysis-hub.org)

Surface Reactions database contains thousands of reaction energies and barriers from density functional theory (DFT) calculations on surface systems

* [Materials Project](https://materialsproject.org)

Besides providing a collection of over 130,000 inorganic compounds and 49,000 molecules and counting, with calculated phase diagrams, structural, thermodynamic, electronic, magnetic, and topological properties it also provides analysis tools for post-processing. 

* [OQMD from Chris Wolverton's Group](http://oqmd.org)

815,000+ materials with calculated thermodynamic and structural properties.

* [ICSD](https://icsd.products.fiz-karlsruhe.de/en/products/icsd-products)

210,000+ inorganic crystal structures from literature. Requires subscription.

* [JARVIS by NIST](https://jarvis.nist.gov)

Includes calculated materials properties, 2D materials, and tools for ML and high-throughput tight-binding.

* [C2DB](https://cmr.fysik.dtu.dk/c2db/c2db.html)

Structural, thermodynamic, elastic, electronic, magnetic, and optical properties of around 4000 two-dimensional (2D) materials distributed over more than 40 different crystal structures. 

* [AFLOW](http://www.aflowlib.org)

Millions of materials and calculated properties, focusing on alloys.

* Citrination

Contributed and curated datasets from Citrine Informatics 

* [MDPS](https://mpds.io/tutorial/#MPDS-intro)

Fascinating resource linking scientific publications using the Pauling File database (relational database of published literature for material scientists)

* [Curated list of material informatics packages](https://github.com/tilde-lab/awesome-materials-informatics)

## Packages:

* [Perovskite oxide stability](https://github.com/uw-cmg/perovskite-oxide-stability-prediction)

* [DOSNet](https://github.com/vxfung/DOSnet)

* [Open catalysis dataset](https://github.com/Open-Catalyst-Project/ocp)

* [AENet](https://github.com/atomisticnet/MLP-beginners-guide)

* [AMP](https://bitbucket.org/andrewpeterson/amp/src/master/)

* [AMPtorch (PyTorch implementation of AMP)](https://github.com/ulissigroup/amptorch)

* [Feature engineering for Perovskite's electronic structure properties](https://github.com/zhengl0217/Perovskite-Electronic-Structure-Feature-Python)

* [MEGNET](https://github.com/materialsvirtuallab/megnet)

* [SISSO](https://github.com/rouyang2017/SISSO)

* [Catlearn](https://github.com/SUNCAT-Center/CatLearn)

* [Matminer](https://github.com/hackingmaterials/matminer)

* [Mat2Vec](https://github.com/materialsintelligence/mat2vec)
