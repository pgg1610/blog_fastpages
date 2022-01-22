---
toc: true
layout: post
description: Rdkit code snippets and recipes that I revisit now and again.
categories: [chemical-science, exploratory-data-analysis, machine-learning, resources]
title: Rdkit quick tips 
---

Rdkit code snippets and recipes that I revisit now and again. The snippets are adopted from different python scripts written over time, ignore the variable names.

### Fingerprints 

**Quick ECFP fingerprint** 

```python
from rdkit.Chem import AllChem
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

# Convert to Chem.Mol: 
mol = Chem.MolFromSmiles(smiles)

# Counts by default - unfolded 
rdMolDescriptors.GetMorganFingerprint(mol, radius) 

# Folded counts 
rdMolDescriptors.GetHashedMorganFingerprint(mol, radius, nBits=num_bits)

#Folded FP bit vectors as per the size of the bits 
morgan_fp_bit_vect = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=num_bits)

# Convert to numpy 
fp = np.zeros((0,), dtype=np.int16)
DataStructs.ConvertToNumpyArray(morgan_fp_bit_vect, fp)
```

### Loading data

**Tanimoto similarity matrix**

Adapted from Andrew White: 

```python
import itertools
def tanimoto_matrix(slist):
    '''
    Compute pair-wise Tanimoto similarity between a list of smiles with ECFP4 FPs
    '''
    fp = [ AllChem.GetMorganFingerprint( Chem.MolFromSmiles(s), 2 ) for s in slist ]
    ts = list(
    DataStructs.cDataStructs.TanimotoSimilarity(x,y) for x, y, in itertools.product(fp, repeat=2)
    )
    return np.array(ts).reshape(len(fp), len(fp))
```

**Loading ZINC dataset** 

Adapted from Andrew White:

```python
tranches = pd.read_csv('https://gist.githubusercontent.com/whitead/f47887e45bbd2f38332182d2d422da6b/raw/a3948beac9b9034dab432b697c5ec238503ac5d0/tranches.txt')
def get_mol_batch(batch_size = 32):
  for t in tranches.values:
    d = pd.read_csv(t[0], sep=' ')    
    for i in range(len(d) // batch_size):
      yield d.iloc[i * batch_size:(i + 1) * batch_size, 0].values
```

### Viewing molecules 

**Viewing molecules in a grid**

```python
import pandas as pd
from rdkit.Chem import PandasTools

>> PandasTools.AddMoleculeColumnToFrame(df, smilesCol='smiles')
>> esol_data.head(1)

>> PandasTools.FrameToGridImage(df.head(8), legendsCol="logSolubility", molsPerRow=4)
```

**Adding new values as a column** 

```python
df["n_Atoms"] = df['ROMol'].map(lambda x: x.GetNumAtoms())
df.head(1)
```

**Molecules in a xlsx file**

```python
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools

>> smiles = ['c1ccccc1', 'c1ccccc1O', 'c1cc(O)ccc1O']
>> df = pd.DataFrame({'ID':['Benzene', 'Phenol', 'Hydroquinone'], 'SMILES':smiles})
>> df['Mol Image'] = [Chem.MolFromSmiles(s) for s in df['SMILES']]
>> PandasTools.SaveXlsxFromFrame(df, 'test.xlsx', molCol='Mol Image')
```

**Viewing substructures** 

```python
def viz_substruct(main_smile, substructure_smarts):
    
    mol_file = Chem.MolFromSmiles(main_smile)
    sub_pattern = Chem.MolFromSmarts(substructure_smarts)
    
    hit_ats = list(mol_file.GetSubstructMatch(sub_pattern)) # Returns the indices of the molecule’s atoms that match a substructure query
    hit_bonds = []

    for bond in sub_pattern.GetBonds():
        aid1 = hit_ats[bond.GetBeginAtomIdx()]
        aid2 = hit_ats[bond.GetEndAtomIdx()]

        hit_bonds.append( mol_file.GetBondBetweenAtoms(aid1, aid2).GetIdx() )

    d2d = rdMolDraw2D.MolDraw2DSVG(400, 400) # or MolDraw2DCairo to get PNGs
    rdMolDraw2D.PrepareAndDrawMolecule(d2d, mol_file, highlightAtoms=hit_ats,  highlightBonds=hit_bonds)
    d2d.FinishDrawing()
    return SVG(d2d.GetDrawingText())

>> diclofenac = 'O=C(O)Cc1ccccc1Nc1c(Cl)cccc1Cl'
>> substruct_smarts = 'O=CCccN'
>> viz_substruct(diclofenac, substruct_smarts)
```

**View a reaction**

```python
rxn = AllChem.ReactionFromSmarts('[C:1]=[C:2].[C:3]=[*:4][*:5]=[C:6]>>[C:1]1[C:2][C:3][*:4]=[*:5][C:6]1')
```

Get changed atoms in a reaction: https://greglandrum.github.io/rdkit-blog/tutorial/reactions/2021/11/26/highlighting-changed-bonds-in-reactions.html

**Edit, merge, react molecules** 

Molecule tinkering using Rdkit: http://asteeves.github.io/blog/2015/01/14/editing-in-rdkit/

**Using mol2grid**

mols2grid is an interactive chemical viewer for 2D structures of small molecules, based on RDKit.

* [Jupyter notebook explaining simple application](https://practicalcheminformatics.blogspot.com/2021/10/exploratory-data-analysis-with.html)