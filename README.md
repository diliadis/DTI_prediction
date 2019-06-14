# Flight radar route network visualization

This is a drug-target interaction prediction (DTI) project written in python. It is based on my [master's thesis](https://drive.google.com/file/d/1ADn2lGBbg35vnDhezeKz62QZV67FF5JP/view?usp=sharing)   


## Basic info

This implementation uses an imbalance aware version of a supervised multi-label method, called classifier chain. More specifically this is a unique ensemble version of the classifier chain that is based on the following [paper](https://arxiv.org/abs/1807.11393).

The paper introduced the basis ensemble of classifier chains with random undersmapling (ECCRU) method as well as two other variants that try to optimize the exploitation of the computational bUdget (ECCRU2 and ECCRU3). All three methods are implemented in the project


The user has the capability of choosing to run the training and testing in a multi-processing mode. This parallelization is implemented on the ensemble level. That means that each classifier chain runs on a single process. 
The nature of the classifier chain algorithm makes the parallelization of the chain itself impossible because each member in the chain needs the results of the previous member as features for training.


## Datasets

The methods mentioned above are tested on the following datasets:

### 1) ChEMBL dataset

This dataset was compiled as a target prediction benchmark dataset out of the ChEMBL database to be used by [Unterthiner et al.](http://www.bioinf.jku.at/publications/2014/NIPS2014a.pdf).
The whole dataset as well as detailed notes on preprocessing are freely available at [this](http://www.bioinf.jku.at/research/VirtScreen/) page.
This dataset contains more than 1,200 targets, 1.3 million compounds and 13 million ECFP12 features.

### 2) Gold standard dataset

The is considered as the gold standard in the area of drug-target interaction prediction. 
It is publicly [available](http://web.kuicr.kyoto-u.ac.jp/supp/yoshi/drugtarget/) and contains four different types of drug-target interactions networks. 
Each network contains a different type of protein targets, namely, enzymes, ion-channels, G-protein-coupled receptors, and nuclear receptors.
All this drug target interaction information was extracted from the following databases: 

* [DrugBank](https://www.drugbank.ca/)
* [KEGG BRITE](https://www.genome.jp/kegg/brite.html)
* [BRENDA](https://www.brenda-enzymes.org/) 
* [SuperTarget](http://insilico.charite.de/supertarget/index.php). 


## Preprocessing

### 1) ChEMBL dataset
After [Unterthiner et al.](http://www.bioinf.jku.at/publications/2014/NIPS2014a.pdf) applied thresholds to the activity data that they extracted from the ChEMBL database, the final dataset contained 2,103,018 activity measurements across 5,069 protein targets and 743,339 chemical compounds. 
Because it is very important that compounds that share a scaffold, do not co-exist in different folds, they also clustered the compounds using single linkage clustering to guarantee a minimal distance between train and test set. 
This method produced 400,000 clusters that were partitioned into three folds of roughly equal size.

The number of interactions of each protein target varied significantly. Some targets have over 50,000 interactions while others have very few measurements.
Furthermore, there is great label imbalance for many of the targets in the dataset. 
This bias stems from the fact that researchers are more likely to experiment with compounds that have the potential to return be highly active with protein targets.

To ensure that each target has sufficient samples for our models to train on, we used only the targets that have more than 15 active instances and more than 15 inactive instances. 
The enforcement of this limit reduces the number of available targets to 1,230.

The result of all this pre-processing was the production of mainly four files, which we used as the input of our implementation:

* chembl.fpf: this file contains 1,318,187 records. Each line begins with the id of a chemical compound (e.g., CHEMBL153534) and continues with the features. Each feature consists of the
   id, the character ':' and the value 1. The features are represented as ECFP12 fingerprints. This is in essence a sparse representation where the ones represent substructures that the compound has.

<p align="center">
  <img src="https://github.com/diliadis/DTI_prediction/blob/master/images/chemblScreenshot2.png">
</p>

* SampleIdTable.txt: This file contains 1,318,187 records. 
Each line contains only the id of a chemical compound which is in the same format with the id in the chembl.fpf file.

<p align="center">
  <img src="https://github.com/diliadis/DTI_prediction/blob/master/images/SampeldTableScreenshot2.png">
</p>

* cluster.info: this file contains 1,318,187 records. Each line has two columns. 
The first column has the id of the assigned cluster and the second column has the id of the chemical compound (there are three folds-ids and the possible values are 0,1,2). 
In this file the id of the chemical compound does not have the same formatting as the corresponding ids in the chembl.fpf and SampleIdTable.txt files. 
On this file, the value of the new id corresponds to the location of the old id in the file SampleIdTable.txt.

<p align="center">
  <img src="https://github.com/diliadis/DTI_prediction/blob/master/images/cluster1Screenshot2.png">
</p>


* targetActivities.txt: This file contains 3.172.523 records and each line has three columns. 
The first column contains the id of the type of activity that is observed for a compound-target pair. 
The possible values, as well as their meaning, are the following:

    * 1: inactive
    * 11: weak inactive
    * 3: active
    * 13: weak active
    * 2: unknown
    * 0: contradicting
    * 1: 10: weakly contradicting


<p align="center">
  <img src="https://github.com/diliadis/DTI_prediction/blob/master/images/targetActivitiesScreenshot2.png">
</p>



We also had to implement additional preprocessing steps to be able to use the interaction information in the above files.

In the first step of the implementation, the id of every compound in the chembl.fpf file is replaced based on its location in the SampleIdTable.txt file. 
For example if the compound with id  CHEMBL153534 is located in the tenth line of the SampleIdTable.txt, then the new id of the compound will be 10. 
The above step is implemented so that we can utilize the information located in the cluster.info and TargetActivities.txt files. 
The next step involves the use of the cluster.info file and the goal is to use its cluster assignments to split the chembl.fpf file in train and test sets. 
This process is repeated three times so that in every iteration two folds produce the train set and the third fold produces the test set.

We then use the training set to map the train and test set features to a new space where the ids range starts from one and ends in the total number of features. 
At this stage, we also discard features with frequencies less than a desired threshold (our threshold was 100 compounds per feature).
Features that were found only in the test set and not in the train set were also rejected, as the trained models cannot recognize them and therefore use them in their predictions.

In the next step we use the targetActivities.txt file's information so that for each biological target, only the active and inactive chemical compounds are kept. 
At this point, it is important to note an additional step that filters the targets to address the unbalanced nature of the data. 
Targets that contain fewer than 15 active and 15 inactive samples are discarded. 
With this step, we try to ensure that every trained model will not show bias in predictions towards the majority class.




### 2) Gold standard dataset
The number of proteins in these 4 different datasets are 664, 204, 95 and 26. 
These proteins interact with, respectively, 445, 210, 223 and 54 drugs through 2926, 1476, 635 and 90 positive interactions.
A brief description of the datasets is given in the table below.

This final dataset is produced after a crawling script is run (golden_datasets_preprocessing.py). 
This script has as input the interaction datasets that we obtain from [http://web.kuicr.kyoto-u.ac.jp](http://web.kuicr.kyoto-u.ac.jp/supp/yoshi/drugtarget/).
For every compound id we first obtain the ChEMBL code from [KEGG](https://www.kegg.jp/), and then we capture the SMILES representation from [https://www.ebi.ac.uk](https://www.ebi.ac.uk/chembldb/). 
The result of these two steps is that we have the SMILES representation of every compound in the dataset. 
This representation is then given to the [RDKit library](https://www.rdkit.org/) so that we can obtain the extended connectivity fingerprints or ECFPs that will be used as features in the experiments. 


| Dataset | Compounds | Proteins | Interactions | Mean(ImR) | min(ImR) | max(ImR) |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| `Enzyme` | 418 | 664 | 2926 | 251.67 | 417 | 6.08 |
| `Ion Channel` | 191 | 204 | 1476 | 63.08 | 190 | 4.61 |
| `GPCR` | 203 | 95 | 635 | 94,27 | 202 | 5.58 |
| `Nuclear Receptor` | 54 | 26 | 90 | 26.97 | 53 | 2.37 |
| `ChEMBL` | 743,339 | 5,069 | 2,103,018 | 8191.51 | 20.29 | 43943.06 |



## Results

The datasets that are used in the literature of drug-target interaction prediction are usually highly imbalanced.
The number of inactive pairs (negative samples) significantly exceeds that of the active pairs (positive samples).
In this type of datasets traditional metrics like accuracy convey a false picture of performance.
Two measures that are far more suitable for imbalanced datasets are the area under the Receiver Operating Characteristic (ROC) curve, auROC, and the area under the precision-recall (PR) curve, auPR.
These two measures have become standard metrics for comparison in the area of drug-target interaction prediction.


### 1) Gold standard dataset

The gold standard datasets, although convenient for comparison between different methods in this area, they present serious limitations like the fact that they contain only true positive interactions.
This characteristic makes the datasets ignore many important aspects of the drug-target interactions like its quantitative affinity.

Additionally, the majority of the methods developed in the area use prediction formulations that are based on the 
practically unrealistic assumption that during the construction of the models and the evaluation of their predictive accuracy, we have the full information about the drug and target space.

In particular, the typical evaluation method assumes that the drug-target pairs to be predicted in the validation set are randomly distributed in the known drug-target interaction matrix

<p align="center">
  <img src="https://github.com/diliadis/DTI_prediction/blob/master/images/unrealistic.pdf">
</p>