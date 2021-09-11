# Is sensor space analysis good enough? Spatial patterns as a tool for assessing spatial mixing of EEG/MEG rhythms

This repository provides analysis code to analyze spatial mixing in electrophysiological data through lead field and spatial pattern coefficients. 

## Reference

Schaworonkow N & Nikulin VV: Is sensor space analysis good enough? Spatial patterns as a tool for assessing spatial mixing of EEG/MEG rhythms. bioRxiv (2021).

## Dataset

The results are based on following available openly available data set: ["Leipzig Cohort for Mind-Body-Emotion Interactions" (LEMON dataset)](http://fcon_1000.projects.nitrc.org/indi/retro/MPI_LEMON.html), from which we used the preprocessed EEG data. The associated data set research article: 
- Babayan A et al.: [A mind-brain-body dataset of MRI, EEG, cognition, emotion, and peripheral physiology in young and old adults.](http://www.nature.com/articles/sdata2018308) _Scientific Data_ (2018).

We also use the [New York Head](https://www.parralab.org/nyhead/), a head model and pre-computed lead field. The asssociated research articles are: 
-  Huang Y, Parra LC, Haufe S: [The New York Head -- A precise standardized volume conductor model for EEG source localization and tES targeting](https://www.sciencedirect.com/science/article/pii/S1053811915011325), _NeuroImage_ (2015).
- Haufe S , Huang Y, Parra LC: [A highly detailed FEM volume conductor model of the ICBM152 average head template for EEG source imaging and tCS targeting](https://www.parralab.org/nyhead/HauHuaPar-embc-2015.pdf). In: _Conf Proc IEEE Eng Med Biol Soc_ (2015).

To reproduce the results, the preprocessed EEG data and leadfield matrix (file name: ```sa_nyhead.mat```) should be downloaded and placed into the folder ```data``` (or otherwise, the path to the data needs to be adjusted).

## Requirements

The provided python3 scripts are using ```scipy``` and ```numpy``` for general computation, ```pandas``` for saving intermediate results to csv-files. ```matplotlib``` for visualization. For EEG-related analysis, the ```mne``` package is used. For computation of aperiodic exponents: [```specparam```](https://specparam-tools.github.io/). Specifically used versions can be seen in ```requirements.txt```. 

# Pipeline
To reproduce the figures from the command line, navigate into the ```code``` folder and execute ```make all```. This will run through the preprocessing steps and generate the figures. The scripts can also be executed separately in the order described in the ```Makefile```.