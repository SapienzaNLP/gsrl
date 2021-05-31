<div align="center">    
 
# <u>G</u>enerating <u>S</u>enses and <u>R</u>o<u>L</u>es: <br> An End-to-End Model for Dependency- and Span-based Semantic Role Labeling

[![Paper](https://img.shields.io/badge/paper-IJCAI--Proceedings-blue)](https://github.com/SapienzaNLP/gsrl/blob/master/docs/IJCAI_2021_GSRL_CameraReady.pdf)
[![Conference](https://img.shields.io/badge/Conference-IJCAI--2021-red)](https://ijcai-21.org/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

</div>
GSRL (<u>G</u>enerating <u>S</u>enses and <u>R</u>o<u>L</u>es), is a novel approach to sequence-to-sequence end-to-end Semantic Role Labeling, i.e., it performs predicate disambiguation, argument identification and classification as a single generation problem in an autoregressive fashion.

If you find our paper, code or framework useful, please reference this work in your paper:

```
@inproceedings{blloshmi-etal-2021-generating,
    title = {Generating Senses and RoLes: An End-to-End Model for Dependency- and Span-based Semantic Role Labeling},
    author = {Blloshmi, Rexhina and Conia, Simone and Tripodi, Rocco and Navigli, Roberto},
    booktitle = {Proceedings of IJCAI},
    year = {2021}
}
```


## Pretrained Checkpoints

####CoNLL-2009

**Paper experiments:** 

|   Model	| Checkpoint  	|   F1 (test)	| 
|---	|---	|---	
|`GSRL_nested`	|   [best-dep-srl_nested_checkpoint.pt](https://drive.google.com/file/d/1Xml-0PqDm-SRFlN3UmPzZQTtRR-Zk4Yn/view?usp=sharing)	|   89.0 	|
|`GSRL_flattened`	|   [best-dep-srl_flattened_checkpoint.pt](https://drive.google.com/file/d/1zAZQuJgSwPaWwGU_17bcK2G4-OPyF39p/view?usp=sharing)	|    92.4	|

**Extra experiments:** 

|   Model	| Checkpoint  	|   F1	(test)| 
---	|---	|---
|`GSRL_nested`    ( **-** predicate identifiers)| [best-dep-srl_nested_nopred-identifiers_checkpoint.pt](https://drive.google.com/file/d/11ioSGdWuldOrnpqxJTMpR9nWNMikOUKg/view?usp=sharing)   |   	83.2|
|`GSRL_flattened` ( **-** BART pretraning)	|   [best-dep-srl_flattened_nopretraining_checkpoint.pt](https://drive.google.com/file/d/1MxSxEcV0UO50BQpnHtCBn5K9STxuoOin/view?usp=sharing)	|   85.5	|   	

####CoNLL-2012

**Paper experiments:** 

|   Model	| Checkpoint  	|   F1 (test)	| 
|---	|  ---	| ---	
|`GSRL_nested`	|   [best-span-srl_nested_checkpoint.pt](https://drive.google.com/file/d/13gZesBdkpqxvktpO2EEqxILXVPdK-mMb/view?usp=sharing)	|   86.8 	|
|`GSRL_flattened`	|   [best-span-srl_flattened_checkpoint.pt](https://drive.google.com/file/d/1fXlUzQQMyni9jnk-sKykZhl6PoXWiYIE/view?usp=sharing)	|  87.3 	|


**Extra experiments:** 

|   Model	| Checkpoint  	|   F1 (test)	| 
---	|---	|---
|`GSRL_nested` ( **-** predicate identifiers)| [best-span-srl_nested_nopred-identifiers_checkpoint.pt](https://drive.google.com/file/d/1lf10JJ7m8A61WB_O_lDfcwIeLn87kJXc/view?usp=sharing) |  71.8 |
|`GSRL_flattened` ( **-** BART pretraning)	  |   [best-span-srl_flattened_nopretraining_checkpoint.pt](https://drive.google.com/file/d/12jXufD_40hR36uLduDi2mKZiU7iF-wvC/view?usp=sharing)	|   76.6	|   	


## Evaluation Framework
* Coming soon

## 1. Install 

Create a conda environment with **Python 3.8** and **PyTorch 1.5.0** and install the dependencies [requirements.txt](requirements.txt).

Via conda:

    conda create -n gsrl python=3.8
    conda activate gsrl
    bash ./download_artifacts.sh

To enable wandb logging**: 

    wandb login
**Also set _log_wandb_ to **True** (currenly **False**) in `configs` files and fill in _wandb-project_  and _team_ information accordingly.

## 2. Add the CoNLL-2009 and CoNLL-2012 datasets inside `data/` directory. 

Modify the data paths in the configuration files in `configs/` or follow our file structure. 

E.g., the folder structure for CoNLL-2009 should look as below:

(gsrl)$ tree data/conll-2009 -L 2 data/conll-2009

    conll-2009
        └── en
            │ ── dev
            │   └──CoNLL2009_dev.txt
            │── ood
            │   └── CoNLL2009_test_ood.txt
            ├── test
            │   └── CoNLL2009_test.txt
            └── training
                └── CoNLL2009_train.txt

## Training & Evaluation
- All configuration and parameters to reproduce our main results are included in `configs/` directory.

- Logs of wandb and model checkpoints are saved in `runs/`.

- Evaluation scripts are in `scripts/` and their output is saves in `out/`.  

- Vocabulary additions are included in `data/vocab/`. To allow reproducability do not change the files.
### Span-based Semantic Role Labeling

1. To train a GSRL model with _nested_ linearization:
```shell script
python -m src.bin.train 
        --config configs/config-span-srl.yaml 
        --task-type span
```
Evaluate the model using the following command: 
```shell script
python -m src.bin.predict_srl 
       --datasets data/conll-2012/en/test/CoNLL2012_test.txt 
       --checkpoint runs/[checkpoint_name_here] 
       --task-type span 
       --beam-size 1 
       --eval-name nested-span-srl-result
```
2. To train a GSRL model with _flattened_ linearization:
```shell script
python -m src.bin.train 
       --config configs/config-span-srl.yaml 
       --task-type span 
       --duplicate-per-predicate
```
Evaluate the model using the following command: 
```shell script
python -m src.bin.predict_srl 
       --datasets data/conll-2012/en/test/CoNLL2012_test.txt 
       --checkpoint runs/[checkpoint_name_here] 
       --task-type span 
       --beam-size 1 
       --duplicate-per-predicate 
       --eval-name flattened-span-srl-result
```

### Dependency-based Semantic Role Labeling
2. To train a GSRL model with _nested_ linearization:
```shell script
python -m src.bin.train 
       --config configs/config-dep-srl.yaml 
       --task-type dep
```
Evaluate the model using the following command: 
```shell script
python -m src.bin.predict_srl 
       --datasets data/conll-2009/en/test/CoNLL2009_test.txt 
       --checkpoint runs/[checkpoint_name_here] 
       --task-type dep 
       --beam-size 1 
       --eval-name nested-dep-srl-result
```
2. To train a GSRL model with _flattened_ linearization:
```shell script
python -m src.bin.train 
      --config configs/config-dep-srl.yaml 
      --task-type dep 
      --duplicate-per-predicate
```
Evaluate the model using the following command: 
```shell script
python -m src.bin.predict_srl 
       --datasets data/conll-2009/en/test/CoNLL2009_test.txt 
       --checkpoint runs/[checkpoint_name_here] 
       --task-type dep 
       --beam-size 1 
       --duplicate-per-predicate 
       --eval-name flattened-dep-srl-result
```
## Extra

* To run without predicate identifiers in input, add `--identify-predicate` in both training and evaluation scripts above.

## License
This project is released under the CC-BY-NC 4.0 license (see `LICENSE`). If you use `GSRL`, please put a link to this repo.

## Acknowledgements
The authors gratefully acknowledge the support of the [ERC Consolidator Grant MOUSSE](http://mousse-project.org) No. 726487 and the [ELEXIS project](https://elex.is/) No. 731015 under the European Union’s Horizon 2020 research and innovation programme.

This work was supported in part by the MIUR under the grant "Dipartimenti di eccellenza 2018-2022" of the Department of Computer Science of the Sapienza University of Rome.

We adopted modules or code snippets from the open-source projects:
* [SPRING](https://github.com/SapienzaNLP/spring) 