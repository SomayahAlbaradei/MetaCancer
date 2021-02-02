# deepmet
DeepMet: A Deep Learning-Based Pan-cancer Metastasis Prediction Model Developed using Multi-omics Data

Predicting metastasis in the early stages means that clinicians have more time to adjust a treatment regimen to target the primary and metastasized cancer. In this regard, several computational approaches are being developed to identify metastasis early. However, most of the approaches focus on changes on one genomic level only, and they are not being developed from a pan-cancer perspective. Thus, we here present a deep learning (DL)–based model, DeepMet, that differentiates pan-cancer metastasis status based on three heterogeneous data layers. In particular, we built the DL-based model using 400 patients' data that includes RNA sequencing (RNA-Seq), microRNA sequencing (microRNA-Seq), and DNA methylation data from The Cancer Genome Atlas (TCGA). We quantitatively assess the proposed convolutional variational autoencoder (CVAE) and alternative feature extraction methods. We further show that integrating mRNA, microRNA, and DNA methylation data as features improves our model's performance compared to when we used mRNA data only. Also, we show the mRNA-related features make a more significant contribution when attempting to distinguish the primary tumors from metastatic ones computationally. Lastly, we show that our DL model significantly outperformed a machine learning (ML) ensemble method based on various metrics.

### Requirements
  - Models run on linux machine.
  - Anaconda Python 3.6 or later.
  - keras.
    
### Usage

To successfully run deepmet models we recomend you to create a virtul environment based on requirements.txt:
```
 $ conda create --name deepmet --file requirements.txt
```
Activate your virtual environment:
```
 $ source activate deepmet 
```

To run an Acceptor model:

```
  $ python deepmet.py  Input= 'pass file' Output='output file name'
```
    
```
Predections will be stored in output file.
```



For comments please contact somayah.albaradei@kaust.edu.sa
  
 

