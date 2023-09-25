# Development and validation of a machine learning model of radiation induced hypothyroidism with clinical and dose–volume features (Radiotherapy & Oncology 2023)
**[Paper](https://doi.org/10.1016/j.radonc.2023.109911)**

This is the official repository for the Radiotherapy and Oncology 2023 paper "_Development and validation of a machine learning model of radiation induced hypothyroidism with clinical and dose–volume features_".  

Authors: 
Mu-Hung Tsai, Joseph T.C. Chang, Hsi-Huei Lu, Yuan-Hua Wu, Tzu-Hui Pao, Yung-Jen Cheng, Wen-Yen Zhen, Chen-Yu Chou, Jing-Han Lin, Tsung Yu, Jung-Hsien Chiang

## Predict-RIHT
In this repository, you will find the dataset to recreate results from the paper.
Please note that you will need to install [scikit-survival](https://github.com/sebp/scikit-survival) with:
`conda install -c conda-forge scikit-survival==0.21.0`
Other requirements include `numpy`, `pandas`, and `pickle`.

### Dataset
We are providing public access to the dataset, including the [developmental cohort](https://github.com/accordtsai/predict-RIHT/data_training.csv) and [external validation cohort](https://github.com/accordtsai/predict-RIHT/data_validation.csv). 


### Training / evaluation: Recreating results from the paper
To recreate results from the paper, ensure `data_training.csv` and `data_validation.csv` are in the same directory, and run `training.py` as below:
```shell
python training.py
```

Results from Table 3 are recreated. Please note this will overwrite the models files in the `models/` directory from this repository, however the results should be the same.

### Example of using ML models in a treatment planning system
Please see `predict_example.py` for an example of using ML models in a treatment planning system (TPS). 

This script illustrates predicting radiation-induced hypothyroidism risk in a single patient, using a thyroid-only model and thyroid DVH data acquired from the treatment planning system via [Varian ESAPI](https://github.com/VarianAPIs/PyESAPI).

This script requires `matplotlib` and `pyesapi` libraries. Please ensure that the machine is able to connect to the TPS database (i.e. working ESAPI). A trained model should be in the `models/` directory.

First, edit the script to define the patient ID to extract information from; this patient must have an organ named 'Thyroid'. 
```
patient_id = '12345678'
```

Then, run the script to automatically extract the DVH information from the TPS, and predict freedom from >= grade 1 RIHT:
```shell
python predict_example.py
```
which will show a predicted survival curve for this patient.

## Acknowledgements
This study was supported by a research grant from [Varian Medical Systems](https://www.varian.com/).

## Citation
If you found our project helpful, please cite our paper:

```
@article{
    riht2023, 
    title={Development and validation of a machine learning model of radiation-induced hypothyroidism with clinical and dose–volume features},
    volume={189},
    DOI={10.1016/j.radonc.2023.109911},
    journal={Radiotherapy and Oncology},
    author={Tsai, Mu-Hung and Chang, Joseph T.C. and Lu, Hsi-Huei and Wu, Yuan-Hua and Pao, Tzu-Hui and Cheng, Yung-Jen and Zheng, Wen-Yen and Chou, Chen-Yu and Lin, Jing-Han and Yu, Tsung and et al.}, 
    year={2023},
    pages={109911}
} 
```