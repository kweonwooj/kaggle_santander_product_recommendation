## kaggle_santander
Kaggle / Santander Product Recommendation

## Objective
- Bronze Medal
- Modular Framework
- Efficient Decision Making via Visualization/Bias_Variance Analysis

## Evaluation
- [Mean Average Precision @ 7](https://www.kaggle.com/c/santander-product-recommendation/details/evaluation)
  - Top Score : 
  - Training Error : 
  - Validation Error : 
- Decide whether it is Overfitting or Underfitting > Proceed with appropriate actions
  - Underfitting : bigger model, train longer
  - Overfitting : More data, regularization
  - Always try : New model architecture, feature engineering

## Workflow
- [00] Run 10% of data as quick iteration > 100% for submission only
- [01] Data Exploration
  - [Example]
  - filename
    - shape : (row / col)
    - colname : description [type / unique]
  - train.csv
    - shape : (13,647,309 / 48)
    - fecha_dato : [object / ]
    - ncodepers : [float / ]
    - ind_empledado : [object / ]
    - pais_residencia : [object / ]
    - sexo : [object / ]
    - age : [float / ]
    - fecha_alta : [object / ]
    - ind_nuevo : [float / ]
    - antiguedad : [float / ]
    - indrel : [float / ]
    - ult_fec_cli_lt : [object / ]
    - indrel_lmes : [object / ]
    - tiprel_lmes : [object / ]
    - indresi : [object / ]
    - conyuemp : [object / ]
    - canal_entrada : [object / ]
    - indfall : [object / ]
    - tipodom : [float / ]
    - cod_prov : [float / ]
    - nomprov : [object / ]
    - ind_actividad_cliente : [float / ]
    - renta : [float / ]
    - segmento : [object / ]
    - ind_ahor_fin_ult1 : [int / ]
    - ind_aval_fin_ult1 : [int / ]
    - ind_cco_fin_utl1 : [int / ]
    - ind_cder_fin_utl1 : [int / ]
    - ind_cno_fin_ult1: [int / ]
    - ind_
- [02] Local CV Strategy
- [03] Feature Engineering
  - [01] Basic Feature
  - [02] Advanced Feature Generation
  - [03] Feature Reduction
  - [04] Feature Selection
- [04] Model Selection
- [05] Ensemble

## Benchmark
- [01] Initial Benchmark
- [02] 

## Lessons learnt

## How to run

## What to do
- [DO] Split CV
- [DO] Initial Benchmark
- [DO] Simple Feature Aggregated submission
- [DO] Feature Engineered v1

## What I did
