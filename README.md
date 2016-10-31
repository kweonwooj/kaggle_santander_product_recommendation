## kaggle_santander
Kaggle / Santander Product Recommendation

## Objective
- Bronze Medal
- Modular Framework
- Efficient Decision Making via Visualization/Bias_Variance Analysis
- Replicate and Learn from the Best [ZFTurbo]

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
    - fecha_dato : date from 2015-01-28 ~ [date / 17]
    - ncodpers : customer code [int / 956,645]
    - ind_empleado : emplyee index: A active ,B ex employed, F filial, N not employeed, P passive [string / 6]
    - pais_residencia : customer's country residence [string / 119]
    - sexo : H, V, NaN [string / 3]
    - age : [int / 121]
    - fecha_alta : date the customer became first holder of a contract [date / 6,757]
    - ind_nuevo : new customer index (1 if customer is registered in last 6 months) [int / 3]
    - antiguedad : customer seniority in integer [int / 259]
    - indrel : 1 (First/Primary), 99(Primary customer during the month but not at the end of the month) [int / 3]
    - ult_fec_cli_lt : last date as primary customer [date / 224]
    - indrel_lmes : customer type at the beginning of the month, 1 (First/Primary customer), 2 (co-owner), P=5 (Potential), 3 (Former primary), 4 (former co-owner) [int / 6]
    - tiprel_lmes : customer relation type at the beginning of the month. A (active), I (inactive), P (former customer), R (Potential), N [string / 6]
    - indresi : Residence index (S(yes) or N(no) if customer's residence country is same than bank country) [string / 3]
    - indext : Foreinger index (S(yes) or N(no) if customer's birth country is different than bank country [string / 3]
    - conyuemp : spouse index. 1 if customer is spouse of an employee N,S,X [string / 3]
    - canal_entrada : channel used by the customer to join [string / 163]
    - indfall : deceased index N/S [string / 3]
    - tipodom : address type. 1, primary address [int / 2]
    - cod_prov : province code (customer's address) [int / 53]
    - nomprov : province name [string / 53]
    - ind_actividad_cliente : activity index (1, active customer; 0, inactive customer) [int / 3]
    - renta : gross income of the household [int / 213,757]
    - segmento : segmentation: 01-VIP, 02-Individual, 03-College graduated[string / 4]
    - ind_ahor_fin_ult1 : saving account [int / 2]
    - ind_aval_fin_ult1 : guarantees [int / 2]
    - ind_cco_fin_utl1 : current account [int / 2]
    - ind_cder_fin_utl1 : derivada account [int / 2]
    - ind_cno_fin_ult1: payroll account [int / 2]
    - ind_ctju_fin_ult1 : junior account [int / 2]
    - ind_ctma_fin_ult1 : mas particular account [int / 2]
    - ind_ctop_fin_ult1 : particular account [int / 2]
    - ind_ctpp_fin_ult1 : particular plus account [int / 2]
    - ind_deco_fin_ult1 : short-term deposits [int / 2]
    - ind_deme_fin_ult1 : medium-term deposits [int / 2]
    - ind_dela_fin_ult1 : long-term deposits [int / 2]
    - ind_ecue_fin_ult1 : e-account [int / 2]
    - ind_fond_fin_ult1 : funds [int / 2]
    - ind_hip_fin_ult1 : mortgage [int / 2]
    - ind_plan_fin_ult1 : pensions [int / 2]
    - ind_pres_fin_ult1 : loans [int / 2]
    - ind_reca_fin_ult1 : taxes [int / 2]
    - ind_tjcr_fin_ult1 : credit card [int / 2]
    - ind_valo_fin_ult1 : securities [int / 2]
    - ind_viv_fin_ult1 : home account [int / 2]
    - ind_nomina_ult1 : payroll [int / 2]
    - ind_nom_pens_ult1 : pensions [int / 2]
    - ind_recibo_ult1 : direct debit [int / 2]
  - test.csv
    - shape : (934,445 / 24)
    - fecha_dato : single day on 2016-06-28 [date / 1]
    - ncodpers : customer code [int / 934,445]
    - ind_empleado : emplyee index: A active ,B ex employed, F filial, N not employeed, P passive [string / 5]
    - pais_residencia : customer's country residence [string / 118]
    - sexo : H, V, NaN[string / 3]
    - age : [int / 118]
    - fecha_alta : date the customer became first holder of a contract [date / 6,780]
    - ind_nuevo : new customer index (1 if customer is registered in last 6 months) [int / 2]
    - antiguedad : customer seniority [int / 259]
    - indrel : 1 (First/Primary), 99 (Primary customer during the month but not at the end of the month) [int / 2]
    - ult_fec_cli_lt : last date as primary customer [date / 22]
    - indrel_lmes : customer type at the beginning of the month, 1 (First/Primary customer), 2 (co-owner), P=5 (Potential), 3 (Former primary), 4 (former co-owner) [int / 6]
    - tiprel_lmes : customer relation type at the beginning of the month. A (active), I (inactive), P (former customer), R (Potential), N [string / 6]
    - indresi : Residence index (S(yes) or N(no) if customer's residence country is same than bank country) [string / 2]
    - indext : Foreinger index (S(yes) or N(no) if customer's birth country is different than bank country [string / 2]
    - conyuemp : spouse index. 1 if customer is spouse of an employee N,S,X [string / 3]
    - canal_entrada : channel used by the customer to join [string / 163]
    - indfall : deceased index N/S [string / 2]
    - tipodom : address type. 1, primary address [int / 1]
    - cod_prov : province code (customer's address) [int / 53]
    - nomprov : province name [string / 53]
    - ind_actividad_cliente : activity index (1, active customer; 0, inactive customer) [int / 2]
    - renta : gross income of the household [int / 212,964]
    - segmento : segmentation: 01-VIP, 02-Individual, 03-College graduated[string / 4]
- [02] Local CV Strategy
  - Intersection Analysis 
    - ncodpers [trn : tst]
      - trn : 0.03
      - tst : 0.00
      - int : 0.97
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
- KJ method
  - [DONE] Data Exploration on train.csv
  - [DONE] Split CV
    - [a] begin with pure random shufflesplit
    - According to train/test relationship > no outstanding relationship
    - [b] use 2016-05-28 as vld
  - [DO] Initial Benchmark
    - RandomForest, XGBoost, Ridge, Logistic etc (base models)
      - try on split_a and split_b
    - Memory Issue!!
  - [DO] Simple Feature Aggregated submission
  - [DO] Feature Engineered v1
- [ZFTurbo](https://www.kaggle.com/zfturbo/santander-product-recommendation/santander-battle/code)
  - Method
    - Collaborative Filtering: uses feature set to group user into demographics, and recommend most popular item within demographics purchased product that a user does not have. Extends to overall demographics if new user or 7 recommendation is not filled.
    - defining feature set to group users can be key
    - Very memory efficient and fast method!
  - [01] As-is
    - no preprocessing on raw data
    - Grouping Condition (4670 groups)
      - pais_residencia, sexo, age, ind_nuevo, segmento, ind_empleado, ind_actividad_cliente, indresi
    - Local CV: 0.022101
    - Public LB: 0.0241798
  - [01-a] Other variants
    - Group
      - pais
  - [02] v2
    - Preprocessing
      - Age : bin
      - others
  
## What I did
  - Tried KJ style.. Hold
  - Begin Replicating ZFTurbo
