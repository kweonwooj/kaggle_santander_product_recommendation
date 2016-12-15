## Kaggle/Santander Product Recommendation
<div align="center">
  <img src="https://kaggle2.blob.core.windows.net/competitions/kaggle/5558/media/santander-banner-ts-660x.png"><br><br>
</div>

## Introduction
**Santander Bank** offers a lending hand to their customers through personalized product recommendations. In their second competition, Santander is challenging Kagglers to predict which products their existing customers will use in the next month based on their past behavior and that of similar customers.

Competition data consists of customer data from 2015-01 ~ 2016-05 (total of 17 month timestamps) including customer's demographic information and their product purchase behavior. Competition challenges you to predict top 7 products out of 24, that each customer in the test data is most likely to purchase on 2016-06.

Evaluation metric is in MAP@7, which made the direct optimization difficult during training phase. Instead, the mlogloss was widely used among kagglers to indirectly optimize the solution.

With BreakfastPirates generos info, using 2015-06 data-only as a training data seemed to perform pretty well in the leaderboard (reaching almost ~0.03). Single model performance was enough to place you on top of the leaderboard, since MAP@7 made the effect of ensemble almost negligible.

As always, feature engineering seemed to be the most important factor in this competition, along with good cv scheme to reach the best hyper-parameter that squeezes the performance from the given data.

## Objective
My goals in this competition were as below:
- win a first Bronze Medal
- replicate past competition winning scripts
- do an Eeficient Decision Making via Visualization/Bias_Variance Analysis

Fortunately, I have won my first Bronze medal in Kaggle competition! (yay!)
Instead of replicating the past competition, I eagerly followed Forum posts to get the idea for feature engineering, cv scheme, optimization goal and so on.
From next time, I should equally spend my time replicating the past competition's winning scripts.
Once I was able to fix the cv scheme, I was able to stabilize my machine learning pipeline. Confirmations were made via forum posts, not from my own experiment or theory. From next time, I should be independent and responsible for my own cv scheme.

## Related works
- Current state-of-the-art recommendation systems follow two main streams
  - Factorization-based approache
  - Topic models
  - [Integrating Topic Models and Latent Factors for Recommendation](https://arxiv.org/abs/1610.09077)
  
- Actual Implementation
  - lag-5 features 
  - feature engineering
  - XGBoost 
  
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
- 
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
      - static categorical variable
    - pais_residencia : customer's country residence [string / 119]
      - static categorical variable
    - sexo : H, V, NaN [string / 3]
      - static categorical variable
    - age : [int / 121]
      - 3 unique values range(1~3) per user(25:35:40 = 1:2:3)
    - fecha_alta : date the customer became first holder of a contract [date / 6,757]
      - static categorical variable
      - 3 unique values range(0~2) per user(99% == 1)
    - ind_nuevo : new customer index (1 if customer is registered in last 6 months) [int / 3]
      - 3 unique values range(0~2) per user(1:90:9 = 0:1:2)
    - antiguedad : customer seniority in integer [int / 259]
      - 12 unique values range(1~12) per user(mostly on 10~12)
    - indrel : 1 (First/Primary), 99(Primary customer during the month but not at the end of the month) [int / 3]
      - 3 unique values range(0~2) per user(mostly on 1)
    - ult_fec_cli_lt : last date as primary customer [date / 224]
      - 6 unique values range(0~5) per user(mostly on 0 == NaN)
    - indrel_lmes : customer type at the beginning of the month, 1 (First/Primary customer), 2 (co-owner), P=5 (Potential), 3 (Former primary), 4 (former co-owner) [int / 6]
      - 7 unique values range(0~6) per user(mostly on 2)
    - tiprel_lmes : customer relation type at the beginning of the month. A (active), I (inactive), P (former customer), R (Potential), N [string / 6]
      - 5 unique values range(0~4) per user(mostly on 1)
    - indresi : Residence index (S(yes) or N(no) if customer's residence country is same than bank country) [string / 3]
      - 3 unique values range(0~2) per user(mostly on 1)
    - indext : Foreinger index (S(yes) or N(no) if customer's birth country is different than bank country [string / 3]
      - 3 unique values range(0~2) per user(mostly on 1)
    - conyuemp : spouse index. 1 if customer is spouse of an employee N,S,X [string / 3]
      - 2 unique values range(0~1) per user(mostly on 0)
    - canal_entrada : channel used by the customer to join [string / 163]
      - 4 unique values range(0~3) per user(mostly on 1)
    - indfall : deceased index N/S [string / 3]
      - 3 unique values range(0~2) per user(mostly on 1)
    - tipodom : address type. 1, primary address [int / 2]
      - 2 unique values range(0~1) per user(mostly on 1)
    - cod_prov : province code (customer's address) [int / 53]
      - 4 unique values range(0~3) per user(mostly on 1)
    - nomprov : province name [string / 53]
      - 4 unique values range(0~3) per user(mostly on 1)
    - ind_actividad_cliente : activity index (1, active customer; 0, inactive customer) [int / 3]
      - 3 unique values range(0~2) per user(mostly on 1)
    - renta : gross income of the household [int / 213,757]
      - 2 unique values range(0~1) per user(0:1 = 25:75)
    - segmento : segmentation: 01-VIP, 02-Individual, 03-College graduated[string / 4]
      - 4 unique values range(0~3) per user(mostly on 1)
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

## What to do
- Exploratory Data Analysis
  - Check whether description matches. 
  - Explore feature engineering ideas : try to capture change in status which triggers purchase
- Experiment with 1/100 data
  - Establish Pipeline
    - Feature Engineering
    - Linear, Tree Model
    - Evaluation
    - Generate Submission
- Validate Pipeline
  - validate feature engineering
    - check the range of each feature, distribution etc
  - validate evaluation method
    - check via putting handcrafted values
  - validate submission
    - check correlation with CV
- [Data] Scale upto 1/10, 1
  - Run pipeline
  - check improvement in evaluation score
  - check CV-LB correlation
- [Model] Parameter Tuning
  - Tune with 1/100 > apply to 1/10
  - Tune with 1/10 > apply to 1
- Single Model Process ends here
- Ensemble
  - bagging technique using different seed with 'best single model'
  - make different DB by sampling (0.6 ~ 0.8) of columns and apply differing models > Stacking
  
## What I did
- [ZFTurbo](https://www.kaggle.com/zfturbo/santander-product-recommendation/santander-battle/code)
  - Method
    - Collaborative Filtering: uses feature set to group user into demographics, and recommend most popular item within demographics purchased product that a user does not have. Extends to overall demographics if new user or 7 recommendation is not filled.
    - defining feature set to group users can be key
    - Very memory efficient and fast method!
  - Result
    - Best PLB : 0.0257818 | CV: 0.0261322 
- BreakfastPirate's advice
  - Method
    - Use 2015-06 data as training data.
    - Change problem into multi-class classification via melting the 24 target cols into single target col.
    - drop customers with no purchases
    - use lag-5 feature
    - feature engineer on product purchase behavior side
  - Possible improvements
    - clearer preprocessing methodology
    - XGBoost param tuning experience
    - feature engineering ideas
  - Result
    - Best PLB : 0.029975 | CV : ~0.94 (mlogloss)


## Related Works from past Kaggle Competitions
  - 1. [Coupon Purchase Prediction](https://www.kaggle.com/c/coupon-purchase-prediction)
    - General: 
      - Problem: Predict which coupons a customer will purchase
      - Evaluation metric: MAP@10
      - Submission format: user, coupons
    - Herra [1st place]
      - Probabilistic Modelling
        - P(purchase) = P(user online) x P(visit|online) x P(purchase|visit)
        - P(user online) is simple beta-bernoulli
        - other two were heirarchical logistic regression, inferenced by step wise doubly stochastic variational Bayes
          - first learn the population parameters/distributions
          - find out user specific params
        - Related papers
          - Mean VB + analytically optimized vairance hyperparamters, [section 3.2](http://jmlr.csail.mit.edu/proceedings/papers/v32/titsias14.pdf)
    - Halla [2nd place]
      - 1. For each week from 2012-01-08 through 2012-06-17, find all coupons with DISPFROM starting in that week.
      - 2. For each (user, coupon) pair in that week, construct features using data known as of that date.
      - 3. Code the target as 1 if a purchase occurred in that week, 0 if no purchase occurred.
      - 4. Throw out most of the 0's, since otherwise the dataset would be huge.
      - 5. Feed into XGBoost (a single, non-ensembled model).
      - 6. Validate using log-likelihood and a confusion matrix, and the last week (2012-06-17) only.
    - threecourse [3rd place]
      - 1. For each weeks, find all coupons with DISPFROM starting in that week.
      - 2. For each (user, coupon) pair in that week, construct features using data, where information in that week is excluded.
      - 3. Code the target as 1 if a purchase occurred in that week, 0 if no purchase occurred.
      - 4. Throw out all the 0's, since otherwise the dataset would be huge.
      - 5. Feed into XGBoost and Vowpal Wabbit (XGBoost was much better...)
      - 6. Validate by predicting and calculating MAP@10. Test set was every 5 weeks.
      - Total run time on an AWS r3.x2large instance(64GB RAM) : about 5-6 hours. My trick was giving up predicting genres which have little relation to places. In other words, excluded Gift, Delivery, Lesson and Other genres from training and prediction.
      - [Winning Script](https://github.com/threecourse/kaggle-coupon-purchase-prediction)
    - nagadomi [5th place]
      - 3 layer neural network
      - [Winning Script](https://github.com/nagadomi/kaggle-coupon-purchase-prediction)
    - Gert [27th place]
      - 1. Data Prep: complete pref of the user when missing, by filling in the most popular pref of the coupons that the user purchased
      - 2. Target Definition: count the number of times each coupon is purchased by all users together- separately for users in the SAME PREF as the coupon, and users in a DIFFERENT PREF
      - 3. Modeling: run two random forest regressors (SAME PREF and DIFFERENT PREF) with only the raw coupon attributes as features
      - 4. Submission: for each user, take at most the top 7 (if any) of coupons from the users own pref from the SAME PREF model, and fill it up to 10 coupons by taking coupons from other prefs according to the DIFFERENT PREF model
    - Aakansh [37th place]
      - XGBoost + BPRMF approach
      - [Blogpost](http://datascience.blog.uhuru.co.jp/machine-learning/predicting-coupon-purchases-on-ponpare/)
      - [Winning Script](https://github.com/aakansh9/kaggle-coupon-purchase-prediction)
    - tmain [68th place]
      - Cosine Similarity + [Hybrid Matrix Factorization](https://github.com/lyst/lightfm) + Regression
      - Blended via weighted arithmetic mean
      - [Winning Script](https://github.com/tdeboissiere/Kaggle/tree/master/Ponpare)
  - 2. [Expedia Hotel Recommendation](https://www.kaggle.com/c/expedia-hotel-recommendations)
    - General:
      - Problem: predict which hotels a customer will purchase
      - Evaluation metric: MAP@5
      - Submission format: id, hotel_clusters
    - idle_speculation [1st place]
      - 1. map user cities and clusters to latitude and longitude using gradient descent
      - 2. build a factorization machine model for each cluster
      - 3. calculate historical click and book rates by a variety of factors
      - 4. build a modified "rank:pairwise" xgboost model on 1-3
      - [Winning Solution Explained in Forum](https://www.kaggle.com/c/expedia-hotel-recommendations/forums/t/21607/1st-place-solution-summary)
    - beluga [2nd place]
      - 1. Map city to globe and calculate lat-long
      - 2. create seasonality proxy for destinations
      - 3. Hotel cluster frequencies based on factors
      - 4. user preferences
      - Split data based on leakage hotel matches 1:2 Trained binary xgb models separately for each hotel clusters. I used 8-20% of the negative samples in each binary classifiers to speed up training. separate feature selection and paropt helped.
    - Chenxia Ma [35th place]
      - [Winnnig Solution in Git](https://github.com/shawnhero/ICDM2013-Expedia-Recommendation-System)
  - 3. [Allstate Purchase Prediction](https://www.kaggle.com/c/allstate-purchase-prediction-challenge)
    - General:
      - Problem: predict which quote a customer will purchase
      - Evaluation metric: all or none
      - Submission format: customer, plan
  - 4. [Airbnb New User Bookings](https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings)
    - General:
      - Problem: predict which product a user will purchase first
      - Evaluation metric: NDCG
      - Submission format: id, country
  - 5. [GigaOM WordPress Challenge](https://www.kaggle.com/c/predict-wordpress-likes)
    - General:
      - Problem: predict which people will like which blogposts
      - Evaluation metric: MAP@100
      - Submission format: > 4 years
  - 6. [Job Recommendation Challenge](https://www.kaggle.com/c/job-recommendation)
    - General:
      - Problem: predict which jobs a user will apply
      - Evaluation metric: MAP@150
      - Submission format: > 4 years
  - 7. [Acquire Valued Shopper Challenge](https://www.kaggle.com/c/acquire-valued-shoppers-challenge)
    - General:
      - Problem: predict which new users will become loyal users
      - Evaluation metric: ROC
      - Submission format: id, prob
  - 8. [Homesite Quote Conversion](https://www.kaggle.com/c/homesite-quote-conversion)
    - General:
      - Problem: predict which customer will puchase a given quote
      - Evaluation metric: ROC
      - Submission format: quote, prob
  - 9. [Display Advertising Challenge](https://www.kaggle.com/c/criteo-display-ad-challenge)
    - General:
      - Problem: predict probability that a user will click an ad (CTR)
      - Evaluation metric: logloss
      - Submission format: id, prob
  - 10. [Click-Through Rate Prediction](https://www.kaggle.com/c/avazu-ctr-prediction)
    - General:
      - Problem: predict probability that a user will click an ad (CTR)
      - Evaluation metric: logloss
      - Submission format: id, prob
      
