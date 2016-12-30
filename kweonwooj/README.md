# What I have learned

## 2nd place

[Instructions for Tom's solution](https://github.com/ttvand/Santander-Product-Recommendation/blob/master/Instructions%20Santander%20Product%20Recommendation.pdf)

- feature selection
  - using 5 folds combined with 100 xgboost rounds for the base models to generate the feature ranking
- feature ordering
  - generate first level base model relative weights using model weights
- base model generation
  - using 10 folds combined with 200 xgboost rounds trained on the top 100 features for each lag-product pair.
- post-processing
  - taking into an account of MAP characteristic
- project management
  - organized and deep project flow

## 3rd place

```
1_make_data_v3.R
```

- saving the data separately with modularity (train-{date}.csv, count-{date}.csv)
- use of change of index (count-{date}.csv)

```
2_xgboost.R
```

- use of concatenation of two columns to generate new column
- *setting average target value to each categorical feature when unique count > 2

- R is more efficient in data processing than python

## 8th place

```
clean.py
```

smooth workflow containing..

- concatenating train and test set at first
- preprocessing, feature engineering on single big table
- using subset of data necessary for train_predict

```
main.py
```

- use of ```custom_one_hot``` function extracting the subset of whole one-hot-encoding efficiently.
- use of ```.map(lambda x: func(x))``` in handling pandas dataframe, series efficiently.
    - use of ```.map(math.log)``` for direct log transformation
    - use of if/else statement within map ```.map(lambda x: 0 if math.isnan(x) else x+1.0)``` 
    - use of ```x.__class__ is float``` to check for NaN
- use of ```df.merge(prev_df, on=['ncodpers','int_date'], how=how)``` to merge via two columns
- use of ```for begin, end in [(1,3), (1,5), (2,5)]:``` iteration over list of tuples to efficiently loop over
- use of ```mp_df = train_df.as_matrix(columns=prods)``` to subset dataframe to form numpy array directly
- smart method to obtain purchase user data and get single multi-class target label at once in line 286 of ```main.py```.
- use of ```wegiths = np.exp(1/counts - 1)``` exponential weight to feed into xgboost/lightgbm
- subtracting prev month posession from prediction ```Y_test_xgb - Y_prev``` 

```
engines.py
```

- use of ```X_train = XY_train.as_matrix(columns=[features])```, ```Y_train = XY_train.as_matrix(columns=['y'])``` to subset input dataframe into necessary dataformat for xgboost model construction. 

```
utils.py
```
- use of ```[apk() for a,p in zip(actual, predicted)]``` zip method in loop

