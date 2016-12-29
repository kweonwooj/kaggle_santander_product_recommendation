# Santander Product Recommendation - 8th place

### Caution
* make_data() step in main.py needs 30GB of memory but it can be optimized.

### This code produces 3 submissions
* xgboost - 0.03061 public LB
* lightgbm - 0.03059 public LB
* xgboost+lightgbm - 0.03063 public LB

### Steps
* place train_ver2.csv, test_ver2.csv to ../input/
* install pandas, scikit-learn, numpy, xgboost, lightgbm (or comment out lightgbm part) libs for python3
* set proper number of threads in engines.py
* ./run.sh

--
### What I have learned

```clean.py```

smooth workflow containing..

- concatenating train and test set at first
- preprocessing, feature engineering on single big table
- using subset of data necessary for train_predict

```main.py```

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

```engines.py```

- use of ```X_train = XY_train.as_matrix(columns=[features])```, ```Y_train = XY_train.as_matrix(columns=['y'])``` to subset input dataframe into necessary dataformat for xgboost model construction. 

```utils.py```
- use of ```[apk() for a,p in zip(actual, predicted)]``` zip method in loop

