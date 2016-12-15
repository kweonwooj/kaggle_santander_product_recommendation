from feats import clean, base, numeric, date, category, advanced

def main():
    # convert trimmed train.csv to train_clean.csv
    print 'trim train.csv to train_clean.csv'
    #clean.main()

    # convert train_clean.csv into LabelEncoded train_base.csv
    print 'convert train_clean.csv into LabelEncoded train_base.csv'
    #base.main()

    # numerical features
    print 'numerical feature'
    #numeric.main()

    # date features
    print 'date feature'
    #date.main()

    # categorical features
    print 'categorical feature'
    #category.main()

    # advanced features
    print 'advanced feature'
    #advanced.main()


if __name__=='__main__':
    main()