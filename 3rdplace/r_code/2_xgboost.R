library(data.table)
library(xgboost)

# drop ind_ahor_fin_ult1, ind_aval_fin_ult1, ind_deco_fin_ult1 and ind_deme_fin_ult1
product.list <- colnames(fread("train_ver2.csv", select=25:48, nrows=0))[-c(1,2,10,11)]

for(product in product.list) {
  
  print(product)
  
  if(product == "ind_cco_fin_ult1") {
    train.date <- "2015-12-28"
  } else if(product == "ind_reca_fin_ult1") {
    train.date <- "2015-06-28"
  } else {
    train.date <- "2016-05-28"
  }
  
  data.1 <- fread(paste0("train_", train.date, ".csv"), colClasses=c(indrel_1mes="character", conyuemp="character"))
  data.2 <- fread(paste0("count_", train.date, ".csv"))
  data.train <- cbind(data.1, data.2)
  
  if(train.date == "2016-05-28") {
    data.1 <- fread("train_2016-04-28.csv", colClasses=c(indrel_1mes="character", conyuemp="character"))
    data.2 <- fread("count_2016-04-28.csv")
  } else {
    data.1 <- fread("train_2016-05-28.csv", colClasses=c(indrel_1mes="character", conyuemp="character"))
    data.2 <- fread("count_2016-05-28.csv")
  }
  data.val <- cbind(data.1, data.2)
  
  data.1 <- fread("test_2016-06-28.csv", colClasses=c(indrel_1mes="character", conyuemp="character"))
  data.2 <- fread("count_2016-06-28.csv")
  data.test <- cbind(data.1, data.2)
  
  rm(data.1)
  rm(data.2)
  gc()
  
  data.train <- data.train[data.train[[paste0(product,"_last")]] == 0]
  data.val <- data.val[data.val[[paste0(product,"_last")]] == 0]
  data.test <- data.test[data.test[[paste0(product,"_last")]] == 0]
  
  data.train$ind_actividad_cliente_from_to <- paste(data.train$ind_actividad_cliente_last, data.train$ind_actividad_cliente)
  data.val$ind_actividad_cliente_from_to <- paste(data.val$ind_actividad_cliente_last, data.val$ind_actividad_cliente)
  data.test$ind_actividad_cliente_from_to <- paste(data.test$ind_actividad_cliente_last, data.test$ind_actividad_cliente)
  
  data.train$tiprel_1mes_from_to <- paste(data.train$tiprel_1mes_last, data.train$tiprel_1mes)
  data.val$tiprel_1mes_from_to <- paste(data.val$tiprel_1mes_last, data.val$tiprel_1mes)
  data.test$tiprel_1mes_from_to <- paste(data.test$tiprel_1mes_last, data.test$tiprel_1mes)

  data.train[, ind_actividad_cliente_last:=NULL]
  data.val[, ind_actividad_cliente_last:=NULL]
  data.test[, ind_actividad_cliente_last:=NULL]
  
  data.train[, tiprel_1mes_last:=NULL]
  data.val[, tiprel_1mes_last:=NULL]
  data.test[, tiprel_1mes_last:=NULL]
  
  data.train$n_products_last <- apply(data.train[, (1:20)+20+20, with=FALSE], 1, sum)
  data.val$n_products_last <- apply(data.val[, (1:20)+20+20, with=FALSE], 1, sum)
  data.test$n_products_last <- apply(data.test[, (1:20)+20, with=FALSE], 1, sum)
  
  data.train$products_last <- ""
  data.val$products_last <- ""
  data.test$products_last <- ""
  for(j in 1:20) {
    data.train$products_last <- paste0(data.train$products_last, data.train[[j+20+20]])
    data.val$products_last <- paste0(data.val$products_last, data.val[[j+20+20]])
    data.test$products_last <- paste0(data.test$products_last, data.test[[j+20]])
  }
  
  exp.var <- colnames(data.test)[-(1:2)]
  for(var in exp.var) {
    if(class(data.train[[var]])=="character") {
      levels <- levels(as.factor(data.train[[var]]))
      if(length(levels) == 2) {
        temp <- c(data.train[[var]], data.val[[var]], data.test[[var]])
        temp <- ifelse(temp==levels[1], 0, ifelse(temp==levels[2], 1, NA))
        data.train[[var]] <- temp[1:nrow(data.train)]
        data.val[[var]] <- temp[(nrow(data.train)+1):(nrow(data.train)+nrow(data.val))]
        data.test[[var]] <- temp[(nrow(data.train)+nrow(data.val)+1):length(temp)]
      } else {
        # replace with target mean
        data.temp <- data.train[, c(var, product), with=FALSE]
        colnames(data.temp)[ncol(data.temp)] <- "target"
        target.mean <- data.temp[, .(target=mean(target)), by=var]
        data.val[[var]] <- target.mean$target[match(data.val[[var]], target.mean[[var]])]
        data.test[[var]] <- target.mean$target[match(data.test[[var]], target.mean[[var]])]
        temp <- rep(NA, nrow(data.train))
        for(j in 1:4) {
          ids.1 <- -seq(j, nrow(data.train), by=4)
          ids.2 <- seq(j, nrow(data.train), by=4)
          target.mean <- data.temp[ids.1, .(target=mean(target)), by=var]
          temp[ids.2] <- target.mean$target[match(data.train[[var]][ids.2], target.mean[[var]])]
        }
        data.train[[var]] <- temp
      }
    }
  }
  rm(data.temp)
  
  x.train <- data.train[, exp.var, with=FALSE]
  y.train <- data.train[[product]]
  dtrain <- xgb.DMatrix(data=as.matrix(x.train), label=y.train)
  rm(data.train)
  rm(x.train)
  
  x.val <- data.val[, exp.var, with=FALSE]
  y.val <- data.val[[product]]
  dval <- xgb.DMatrix(data=as.matrix(x.val), label=y.val)
  data.val <- data.val[, .(ncodpers)]
  rm(x.val)
  
  x.test <- data.test[, exp.var, with=FALSE]
  dtest <- xgb.DMatrix(data=as.matrix(x.test), label=rep(NA, nrow(data.test)))
  data.test <- data.test[, .(ncodpers)]
  rm(x.test)
  
  gc()
  
  nrounds <- 1000
  early_stopping_round <- 50
  params <- list("eta"=0.05,
                 "max_depth"=4,
                 "min_child_weight"=1,
                 "objective"="binary:logistic",
                 "eval_metric"="auc")
  
  set.seed(0)
  model.xgb <- xgb.train(params=params,
                         data=dtrain,
                         nrounds=nrounds,
                         watchlist=list(train=dtrain, val=dval),
                         early_stopping_round=early_stopping_round,
                         print_every_n=10,
                         base_score=mean(y.train))
  
  result <- data.table(data.val$ncodpers, predict(model.xgb, dval))
  colnames(result) <- c("ncodpers", product)
  result <- result[order(ncodpers)]
  write.csv(result, paste0("validation_", product, "_", train.date, ".csv"), quote=FALSE, row.names=FALSE)
  
  result <- data.table(data.test$ncodpers, predict(model.xgb, dtest))
  colnames(result) <- c("ncodpers",product)
  result <- result[order(ncodpers)]
  write.csv(result, paste0("submission_", product, "_", train.date, ".csv"), quote=FALSE, row.names=FALSE)
  
  rm(dtrain)
  rm(dval)
  rm(dtest)
  rm(result)
  gc()
  
  save(model.xgb, file=paste0("xgboost_", product, "_", train.date, ".model"))
  
}
