library(data.table)

mode <- "submission"
product.list <- colnames(fread("train_ver2.csv", select=25:48, nrows=0))
date.list <- "2016-05-28"

submission <- fread("sample_submission.csv")[, .(ncodpers)]

result <- data.table()
for(i in 1:length(date.list)) {
  print(date.list[i])
  result.temp <- data.table()
  for(j in 1:length(product.list)) {
    product <- product.list[j]
    
    if(j %in% c(1,2,10,11)) {
      temp <- submission
      temp$pr  <- 1e-10
    } else {
      if(product == "ind_cco_fin_ult1") {
        train.date <- "2015-12-28"
      } else if(product == "ind_reca_fin_ult1") {
        train.date <- "2015-06-28"
      } else {
        train.date <- "2016-05-28"
      }
      temp <- fread(paste0("submission_", product, "_", train.date, ".csv"))
      colnames(temp)[2] <- "pr"
      temp$product <- product
    }
    temp$product <- product
    result.temp <- rbind(result.temp, temp)
  }
  
  # normalize
  pred.sum <- result.temp[product %in% product.list[-c(3,18)], .(sum=sum(pr)), by=ncodpers]
  result.temp <- merge(result.temp, pred.sum, by="ncodpers")
  result.temp <- result.temp[, .(ncodpers, log_pr=log(pr/sum), product)]
  result <- rbind(result, result.temp[, .(ncodpers, product, log_pr, N=1)])
  result <- result[, .(log_pr=sum(log_pr), N=sum(N)), by=.(ncodpers, product)]
}

# log-average
result$log_pr <- result$log_pr / result$N

# elect top 7 products
result <- result[order(ncodpers, -log_pr)]
for(i in 1:7) {
  print(i)
  temp <- result[!duplicated(result, by="ncodpers"), .(ncodpers, product)]
  submission <- merge(submission, temp, by="ncodpers", all.x=TRUE)
  result <- result[duplicated(result, by="ncodpers"), .(ncodpers, product)]
  colnames(submission)[ncol(submission)] <- paste0("p",i)
  if(nrow(result) == 0) {
    break
  }
}

submission[is.na(submission)] <- ""
submission$added_products <- submission[[paste0("p",1)]]
for(i in 2:7) {
  submission$added_products <- paste(submission$added_products, submission[[paste0("p",i)]])
}

submission <- submission[order(ncodpers)]
file.name <- paste0(mode, ".csv")
write.csv(submission[, .(ncodpers, added_products)], file.name, quote=FALSE, row.names=FALSE)
