library(data.table)
library(dplyr)

# drop fecha_alta, ult_fec_cli_1t, tipodom, cod_prov, ind_ahor_fin_ult1, ind_aval_fin_ult1, ind_deco_fin_ult1 and ind_deme_fin_ult1
data <- fread("train_ver2.csv", drop=c(7,11,19,20,25,26,34,35), colClasses=c(indrel_1mes="character", conyuemp="character"))

date.list <- c(unique(data$fecha_dato), "2016-06-28")
product.list <- colnames(data)[(ncol(data)-19):ncol(data)]

##### data 1: inner join with last month #####

for(i in c(6,12:length(date.list))) {
  print(date.list[i])
  if(date.list[i] != "2016-06-28") {
    out <- data[fecha_dato==date.list[i]]
    out <- merge(out, data[fecha_dato==date.list[i-1], c("ncodpers","tiprel_1mes","ind_actividad_cliente",product.list), with=FALSE], by="ncodpers", suffixes=c("","_last"))
    write.csv(out, paste0("train_", date.list[i], ".csv"), row.names=FALSE)
  } else {
    out <- fread("test_ver2.csv", drop=c(7,11,19,20), colClasses=c(indrel_1mes="character", conyuemp="character"))
    out <- merge(out, data[fecha_dato==date.list[i-1], c("ncodpers","tiprel_1mes","ind_actividad_cliente",product.list), with=FALSE], by="ncodpers", suffixes=c("","_last"))
    colnames(out)[(ncol(out)-19):ncol(out)] <- paste0(colnames(out)[(ncol(out)-19):ncol(out)], "_last")
    write.csv(out, paste0("test_", date.list[i], ".csv"), row.names=FALSE)
  }
}

##### data 2: count the change of index #####

for(i in c(6,12:length(date.list))) {
  print(date.list[i])
  if(date.list[i] != "2016-06-28") {
    out <- merge(data[fecha_dato==date.list[i], .(ncodpers)], data[fecha_dato==date.list[i-1], .(ncodpers)], by="ncodpers")
  } else {
    out <- fread("../root_input/test_ver2.csv", select=2)
  }
  for(product in product.list) {
    print(product)
    temp <- data[fecha_dato %in% date.list[1:(i-1)], c("fecha_dato","ncodpers",product), with=FALSE]
    temp <- temp[order(ncodpers, fecha_dato)]
    temp$n00 <- temp$ncodpers==lag(temp$ncodpers) & lag(temp[[product]])==0 & temp[[product]]==0
    temp$n01 <- temp$ncodpers==lag(temp$ncodpers) & lag(temp[[product]])==0 & temp[[product]]==1
    temp$n10 <- temp$ncodpers==lag(temp$ncodpers) & lag(temp[[product]])==1 & temp[[product]]==0
    temp$n11 <- temp$ncodpers==lag(temp$ncodpers) & lag(temp[[product]])==1 & temp[[product]]==1
    temp[is.na(temp)] <- 0
    count <- temp[, .(sum(n00, na.rm=TRUE), sum(n01, na.rm=TRUE), sum(n10, na.rm=TRUE), sum(n11, na.rm=TRUE)), by=ncodpers]
    colnames(count)[2:5] <- paste0(product, c("_00","_01","_10","_11"))
    count[[paste0(product,"_0len")]] <- 0
    
    for(date in date.list[1:(i-1)]) {
      temp2 <- temp[fecha_dato==date]
      temp2 <- temp2[match(count$ncodpers, ncodpers)]
      flag <- temp2[[product]] == 0
      flag[is.na(flag)] <- 0
      count[[paste0(product,"_0len")]] <- (count[[paste0(product,"_0len")]] + 1) * flag
    }
    out <- merge(out, count, by="ncodpers")
  }
  write.csv(out[, -1, with=FALSE], paste0("count_", date.list[i], ".csv"), quote=FALSE, row.names=FALSE)
}
