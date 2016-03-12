---
layout: post
title: some about R
category: homework
tags: Essay
keywords: r
---



```r
bootstrap.lm<-function(z,nsample){
    x<-z$x;y<-z$y
    nrows<-dim(x)[1]
    value<-list()
    for(i in seq(nsample)){
       rows<-sample(nrow,nrow,T)
       value[[i]]<-lm.fit(x[rows,],y[rows,])
    }
    value
}
```

```r
factor<- function(x, levels= sort(unique(x)),labels = as.character(levels)) {     y <- match(x, levels)   #very important
     names(y) < - names(x) 
     levels(y) <- labels 
    class(y) <- "factor"
    y    }
```