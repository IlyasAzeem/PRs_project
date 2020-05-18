
accept_10_folds <- read.csv2("E:/Documents/Research Work/PRs_project/Accept_Response/R analysis/accept_10_fold_all.csv", sep=',')
response_10_folds <- read.csv2("E:/Documents/Research Work/PRs_project/Accept_Response/R analysis/response_10_fold_all.csv", sep=',')

library(effsize)

View(accept_10_folds)


# PR acceptance/response prediction 10 folds
accept_models <- as.factor(accept_10_folds$Model)
print(accept_models)

kruskal_accept_folds <- kruskal.test(accept_10_folds$AUC~accept_models)
kruskal_accept_folds

pairwise.wilcox.test(as.numeric(accept_10_folds$AUC), accept_10_folds$Model, p.adjust.method="holm")


cliff.delta(as.numeric(accept_10_folds$AUC), accept_10_folds$Compare, return.dm = TRUE)
#cliffDelta(x=as.numeric(accept_10_folds$AUC), y=accept_10_folds$Model)


#treatment <- c(10,10,20,20,20,30,30,30,40,50)
#control <- c(10,20,30,40,40,50)
#res = cliff.delta(treatment,control,return.dm=TRUE)
#print(res)
#print(res$dm)


response_models <- as.factor(response_10_folds$Model)
print(response_models)

kruskal_response_folds <- kruskal.test(response_10_folds$AUC~response_models)
kruskal_response_folds

pairwise.wilcox.test(as.numeric(response_10_folds$AUC), response_10_folds$Model, p.adjust.method="holm")

cliff.delta(as.numeric(response_10_folds$AUC), response_10_folds$Compare)


## comparison of XGBoost and baseline


accept_xgb_baseline <- accept_10_folds[(accept_10_folds$Model == 'XGBoost') | (accept_10_folds$Model == 'baseline'),]

View(accept_xgb_baseline)

accept_xgb_base_models <- as.factor(accept_xgb_baseline$Model)
print(accept_xgb_base_models)
accept_xgb_base_auc <- as.factor(accept_xgb_baseline$AUC)
print(accept_xgb_base_auc)

kruskal_accept_comp <- kruskal.test(accept_xgb_baseline$AUC~accept_xgb_base_models)
kruskal_accept_comp

pairwise.wilcox.test(as.numeric(accept_xgb_baseline$AUC), accept_xgb_baseline$Model, p.adjust.method="holm")

cliff.delta(as.numeric(accept_xgb_base_auc), accept_xgb_base_models)


response_xgb_baseline <- response_10_folds[(response_10_folds$Model == 'XGBoost') | (response_10_folds$Model == 'baseline'),]

response_xgb_base_models <- as.factor(response_xgb_baseline$Model)
print(response_xgb_base_models)

kruskal_response_comp <- kruskal.test(response_xgb_baseline$AUC~response_xgb_base_models)
kruskal_response_comp

pairwise.wilcox.test(as.numeric(response_xgb_baseline$AUC), response_xgb_baseline$Model, p.adjust.method="holm")
