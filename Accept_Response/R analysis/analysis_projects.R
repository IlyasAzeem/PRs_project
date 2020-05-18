
accept_projects <- read.csv2("E:/Documents/Research Work/PRs_project/Accept_Response/R analysis/accept_projects_all.csv", sep=',')
response_projects <- read.csv2("E:/Documents/Research Work/PRs_project/Accept_Response/R analysis/response_projects_all.csv", sep=',')

# PR acceptance/response prediction projects
accept_proj <- as.factor(accept_projects$Model)
print(accept_proj)

kruskal_accept_proj <- kruskal.test(accept_projects$AUC~accept_proj)
kruskal_accept_proj

pairwise.wilcox.test(as.numeric(accept_projects$AUC), accept_projects$Model, p.adjust.method="holm")


response_proj <- as.factor(response_projects$Model)
print(response_proj)

kruskal_response_proj <- kruskal.test(response_projects$AUC~response_proj)
kruskal_response_proj

pairwise.wilcox.test(as.numeric(response_projects$AUC), response_projects$Model, p.adjust.method="holm")



## comparison of XGBoost and baseline


accept_xgb_baseline <- accept_projects[(accept_projects$Model == 'XGBoost') | (accept_projects$Model == 'baseline'),]

accept_xgb_base_models <- as.factor(accept_xgb_baseline$Model)
print(accept_xgb_base_models)

kruskal_accept_comp <- kruskal.test(accept_xgb_baseline$AUC~accept_xgb_base_models)
kruskal_accept_comp

pairwise.wilcox.test(as.numeric(accept_xgb_baseline$AUC), accept_xgb_baseline$Model, p.adjust.method="holm")


response_xgb_baseline <- response_projects[(response_projects$Model == 'XGBoost') | (response_projects$Model == 'baseline'),]

response_xgb_base_models <- as.factor(response_xgb_baseline$Model)
print(response_xgb_base_models)

kruskal_response_comp <- kruskal.test(response_xgb_baseline$AUC~response_xgb_base_models)
kruskal_response_comp

pairwise.wilcox.test(as.numeric(response_xgb_baseline$AUC), response_xgb_baseline$Model, p.adjust.method="holm")

