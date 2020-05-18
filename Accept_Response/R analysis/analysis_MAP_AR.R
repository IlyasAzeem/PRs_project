
map <- read.csv2("E:/Documents/Research Work/PRs_project/Accept_Response/R analysis/All_MAP_results.csv", sep=',')
ar <- read.csv2("E:/Documents/Research Work/PRs_project/Accept_Response/R analysis/All_AR_results.csv", sep=',')

library(effsize)

View(map)
View(ar)


# Mean Average Precision
map_models <- as.factor(map$Model)
print(map_models)

kruskal_map <- kruskal.test(map$Top_10~map_models)
kruskal_map

pairwise.wilcox.test(as.numeric(map$Top_10), map$Model, p.adjust.method="holm")


cliff.delta(as.numeric(map$Top_10), map$Compare, return.dm = TRUE)

# Average Recall
ar_models <- as.factor(ar$Model)
print(ar_models)

kruskal_ar <- kruskal.test(ar$Top_10~ar_models)
kruskal_ar

pairwise.wilcox.test(as.numeric(ar$Top_10), ar$Model, p.adjust.method="holm")


cliff.delta(as.numeric(ar$Top_10), ar$Compare, return.dm = TRUE)
