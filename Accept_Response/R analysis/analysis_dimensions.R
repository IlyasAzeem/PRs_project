
accept_dimensions <- read.csv2("E:/Documents/Research Work/PRs_project/Accept_Response/R analysis/accept_all_dimensions.csv", sep=',')
response_dimensions <- read.csv2("E:/Documents/Research Work/PRs_project/Accept_Response/R analysis/response_all_dimensions.csv", sep=',')

accept_Dim <- as.factor(accept_dimensions$Model)
print(accept_Dim)

kruskal_accept_dim <- kruskal.test(accept_dimensions$AUC~accept_Dim)
kruskal_accept_dim

pairwise.wilcox.test(as.numeric(accept_dimensions$AUC), accept_dimensions$Model, p.adjust.method="holm")


response_Dim <- as.factor(response_dimensions$Model)
print(response_Dim)

kruskal_response_dim <- kruskal.test(response_dimensions$AUC~response_Dim)
kruskal_response_dim

pairwise.wilcox.test(as.numeric(response_dimensions$AUC), response_dimensions$Model, p.adjust.method="holm")
