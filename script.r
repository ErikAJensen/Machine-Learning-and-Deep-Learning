# Pakker
if (!require(C50)) install.packages("C50", dependencies=TRUE)
if (!require(caret)) install.packages("caret", dependencies=TRUE)
library(C50); library(caret)

# Data
df <- read.csv("StressLevelDataset.csv")
df$stress_level <- as.factor(df$stress_level)
cat("Rader:", nrow(df), "  Kolonner:", ncol(df), "\n")

# Split
set.seed(42)
idx <- createDataPartition(df$stress_level, p=0.8, list=FALSE)
train <- df[idx, ]; test <- df[-idx, ]
cat("Train:", nrow(train), "  Test:", nrow(test), "\n")

# Modell
ctrl <- trainControl(method="cv", number=5)
form <- reformulate(setdiff(names(train), "stress_level"), response="stress_level")
m_c50 <- train(form, data=train, method="C5.0", trControl=ctrl, tuneLength=3, metric="Accuracy", preProcess=c("center", "scale", "YeoJohnson"))
print(m_c50)

# Evaluering
pred <- predict(m_c50, newdata=test)
cm <- confusionMatrix(pred, test$stress_level)
print(cm)

# Lagre
write.csv(cbind(test, .pred=pred), "prediksjoner_c50.csv", row.names=FALSE)
saveRDS(m_c50, "modell_c50.rds")
writeLines(capture.output(summary(m_c50$finalModel)), "c50_rules.txt")

# Feature importance
imp <- C50::C5imp(m_c50$finalModel, metric="usage")
imp <- imp[order(imp$Overall, decreasing=TRUE), , drop=FALSE]
print(imp)
barplot(imp$Overall, names.arg=rownames(imp), las=2, cex.names=0.7, main="C5.0 – variabel-importance", ylab="Usage")