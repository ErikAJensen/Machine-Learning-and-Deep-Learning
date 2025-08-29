library(C50)
library(caret)

# --- Les data ---
df <- read.csv("kunder.csv")     # legg fila i samme mappe som script.R
target <- "Churn"                # målkolonnen i eksempelet
df[[target]] <- as.factor(df[[target]])

# --- Train/test-splitt (litt større test for å sikre rader) ---
set.seed(42)
idx <- createDataPartition(df[[target]], p = 2/3, list = FALSE)  # ca. 4 train / 2 test
train <- df[idx, ]
test  <- df[-idx, ]
stopifnot(nrow(train) > 0, nrow(test) > 0)  # sikkerhet

# --- CV-oppsett enklere (tåler små datasett) ---
ctrl <- trainControl(method = "cv", number = 3)  # 3-fold, ingen repeats

# --- Tren C5.0 ---
form <- reformulate(setdiff(names(train), target), response = target)
m_c50 <- train(
  form, data = train,
  method = "C5.0",
  trControl = ctrl,
  tuneLength = 3,
  metric = "Accuracy",
  preProcess = c("center", "scale", "YeoJohnson")
)
print(m_c50)

# --- Evaluer ---
pred <- predict(m_c50, newdata = test)
print(confusionMatrix(pred, test[[target]]))

# --- Lagre resultater ---
write.csv(cbind(test, .pred = pred), "prediksjoner_c50.csv", row.names = FALSE)
saveRDS(m_c50, "modell_c50.rds")
