
### STAT 6301 Final Project ###

#Load packages
library(dplyr)
library(ggplot2)
library(gmodels)
library(agricolae)
library(multcomp)
library(Sleuth2)
library(MASS)
library(car)
library(glmnet)
library(caret)
library(leaps)
library(bestglm)
library(VIM)
library("VIM")
library(forcats)
library(stringr)
library(WriteXLS)

# Question 1 #

#House data (train set only)
house = read.csv("train.csv")
house = subset(house, Neighborhood=="NAmes" | Neighborhood=="Edwards" | Neighborhood=="BrkSide")
house$GrLivArea2 = house$GrLivArea/100
head(house)
str(house)

#Fit the model (ref is automatically set as 'BrkSide')
house.lm <- lm(SalePrice ~ GrLivArea2 + Neighborhood + GrLivArea2:Neighborhood, data = house)
summary(house.lm)

#Before proceeding, is the interaction term between "GrLivArea" and "Neighborhood" statistically significant?

#Re-fit model without interaction term
lm.without = lm(SalePrice ~ GrLivArea2 + Neighborhood, data = house)
summary(lm.without)

#Extra sum of squares test
anova(lm.without, house.lm)

# The extra sum of squares test is statistically significant (p-value < .01). That is, there's sufficient 
# evidence to conclude that any of the slope parameters related to combinations of square footage of the 
# living area of the house and the neighborhood that the house is located in are statistically significant.

#Diagnostic plots of normal data
par(mfrow=c(2,2))
plot(house.lm)

#Histogram of normal data showing the distribution of the studentized residuals
sresid <- rstudent(house.lm) 
hist(sresid, freq=FALSE, main="Distribution of Studentized Residuals")
box()
xfit <- seq(min(sresid), max(sresid), length=40) 
yfit <- dnorm(xfit) 
lines(xfit, yfit, col='blue')

#Scatterplot matrix
pairs(SalePrice ~ GrLivArea2 + Neighborhood + GrLivArea2:Neighborhood, data = house)

#Add leverage, studentized residuals, and Cook's D to data set
house = transform(house, hat = hatvalues(house.lm))
house = transform(house, studres = studres(house.lm))
house = transform(house, cooksd = cooks.distance(house.lm))

#Inspect and remove problematic points
h = subset(house, studres <= -2 | studres >= 2)
View(h) #see which observation points are problematic in order to remove (15 observation points)
house = house %>% filter(Id != 176)
house = house %>% filter(Id != 212)
house = house %>% filter(Id != 251)
house = house %>% filter(Id != 411)
house = house %>% filter(Id != 608)
house = house %>% filter(Id != 643)
house = house %>% filter(Id != 667)
house = house %>% filter(Id != 725)
house = house %>% filter(Id != 729)
house = house %>% filter(Id != 808)
house = house %>% filter(Id != 889)
house = house %>% filter(Id != 1169)
house = house %>% filter(Id != 1299)
house = house %>% filter(Id != 1363)
house = house %>% filter(Id != 1424)

#Re-fit model without problematic points (ref is automatically set as 'BrkSide')
house.BrkSide <- lm(SalePrice ~ GrLivArea2 + Neighborhood + GrLivArea2:Neighborhood, data = house)
summary(house.BrkSide)
confint(house.BrkSide)

#Check diagnostic plots and see if there are any more problematic points
par(mfrow=c(2,2))
plot(house.BrkSide)
h = subset(house, studres <= -2 | studres >= 2)
View(h) #there are no more problematic points

#Fit models with other neighborhoods as reference points

#Fit model (ref=Edwards)
house2 <- within(house, Neighborhood <- relevel(Neighborhood, ref = 'Edwards'))
house.Edwards <- lm(SalePrice ~ GrLivArea2 + Neighborhood + GrLivArea2*Neighborhood, data = house2)
summary(house.Edwards)
confint(house.Edwards)

#Fit model (ref=NAmes)
house3 <- within(house, Neighborhood <- relevel(Neighborhood, ref = 'NAmes'))
house.NAmes <- lm(SalePrice ~ GrLivArea2 + Neighborhood + GrLivArea2*Neighborhood, data = house3)
summary(house.NAmes)
confint(house.NAmes)

#Visual
#The crossed lines on the graph suggest that there is an interaction effect
ggplot(house, aes(x=GrLivArea2, y=SalePrice, shape=Neighborhood, color=Neighborhood)) +
  geom_point(size=2) + 
  geom_smooth(method=lm, aes(fill=Neighborhood)) +
  ggtitle("Iowa House Data") +
  xlab("Above grade (ground) living area square feet (per 100 sqft)") +
  ylab("Property's sale price in dollars") +
  theme_bw() +
  theme(legend.position="bottom", plot.title = element_text(hjust = 0.5)) +
  labs(color='Neighborhood', shape='Neighborhood', fill='Neighborhood')

#Biggest increase of salesprice as it relates to GrLivArea (which has the steepest slope?)
# (1) BrkSide (2) NAmes (3) Edwards

# Question 2 #

#Read in dataset
train = read.csv("train.csv")
train = data.frame(train)
test = read.csv("test.csv")
test = data.frame(test)

#Histogram of SalesPrice
ggplot(train, aes(x=SalePrice)) + geom_histogram(color="white", fill="black") +
  labs(title="Distribution of Sales Price",x="Sales Price", y = "Frequency")+
  theme_classic() #not normally distributed

train.drop = train[,!(names(train) %in% c("Id","SalePrice"))] 
test.drop = test[,!(names(test) %in% c("Id"))]
house.data = rbind(train.drop, test.drop)
str(house.data)

#See which variables have NA values
sapply(house.data, function(x) sum(is.na(x)))

#Variables with the greatest number of missing observations:
# Alley: 2721
# PoolQC: 2909
# Fence: 2348
# MiscFeature: 2814
# FireplaceQu: 1420

#Drop variables with large number of NA values
house.data = house.data[,!(names(house.data) %in% c("Alley","FireplaceQu","PoolQC","Fence","MiscFeature"))]

#See which variables are numeric and impute by the mean
num = house.data[sapply(house.data,is.numeric)]
str(num)
num = kNN(num, k = 10, numFun = mean)
num = num[,-c(37:72)]

#See which variables are categorical and impute by mode (most frequently appearing value)
categorical = house.data[sapply(house.data,is.factor)]
str(categorical)
categorical = kNN(categorical, k = 10, numFun = mode)
categorical = categorical[,-c(39:76)]

#Combine again and see if any more NAs
house.data = cbind(num, categorical)
sapply(house.data, function(x) sum(is.na(x))) #no more NA values

#Remove more variables we don't need from the rest (i.e. random measurements, too many of the same values,
# years things were built, anything that wouldn't tell us much about 'SalePrice', etc.)

var = colnames(house.data)
remove = c('YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF2', 'X2ndFlrSF', 'LowQualFinSF', 'GarageYrBlt', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 'X3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'YrSold', 'Street', 'LandContour', 'Utilities', 'LandSlope', 'Exterior1st', 'Exterior2nd',  'Foundation', 'Heating', 'Electrical', 'GarageType', 'GarageCond', 'PavedDrive')
var = setdiff(var, remove)
house.data = house.data[, var] 

#Make all variables numeric
house.data = data.frame(lapply(house.data, function(x) as.numeric(as.factor(x))))
str(house.data) #there are now 47 variables

#Train and test
set.seed(100)
train.index = nrow(train)
test.index = train.index + 1
total = train.index + nrow(test)

#log variables
house.data$LotFrontage = log(house.data$LotFrontage)
house.data$LotArea = log(house.data$LotArea)

final.train = house.data[1:train.index, ]
final.test = house.data[test.index:total, ]

final.train$SalePrice = log(train$SalePrice)

#Diagnostic plots
lm = lm(SalePrice ~ ., data=final.train)
par(mfrow=c(2,2))
plot(lm)

#Add leverage, studentized residuals, and Cook's D to data set
final.train = transform(final.train, hat = hatvalues(lm))
final.train = transform(final.train, studres = studres(lm))
final.train = transform(final.train, cooksd = cooks.distance(lm))

#Inspect and remove problematic points and hat, studres, and cooksd at the end
h = subset(final.train, studres <= -3 | studres >= 3)
View(h)
final.train = final.train[-c(31,411,463,496,524,589,633,667,682,689,715,875,917,969,971,1183,1299,1325,1433),]
final.train = final.train[,!(names(final.train) %in% c("hat","studres","cooksd"))]

#Specify 10-fold cross-validation
train.control <- trainControl(method = "cv", number = 10)

#Backwards variable selection
back.model <- train(SalePrice ~ ., data = final.train,
                    method = "leapBackward",
                    tuneGrid = data.frame(nvmax = 1:10),
                    trControl = train.control)

#Identify the optimal tuning parameter
back.model$bestTune
#Look at all results from tuning process
back.model$results[10,] #RMSE
#View results of the selection process for the optimal parameter
summary(back.model$finalModel)
#Look at coefficients
coef(back.model$finalModel, 10)
#Predict
back.pred = predict(back.model, as.matrix(final.test), s=back.model$bestTune)

#Forward variable selection
forward.model <- train(SalePrice ~ ., data = final.train,
                       method = "leapForward", 
                       tuneGrid = data.frame(nvmax = 1:10),
                       trControl = train.control)

forward.model$bestTune
forward.model$results[10,] #RMSE
summary(forward.model$finalModel)
coef(forward.model$finalModel, 10)
#Predict
forward.pred = predict(forward.model, as.matrix(final.test), s=forward.model$bestTune)

#Stepwise variable selection
step.model <- train(SalePrice ~ ., data = final.train,
                    method = "leapSeq", 
                    tuneGrid = data.frame(nvmax = 1:10),
                    trControl = train.control)

step.model$bestTune
step.model$results[10,] #RMSE
summary(step.model$finalModel)
coef(step.model$finalModel, 10)
#Predict
step.pred = predict(step.model, as.matrix(final.test), s=step.model$bestTune)

#Based off of variables chosen from forward selection (because lowest RMSE)
final.lm = lm(SalePrice ~ LotArea+OverallQual+OverallCond+BsmtFinSF1+TotalBsmtSF+GrLivArea+GarageCars+CentralAir+GarageFinish+SaleCondition, data=final.train)
summary(final.lm)

final.train = final.train %>% dplyr::select(LotArea,OverallQual,OverallCond,BsmtFinSF1,TotalBsmtSF,GrLivArea,GarageCars,CentralAir,GarageFinish,SaleCondition,SalePrice)
final.test = final.test %>% dplyr::select(LotArea,OverallQual,OverallCond,BsmtFinSF1,TotalBsmtSF,GrLivArea,GarageCars,CentralAir,GarageFinish,SaleCondition)

#Penalized Regression Methods: Ridge, Lasso, EN

#Check VIF scores to see which variables have inflated standard errors
vif(final.lm)
res = cor(final.train) #correlation between variables

#Ridge Regression (note that cv.glmnet by default does 10-fold cross-validation)
rr.glmnet = cv.glmnet(as.matrix(final.train[,1:10]), final.train$SalePrice, alpha=0)
attributes(rr.glmnet)
best.lambda <- rr.glmnet$lambda.min
ridge.coef = coef(rr.glmnet, s=best.lambda)
#RMSE
sqrt(rr.glmnet$cvm[rr.glmnet$lambda == rr.glmnet$lambda.1se])
#Predict
ridge.pred = predict(rr.glmnet, as.matrix(final.test), s=best.lambda)

#Lasso
lasso.glmnet = cv.glmnet(as.matrix(final.train[,1:10]), final.train$SalePrice, alpha=1)
attributes(lasso.glmnet)
best.lambda2 = lasso.glmnet$lambda.min
lasso.coef = coef(lasso.glmnet, s=best.lambda2)
lasso.coef[lasso.coef !=0] #47 variables
#RMSE
sqrt(lasso.glmnet$cvm[lasso.glmnet$lambda == lasso.glmnet$lambda.1se])
#Predict
lasso.pred = predict(lasso.glmnet, newx=as.matrix(final.test), s=best.lambda2)

#Elastic Net
tcontrol = trainControl(method="repeatedcv", number=10, repeats=5)

en.glmnet = train(as.matrix(final.train[,1:10]), final.train$SalePrice, trControl=tcontrol,
                  method="glmnet", tuneLength=10)
attributes(en.glmnet)
en.glmnet$results
en.glmnet$bestTune
en.glmnet2 = en.glmnet$finalModel
en.coef = coef(en.glmnet2, s=en.glmnet$bestTune$lambda)
#RMSE
min(en.glmnet$results$RMSE)
#Predict
en.pred = predict(en.glmnet, as.matrix(final.test), s=en.glmnet$bestTune$lambda)

#Diagnostic plots for final model
par(mfrow=c(2,2))
plot(final.lm)

#Histogram of normal data showing the distribution of the studentized residuals
sresid <- rstudent(final.lm) 
hist(sresid, freq=FALSE, main="Distribution of Studentized Residuals")
box()
xfit <- seq(min(sresid), max(sresid), length=40) 
yfit <- dnorm(xfit) 
lines(xfit, yfit, col='blue')

### Elastic Net has best RMSE but Lasso will get best kaggle score ###

#Write as Excel files

#Backwards Predictions
WriteXLS(data.frame(back.pred))
#Forward Predictions
WriteXLS(data.frame(forward.pred))
#Stepwise Predictions
WriteXLS(data.frame(step.pred))

#Ridge Regression Predictions
WriteXLS(data.frame(ridge.pred))
#Lasso Predictions
WriteXLS(data.frame(lasso.pred))
#EN Predictions
WriteXLS(data.frame(en.pred))

#Note- we converted the log values that we obtained (in excel) and submitted that for our Kaggle scores


