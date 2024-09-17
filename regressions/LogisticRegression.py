#Logistic regression tries to answer yes/no questions
#predY is either 0 (negative/false) or 1 (positive/true)

#our values should be between [0, 1] (inclusive)

#the formula: sigmoid(Theta * X) = 1/(1 + e**-(Theta * X))

#Cost function: -Y*ln(formula) - (1 - Y)*ln(1 - formula)