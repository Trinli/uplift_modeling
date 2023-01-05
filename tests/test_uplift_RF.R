# Toy example for how to use
# upliftRF

# "uplift" needs to be installed.
# install.packages("uplift")
library(uplift)

# Create some data
# features:
x1 = rnorm(100)
x2 = rnorm(100)
tmp = as.data.frame(cbind(x1, x2))
# binary label
y = rbinom(100, 1, .5)
# treatment label
t = rbinom(100, 1, .5)

# Train model:
# Do they use some "default" values in the paper for the parameters?
model = upliftRF(x=as.data.frame(tmp), y=y, ct=t,
                 # Guelman & al. used [i for i in range(len(colnames(x)))]:
                 mtry = 1,  # number of variables to try for each tree ("random subset")
                 # Guelman & al. used [100 * i for i in range(20)]
                 n_tree = 3,  # number of trees in forest. Guelman & al used []
                 split_method = 'KL',  # can also be 'ED', 'Chisq', 'L1', 'Int'
                 minsplit=3,  # minsplit - 1 is the stopping criterion (no more splits)
                 verbose=TRUE)

# Test model (prediction phase):
x1 = rnorm(10)
x2 = rnorm(10)
tmp2 = as.data.frame(cbind(x1, x2))
res = predict.upliftRF(model, as.data.frame(tmp2))
# As output, we get conversion probabilities with and without treatment:
print(res)
