# Propagate
# For each class
    # if either of ratio > 0.6(arbitrarily chosen value)  
    #   -> propagte label for the entire neighbourhood
    # else
    #   -> only use the existing labels



# Automated labelling
    # for each candidate
    # label 1, l2, l3 ... ln = Lambda:example -> [1, 0, 1, -1 .. 0]

    # LAMBDA = [ lambdas(x) for x in X]


# Label we want to predict
# build a query which reflects some properties of that label and form a neighbourhood
# Subset
# Propagate the label to neighbourhood

# each neighbourhood as one labelling function
# if in neighbourhood -> 1
# elif not in neibghbourhood -> -1
# else (i.e. unsure) -> 0

# alpha_i = lambda_i labels correctly
# beta_i = lambda_i did label

# ... where lambda is a classification function generated from a neighbourhood


# M = model distribution
# loss0 = alpha0*beta0*lambda[0](y=Y)+ (1-alpha0)*beta0*lambda[0](y=-Y)+ (1-beta0)*lambda[0](y=0)
# loss1 = lambda[1]         ... where lambda is a classification function generated from a neighbourhood
# ...
# lossN = lambda[N]


# Objective function
# max(P(lambda, Y) ~ M(alpha, beta)) wrt. alpha and beta


# Let's first do it until here

# The learning algorithm

# 
# Alpha = tf.vector(1, num_labeling_functions)
# Beta = tf.vector(1, num_labeling_functions)
# Lambda = [lambda0, lambda1, lambda2 ... lambdaN]

# while step < n_step(= 10000):
#   Lambda(X) = ...
#   model_distribution = Alpha*Beta*Lambda(=right) + (1-Alpha)*Beta*Lambda(=left) + Beta*Lambda(=0)
#   loss = tf.reduce(model_distribution - X)**2
#   gradient = tf.sgd(loss)
#   tf.update(gradient)