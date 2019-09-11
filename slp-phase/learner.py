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


import tensorflow as tf
# Build lambda -> tensify inputs??
# for i in range(number_of_neighborhood)
#   array_of_indeces_for_examples_in_neighborhood_i = []
#   fill in by SLP
#   index_array_if_in_neighbourhood_X = onehotify (potentially propagated)neighbourhood lambda
# 
    # lambda0 R->R3 = tf.reduce(
    #    tf.matmul(index_array_if_in_neighbourhood_X[0], onehot_index_of_target[0]) -> 1,N*N*1=1
    #    tf.matmul(index_array_if_notin_neighbourhood_X[0], onehot_index_of_target[0]) -> 1,N*N*1=1
    #    tf.matmul(index_array_of_all_zeros[0], onehot_index_of_target[0]) -> 1,N*N*1=1
    #)


# Get data ready
# n_classes=3
x = tf.placeholder("float", [None, n_input]) # array of one_hot_encoded vectors shape=(S, n_input)
y_true = tf.placeholder("float", [None, 3]) # hard label, augmented to the number of lambda functions
#  n_input x 3

# Build model distribution and predictor
# num_lambda=10
Alpha = tf.Variable(tf.random_normal([10])) #tf.vector(1, num_labeling_functions)
Beta = tf.Variable(tf.random_normal([10])) #tf.vector(1, num_labeling_functions)
Lambda = tf.vstack([
    tf.matmul(lambdas[0] * x),
    tf.matmul(lambdas[1] * x),
    tf.matmul(lambdas[2] * x),

    tf.matmul(lambdas[10] * x),
])  # -> shape = (10, 3)
log_model_distribution = tf.reduce(Alpha*Beta*Lambda[0:1] + (1-Alpha)*Beta*Lambda[1:2] + Beta*Lambda[2:3])
# -> shape=(1, 1)

y_pred = tf.

# Define loss and optimiser
xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)
cost = tf.reduce_mean(xentropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Launch the graph
with tf.Session() as sess:
	# Initialise the variables and run
	with tf.device("/cpu:0"):
		init = tf.global_variables_initializer()
		sess.run(init)
		# Restore saved model if any
		# Training cycle
		
		epoch_saved = data_saved['var_epoch_saved'].eval()
		print(epoch_saved)
		for epoch in range(epoch_saved, epoch_saved+training_epochs):
			avg_cost = 0.
			total_batch = int(len(data_training)/batch_size)
			
			# Loop over all batches
			for i in range(total_batch):				
				batch = data_training[np.random.choice(data_training.shape[0], size=batch_size,  replace=True)]
				batch_x = batch[:, :len_X]
				batch_y = batch[:, len_X:]

				# Run optimization op (backprop) and cost op (to get loss value)
				_, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y_true: batch_y})
				# Compute average loss
				avg_cost += c / total_batch
			
			# Display logs per epoch step
			if epoch % display_step == 0:
				print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
		print("Optimization Finished!")

		# Test model
		correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
		
        # Calculate accuracy
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		print("Accuracy:", accuracy.eval({x: data_test[:,:len_X], y_true: data_test[:,len_X:]}))

		# Save the variables
		sess.run(data_saved['var_epoch_saved'].assign(epoch_saved + training_epochs))

		save_path = saver.save(sess, ".\\model\\model.ckpt")
		print("Model saved in file: %s" % save_path)