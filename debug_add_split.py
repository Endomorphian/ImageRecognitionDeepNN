def add_shuffle_split_(X_target, X_various):
	# Add the target and various array
	a = np.array([(1, 2, 3, 4, 5), (0.1, 0.2, 0.3, 0.4, 0.5)])
	b = np.array([(6, 7, 8, 9, 95), (3, 3, 3, 3, 3)])
	y1 = np.array((1.1, 2.2, 3.3, 4.4, 5.5))
	y2 = np.array((6.6, 7.7, 8.8, 9.9, 9.595))

	c = np.concatenate((a, b), axis=1)
	d = np.concatenate((y1, y2), axis=0)
	d = np.reshape(d, (1,d.shape[0]))
	e = np.concatenate((c, d), axis=0)
	#print(c.shape, d.shape)
	#print(c)
	#print(d)
	#print("e", e.shape)

	# RANDOMIZE
	np.random.shuffle(e.T)
	#print("shuffle", e)
	#print(e.shape)
	x_n = e.shape[0]
	y_n = e.shape[1]

	#print(x_n, y_n)

	# Split X and Y
	X = e[0:x_n-1, :]
	Y = e[x_n-1:x_n, :]
	#print(X)
	#print(Y)

	# Split 70/30
	col1 = int(y_n*0.7)

	X_train = X[:, 0:col1]
	Y_train = Y[:, 0:col1]
	X_test = X[:, col1:y_n]
	Y_test = Y[:, col1:y_n]

	print(X_train, Y_train)
	print(X_test, Y_test)

	return X_train, Y_train, X_test, Y_test