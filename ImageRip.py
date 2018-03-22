import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os

DIR_TARGET = 'cats'         # The subdirectory containing the images to predict
DIR_VARIOUS = 'various'     # The subdirectory with the other images

def get_filenames(dir_name):
	return os.listdir('./' + dir_name + '/')

def import_images(folder): # Create an array X with images (no_of_images, x, y, 3-rgb-layers)
	file_names = get_filenames(folder)
	n = len(file_names)
	
	temp = []
	for i in range(0, n):
		temp.append(mpimg.imread('.\/' + folder + '\/' + file_names[i]))

	X = np.array(temp) 
	#print("X shape", X.shape)

	return X

def flatten_array(input_X): # Flatten array(a, b, c, d) to (b*c*d, a)
	X_flatten = input_X.reshape(input_X.shape[0], -1).T 

	return X_flatten

def show_image(image):
	
	imgplot = plt.imshow(image)
	plt.show()

def add_shuffle_split(X_target, X_various):
	# Add the target array to various array
	y1 = np.zeros((X_target.shape[1], 1))
	y2 = np.zeros((X_various.shape[1], 1))

	c = np.concatenate((X_target, X_various), axis=1)
	d = np.concatenate((y1, y2), axis=0)
	d = np.reshape(d, (1,d.shape[0]))
	e = np.concatenate((c, d), axis=0)

	# RANDOMIZE
	np.random.shuffle(e.T)
	x_n = e.shape[0]
	y_n = e.shape[1]

	X = e[0:x_n-1, :]
	Y = e[x_n-1:x_n, :]

	# Split 70/30
	col1 = int(y_n*0.7)

	X_train = X[:, 0:col1]
	Y_train = Y[:, 0:col1]
	X_test = X[:, col1:y_n]
	Y_test = Y[:, col1:y_n]

	return X_train, Y_train, X_test, Y_test

def reshape_test(X_train, Y_train):
	None

X_target = import_images(DIR_TARGET)
X_various = import_images(DIR_VARIOUS)

#show_image(X_various[63,:,:,:])

X_target = flatten_array(X_target)
X_various = flatten_array(X_various)

X_train, Y_train, X_test, Y_test = add_shuffle_split(X_target, X_various)

print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

reshape_test(1, 2)

# FLATTEN DATA BY /256 !!!