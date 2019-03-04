import matplotlib.pyplot as plt
import scipy
import pandas as pd
import numpy as np
import math 
import random
from tabulate import tabulate
random.seed(1)



class Self_Organizing_Map: 

	def __init__(self, weights, col_vector, learning_rate,*args):
		self.weights = weights
		self.col_vector = col_vector
		self.learning_rate = learning_rate



class Input_Animal_Cols:
	def __init__(self, name, col_vector):
		self.name = name
		self.col_vector = col_vector
		
	
	def col_vector_true_features(self, *args):
		args = list(args)
		col_vector = np.zeros((13,1))
		col_vector[args] = 1
		return col_vector


def Weight_Vector_Matrix_Gen(Rows,Cols):
	Weight_Vector_Matrix_IN = np.zeros((Rows, Cols, 13))
	Weight_Vectors  = np.zeros((13,1))
	for i in range(Rows):
		for k in range(Cols):
			for j in range(13):
				t = float((random.randint(-10,10))/100)
				Weight_Vector_Matrix_IN[i][k][j] = t
	return Weight_Vector_Matrix_IN

def dot_product_squared (input_col_vector, weights_col_vector):
	
	subtraction = np.subtract(input_col_vector, weights_col_vector)
	squared = np.dot(subtraction.transpose(), subtraction)
	return np.sqrt(squared)


def winner_neighbor_finder(Matrix,Winner_Pos, Rows_Len,Cols_Len, Radius):
	x = Winner_Pos[0]
	y = Winner_Pos[1]
	Neighbours = [(x,y-Radius), (x,y+Radius), (x-Radius, y), (x+Radius,y)]

	for i in Neighbours:
		if (i[0] < 0) or (i[0] > Rows_Len-1) or (i[1] < 0) or (i[1] > Cols_Len-1):
			Neighbours.remove(i)

	return Neighbours


def Find_Winner(Dimension_1, Dimension_2, Weight_Vectors_Matrix, Animal_col_vector):
	Distance_Matrix = np.zeros((10,10))
	for i in range(Dimension_1):		#Y-axis
		for j in range(Dimension_2):	#X-axis
			Weight_Vectors_Matrix[i][j]= np.array(Weight_Vectors_Matrix[i][j])
			Out = dot_product_squared(Animal_col_vector, Weight_Vectors_Matrix[i][j].reshape(13,1))
			Distance_Matrix[i][j] = Out


	Min_in_Matrix = np.where(Distance_Matrix==Distance_Matrix.min())
	return Min_in_Matrix

def Neighbour_function(Animal_col_vector, Winner_Vector_Pos, Neighbours_List, learning_rate, sigma):
	Neighbour_Function_List = []
	
	x = Winner_Vector_Pos[0]
	y = Winner_Vector_Pos[1]
	for i in Neighbours_List:
		Delta_X = x - i[0]
		Delta_Y = y - i[1]
		Square_X = Delta_X**2
		Square_Y = Delta_Y**2
		Out = np.exp(-(Square_X + Square_Y/sigma))
		Neighbour_Function_List.append(Out)
	
	Neighbour_Function_List = np.asarray(Neighbour_Function_List).reshape(len(Neighbours_List),1)
	Neighbour_Function_List_With_LR = Neighbour_Function_List*learning_rate
	return Neighbour_Function_List_With_LR

def weight_updating(Weight_Vectors_Matrix, Winner_Vector, Winner_Vector_Pos, input_col_vector, Neighbour_Functions, Neighbour_Vectors, Neighbours, learning_rate):
	
	
	#Nearest neighbour update of weights
	for i in range(len(Neighbour_Vectors)):
		Delta = np.subtract(input_col_vector, Neighbour_Vectors[i].transpose())
		
		Update = Neighbour_Functions[i]*Delta
		# print(Neighbour_Vectors[i])
		New_Neighbour_Vector = Neighbour_Vectors[i].reshape(13,1) + Update
		
		Weight_Vectors_Matrix[Neighbours[i]] = New_Neighbour_Vector.transpose()


	#Winner Vector update of Weights
	Delta = np.subtract(input_col_vector, Winner_Vector.reshape(13,1))

	Update = learning_rate*Delta    #Neighbour Function = 1 for Winning Vector
	Weight_Vectors_Matrix[Winner_Vector_Pos] = (Winner_Vector + Update.transpose())

	return Weight_Vectors_Matrix




#initialisation of Animal Classes / Initialisation  of Animal Column Vectors


SOM = Self_Organizing_Map(0,0,0.00135)
SOM.weights = Weight_Vector_Matrix_Gen(10,10)



Dove = Input_Animal_Cols('Dove', 0)
Dove.col_vector = Dove.col_vector_true_features(0,3,8,11)

Hen = Input_Animal_Cols('Hen',0)
Hen.col_vector = Hen.col_vector_true_features(0,3,8)

Duck = Input_Animal_Cols('Duck',0)
Duck.col_vector = Duck.col_vector_true_features(0,3,8,11,12)

Goose = Input_Animal_Cols('Goose',0)
Goose.col_vector = Goose.col_vector_true_features(0,3,8,11,12)

Owl = Input_Animal_Cols('Owl',0)
Owl.col_vector = Owl.col_vector_true_features(0,3,8,9,11)

Hawk = Input_Animal_Cols('Hawk',0)
Hawk.col_vector = Hawk.col_vector_true_features(0,3,8,9,11)

Eagle = Input_Animal_Cols('Eagle',0)
Eagle.col_vector=Eagle.col_vector_true_features(1,3,8,9,11)

Fox = Input_Animal_Cols('Fox',0)
Fox.col_vector = Fox.col_vector_true_features(1,4,5,9)

Dog = Input_Animal_Cols('Dog',0)
Dog.col_vector = Dog.col_vector_true_features(1,4,5,10)

Wolf = Input_Animal_Cols('Wolf',0)
Wolf.col_vector = Wolf.col_vector_true_features(1,4,5,7,9,10)

Cat = Input_Animal_Cols('Cat',0)
Cat.col_vector = Cat.col_vector_true_features(0,4,5,9)

Tiger = Input_Animal_Cols('Tiger',0)
Tiger.col_vector = Tiger.col_vector_true_features(2,4,5,9,10)

Lion = Input_Animal_Cols('Lion',0)
Lion.col_vector = Lion.col_vector_true_features(2,4,5,7,9,10)

Horse = Input_Animal_Cols('Horse',0)
Horse.col_vector = Horse.col_vector_true_features(2,4,5,6,7,10)

Zebra = Input_Animal_Cols('Zebra',0)
Zebra.col_vector = Zebra.col_vector_true_features(2,4,5,6,7,10)

Cow = Input_Animal_Cols('Cow',0)
Cow.col_vector = Cow.col_vector_true_features(2,4,5,6)


Animals = [Dove, Hen, Duck, Zebra, Cow, Wolf, Horse, Lion, Tiger, Cat, Dog, Fox, Eagle, Hawk, Goose, Owl]






Dove_Winner_Pos = Find_Winner(SOM.weights.shape[0],SOM.weights.shape[1],SOM.weights,Dove.col_vector)
Dove_Winner = SOM.weights[Find_Winner(SOM.weights.shape[0],SOM.weights.shape[1],SOM.weights,Dove.col_vector)]



### MODEL TRAINING: NEURON VECTORS 

Sigma = 3 
Delta_Sigma = 2/10000
for i in range(0, 10000):
	
	#Pick a Random Animal (input_col_vector)
	Random_Animal_Picked = random.choice(Animals)
	
	#Compute Winner Function: find winner vector/find winner vector position
	
	Animal_Winner_Pos = Find_Winner(SOM.weights.shape[0],SOM.weights.shape[1],SOM.weights,Random_Animal_Picked.col_vector)
	Animal_Winner = SOM.weights[Animal_Winner_Pos]
	

	#Find Neighbours of Winning Vector

	Animal_Winner_Neighbours = winner_neighbor_finder(SOM.weights, Animal_Winner_Pos, SOM.weights.shape[0], SOM.weights.shape[1], 1)


	# Create Array of Winning Vector's Neighbours and their corresponding weight.
	Animal_Winner_Neighbour_Vectors = []
	for i in Animal_Winner_Neighbours:
		Animal_Winner_Neighbour_Vectors.append(SOM.weights[i])
	Animal_Winner_Neighbour_Vectors = np.asarray(Animal_Winner_Neighbour_Vectors)

	Winner_Neighbour_functions = Neighbour_function(Random_Animal_Picked.col_vector, Animal_Winner_Pos, Animal_Winner_Neighbours, SOM.learning_rate, Sigma)

	weight_updating(SOM.weights, Animal_Winner, Animal_Winner_Pos, Random_Animal_Picked.col_vector, Winner_Neighbour_functions, Animal_Winner_Neighbour_Vectors, Animal_Winner_Neighbours, SOM.learning_rate)

	Sigma = Sigma - Delta_Sigma


#	MODEL TESTING: FINDING WINNER



Location_Map = np.zeros((10,10), dtype = object)

for i in Animals: 
	
	Testing_Winner = Find_Winner(SOM.weights.shape[0],SOM.weights.shape[1],SOM.weights,i.col_vector)

	Target_Normal = np.linalg.norm(i.col_vector)
	SOM_Normal = np.linalg.norm(SOM.weights[Testing_Winner])
	print("Winner Vector Magnitude % Accuracy: " + str((1-(abs(Target_Normal - SOM_Normal)/Target_Normal))*100)+ "\n","Map Location: " + str(Testing_Winner) + "\n","Animal Mapped: " + str(i.name) + '\n'  )
	
	
	# Location_Map[Testing_Winner] = i.name
	if Location_Map[Testing_Winner] != 0:
		Location_Map[Testing_Winner] = Location_Map[Testing_Winner]+ " " + i.name
	else :
		Location_Map[Testing_Winner] = i.name



f = open("Table.txt", "w+")


Table = tabulate(Location_Map, tablefmt = 'fancy_grid')

f.write(tabulate(Location_Map, tablefmt = 'fancy_grid'))
f.close()
