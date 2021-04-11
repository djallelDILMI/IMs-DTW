# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 12:37:39 2017

@authors :	Djallel DILMI <djallel.outlook.fr>
			Laurent BARTHES <laurent.barthes@latmos.ipsl.fr>
			Cécile MALLET <cecile.mallet@latmos.ipsl.fr>
			Aymeric CHAZOTTES <aymeric.chazottes@latmos.ipsl.fr>

"""

import numpy as np
import math
from collections import defaultdict

#%%
class IMSDTW():
	"""
	Iterative Multiscale Dynamic Time Warping -IMSDTW-
	
	An algorithm based on a Multiscale Dynamic Time Warping (MsDTW) approach,
	it is based on the DTW algorithm applied on an iterative multiscale 
	framework.
	The proposed algorithm is well suited for time series allowing multiscale
	definition of the warping window then point-to-point pairing between 
	pairs of time step.
	
	read more on : 
		- https://link.springer.com/article/10.1007/s41060-019-00193-1
		- https://hal-insu.archives-ouvertes.fr/insu-02172756/document
		
	Parameters
	----------
		aggregation_step : int, default = 2
		
		radius : int ,default = 0
		
		adding_radius : { 'before', 'after'} , default ='before'
		
		dist : callable or func, default = euclidian distance
				it represents element-wise distance (X[i],Y[j])
		
		
	Returns
	-------
		dissimilarity : float
		
		path : list of list
	
	
	Examples
	--------
	
	>>> X = np.array([1, 2, 1, 4, 1, 0, 10, 2, 10, 4, 10, 0])
	>>> Y = np.array([1, 2, 1, 3, 1, 0, 10, 5, 10, 4, 10, 0,1])
	>>> imsdtw = IMSDTW(aggregation_step = 2, raidus=0)
	>>> dissim,path = imsdtw(X,Y)
	>>> print(dissim)
	0.938083151964686
	>>> path
	[[0, 0],
	 [1, 1],
	 [2, 2],
	 [3, 3],
	 [4, 4],
	 [5, 5],
	 [6, 6],
	 [7, 7],
	 [8, 8],
	 [9, 9],
	 [10, 10],
	 [11, 11],
	 [11, 12]]
	
	"""
	def __init__(self,aggregation_step = 2, radius = 0,
			adding_radius = 'before',dist = None):
		self.aggregation_step = aggregation_step
		self.radius           = radius
		self.dist             = self.__squared_difference if dist is None else dist
		self.adding_radius    = 'before' if(adding_radius not in ['before','after']) else adding_radius
		print(self.__str__())
	
	def __str__(self):
		return "IMSDTW constructed with params : \n \
                 aggregation_step : {self.aggregation_step} \n \
                 radius : {self.radius} \n \
                 dist : {self.dist} \n \
                 adding_radius : {self.adding_radius}".format(self=self)
	
	def __call__(self,x,y):
		"""
		Iterative Multiscale Dynamic Time Warping -IMSDTW-
		 # TODO : add description
		read more on : 
			- https://link.springer.com/article/10.1007/s41060-019-00193-1
			- https://hal-insu.archives-ouvertes.fr/insu-02172756/document
			
		Parameters
		----------
			x : array-like or list
			
			y : array-like or list
			
			
		Returns
		-------
			dissimilarity : float
			
			path : list of list
		
		
		Example
		-------
	
		>>> x = np.array([1, 2, 1, 4, 1, 0, 10, 2, 10, 4, 10, 0])
		>>> y = np.array([1, 2, 1, 3, 1, 0, 10, 5, 10, 4, 10, 0,1])
		>>> imsdtw = IMSDTW(aggregation_step = 2, raidus=0)
		>>> dissim,path = imsdtw(X,Y)
		>>> print(dissim)
		0.938083151964686
		>>> path
		[[0, 0],
		 [1, 1],
		 [2, 2],
		 [3, 3],
		 [4, 4],
		 [5, 5],
		 [6, 6],
		 [7, 7],
		 [8, 8],
		 [9, 9],
		 [10, 10],
		 [11, 11],
		 [11, 12]]
		"""
		lx,ly = len(x),len(y) # we keep values post-processing (__prep_output)
		x, y  = self.__prep_inputs(x, y)
		dissimilarity, path= self.__imsdtw(x, y)
		return self.__prep_output(dissimilarity,path,lx,ly,max(len(x),len(y)))
	
	def __prep_inputs(self,x,y):
		"""
		zero padding to match a shape = (aggregation_step)**n
		
		Parameters
		----------
			x : numpy.array or list
			
			y : numpy.array or list
			
		Returns
		-------
			x  : numpy.array with len = (aggregation_step)**n
			
			y  : numpy.array with len = (aggregation_step)**n
			
			
		Exemple
		-------
		
		>>> x = np.array([1, 2, 1, 4, 2.5])
		>>> y = np.array([1, 2, 1, 3])
		>>> x,y = __prep_inputs(x,y)
		>>> x
		array([  0,  1,  2,  1,  4,  2.5,  0,  0])
		>>> y
		array([  0,  0,  1,  2,  1,  3,  0,  0])
		
		"""
		x = np.asanyarray(x, dtype='float') # in case other formats were given
		y = np.asanyarray(y, dtype='float')
		# data preparation : reshape to match a length=power of aggregation_step
		length = max(len(x),len(y))
		length = self.aggregation_step**(math.floor(
						math.log(length,self.aggregation_step))+1)
		# zero padding to fit an aggregation step's pow length
		x, y = self.__zero_pad(x,length), self.__zero_pad(y,length) 
		return x, y
	
	
	def __zero_pad(self,x,length):
		"""
		zero padding to match a shape = length
		
		Parameters
		----------
			x : numpy.array or list
			
			length : int
			
		Returns
		-------
			numpy.array with len = length
		
		
		Exemple
		-------
		
		>>> x = np.array([1, 2, 1, 4, 2.5])
		>>> length = 8
		>>> x = __zero_pad(x, length)
		>>> x
		array([  0,  1,  2,  1,  4,  2.5,  0,  0])
		
		"""
		pad_width   = (length-len(x))//2
		if((length-len(x))%2):
			return np.pad(x, (pad_width, pad_width+1), 'constant', constant_values=0)
		else: 
			return np.pad(x, (pad_width, pad_width), 'constant', constant_values=0)
		
		
	def __prep_output(self,dissimilarity,path,lx,ly,length):
		"""
		 # TODO : add comments
		Parameters
		----------
			dissimilarity : float
			
			path : list of list
			
			lx : int
			
			ly : int
			
			length : int
		
		Returns
		-------
			dissimilarity : float
				
				dissimilarity = sqrt(2*dissimilarity/(lx+ly))
			
			path__ : list of list
			
			
		Examples
		--------
		
		>>> dissimilarity = 11.0
		>>> path = [(0, 0),
					(1, 0),
					(2, 1),
					(3, 2),
					(4, 3),
					(5, 4),
					(6, 5),
					(7, 6),
					(8, 7),
					(9, 8),
					(10, 9),
					(11, 10),
					(12, 11),
					(13, 12),
					(14, 13),
					(14, 14),
					(15, 15)]
		>>> lx, ly = 12, 13
		>>> length = 13
		>>> dissimilarity, path = __prep_output(dissimilarity,path,lx,ly,length)
		>>> dissimilarity
		  0.938083151964686
		>>> path
		 [[0, 0],
		  [1, 1],
		  [2, 2],
		  [3, 3],
		  [4, 4],
		  [5, 5],
		  [6, 6],
		  [7, 7],
		  [8, 8],
		  [9, 9],
		  [10, 10],
		  [11, 11],
		  [11, 12]]
		
		"""
		# TODO : optimze this function
		path_tab = np.array(path,dtype='int')
		sub_value = min(lx-1,ly-1)
		lim_inf_x=(length-lx)//2
		lim_inf_y=(length-ly)//2
		
		path_tab[:,0] -= lim_inf_x
		path_tab[:,1] -= lim_inf_y
		
		index_inf = (np.where(path_tab[:,0]==0)[0][0], np.where(path_tab[:,1]==0)[0][0])
		path_tab[min(index_inf):max(index_inf),np.argmax(index_inf)] = 0
		path_tab = path_tab[min(index_inf):,:]
		
		index_sup = (np.where(path_tab[:,0]>=lx)[0][0], np.where(path_tab[:,1]>=ly)[0][0])
		path_tab[min(index_sup):max(index_sup),np.argmin(index_sup)] = sub_value
		path_tab = path_tab[:max(index_sup),:]
		
		path__=[[i,j] for i,j in path_tab]
		return np.sqrt(2*dissimilarity/(lx+ly)),path__
	
	def __squared_difference(self,a, b):
		# TODO : hundle multidimentional timeseries
		# ideally : retrun np.linalg.norm(a-b)
		def __str__(self):
			return "squared_diffrence"
		return (a-b)**2
	
	def __imsdtw(self,x, y):
		"""
		 # TODO : add comments
		
		Parameters
		----------
			x : numpy.array or list
			
			y : numpy.array or list
			
		Returns
		-------
			dissimilarity : float
			
			path__ : list of list
			
		Example
		-------
		>>> x = np.array([0, 1, 2, 1, 4, 2.5, 0, 0],dtype='float')
		>>> y = np.array([0, 0, 1, 2, 1, 3, 0, 0],dtype='float')
		>>> dissimilarity, path = __imsdtw(x,y)
		>>> dissimilarity
		 1.25
		>>> path
		 [(0, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 5), (6, 6), (7, 7)]
		
		
		"""
		min_time_size = self.aggregation_step
		# at this resolution : 
		#   - no way to upscale series
		#   - classic DTW will be performed
		if len(x) < min_time_size or len(y) < min_time_size:
			return self.__dtw(x, y, None)
	
		x_upscaled = self.__upscale(x)
		y_upscaled = self.__upscale(y)
		_ , path   = self.__imsdtw(x_upscaled, y_upscaled)
		# difinition de l'ensemble des paths accéptés
		window     = self.__expand_window(path, len(x), len(y)) 
		return self.__dtw(x, y, window) # refelxivity until length = agg. step
	
	
	def __dtw(self,x, y, window):
		"""
		 # TODO : add comments
		Parameters
		----------
			x : numpy.array or list
			
			y : numpy.array or list
			
			window : list of tuple
		
		Returns
		-------
			dissimilarity : float
			
			path : list of list
			
		"""
		len_x, len_y = len(x), len(y)
		if window is None: # classic dtw
			window = [(i, j) for i in range(len_x) for j in range(len_y)]
		window = ((i + 1, j + 1) for i, j in window)
		D = defaultdict(lambda: (float('inf'),))
		D[0, 0] = (0, 0, 0)
		for i, j in window:
			dt = self.dist(x[i-1], y[j-1])
			D[i, j] = min((D[i-1, j-1][0]+dt, i-1, j-1),
						  (D[i-1, j][0]+dt, i-1, j),
						  (D[i, j-1][0]+dt, i, j-1),
						   key=lambda a: a[0]) # l'ordre est important : dans le cas d'une égalité l'algo prednra la diagonale
		path = []
		i, j = len_x, len_y
		while not (i == j == 0):
			path.append((i-1, j-1))
			i, j = D[i, j][1], D[i, j][2]
		path.reverse()
		return (D[len_x, len_y][0], path)
	
	def dtw(self,x, y, window):
		"""
		equivalent to implementation of classic dynamic time warping
		"""
		return self.__dtw(x, y, None)
	
	def __upscale(self,x,T=None):
		"""
		 #TODO : add comments
		Parameters
		----------
			x : array-like or list
			
			T : int
		
		Returns
		-------
			x_ : array-like or list of len = len(x)// aggregation_step
			
			
		Example
		-------
		
		>>> x = np.array([1, 2, 1, 4, 2.5, 0])
		>>> x_ = __upscale(x,T=3)
		>>> x_
		[1.3333333333333333, 2.1666666666666665]
		>>> x_ = __upscale(x,T=2)
		>>> x_
		[1.5, 2.5, 1.25]
		
		"""
		T= self.aggregation_step if T is None else T
		x_=[]
		for i in range(0, len(x), T):
			tmp_ = 0
			for t in range(T):
				tmp_+= x[i+t]
			x_.append( tmp_/T) 
		return x_

	def __add_radius(self,path):
		"""
		 # TODO : add comments
		Parameters
		----------
		
		Returns
		-------
		"""
		if(self.radius!=0):
			path_ = []#path.copy() on déclare un ensemble vide 
			# ajout des rayon , un éléement testé jusqu'à r=10
			for i, j in path:
				for a, b in ((i + a, j + b) 
							for a in range(-self.radius, self.radius+1) 
							for b in range(-self.radius, self.radius+1)):
					path_.append((a, b)) # on préfère els liste pour leur propriété ordonnée qui facilité le prcours en boucle
			return path_
		else:
			return path
	
	def __downscale_indexes(self, path):
		"""
		
		Parameters
		----------
		
		Returns
		-------
		
		Exemples
		--------
		
		"""
		window_ = set()
		i_precedent,j_precedent= -24, -24 
		for i, j in path:
			for a, b in [(i * self.aggregation_step + k,
						  j * self.aggregation_step + l) 
						  for k in range(self.aggregation_step) 
						  for l in range(self.aggregation_step)] :
				# une correspondance i,j engendre à la résolution suivante 2X2=4 corresp. candidates
				# __ __         __ __
				#|__|__| /___  |     |
				#|__|__| \     |__ __|
				# dans le cas où l'intégration se fait par 3 la liste des correp. candidate devraient etre 3x3=9:
				# (( i*3,j*3) , (i*3,j*3+1) , (i*3, j*3+2),
				# (( i*3+1,j*3), (i*3+1,j*3+1), (i*3+1,j*3+2),
				# (( i*3+2,j*3), (i*3+1,j*3+1), (i*3+1,j*3+2)
				window_.add((a, b))
			# on ajoute ici les pixels de liberté de l'IMs-DTW
			if(i==i_precedent+1 and j==j_precedent+1):
				window_.add((i * self.aggregation_step - 1,
							 j * self.aggregation_step))
				window_.add((i * self.aggregation_step ,
							 j * self.aggregation_step - 1))
			i_precedent,j_precedent=i,j
		return window_
		
	def __limit_window(self, window_,len_x,len_y):
		"""
		# TODO : add comments
		eliminate non coherente pixels like (-1,-5)
		
		Parameters
		----------
		
		Returns
		-------
		"""
		window = [] # 
		start_j = 0
		for i in range(0, len_x):
			new_start_j = None
			for j in range(start_j, len_y):
				if (i, j) in window_:
					window.append((i, j))
					if new_start_j is None:
						new_start_j = j
				elif new_start_j is not None:
					break
			start_j = new_start_j
		
		return window    
	
	def __expand_window(self,path, len_x, len_y):
		"""
		# TODO : add comments
		Parameters
		----------
		
		Returns
		-------
		"""
		if(self.adding_radius == 'before'):
			path_   = self.__add_radius(path)
			window_ = self.__downscale_indexes(path_)
			return self.__limit_window(window_,len_x,len_y)
		else:
			window_    = self.__downscale_indexes(path)
			window__   = self.__add_radius(window_)
			return self.__limit_window(window__,len_x,len_y)
