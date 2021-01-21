import numpy as np 

'''
This script contains metrics to ensure the quality of photmoetric redshift estimation in this work
'''

def bias(z_phot,z_spec):

	'''
		The bias measures the deviation of the estimated photometric redshift from the true(i.e., the spetroscopic redshift)
	'''

	b = np.abs((z_phot-z_spec)/(1+z_spec))

	return b

def scatter(z_phot,z_spec):
	'''
		The scatter between the true redshift and the photometric redshift
	'''
	sigma = np.sqrt(np.abs(((z_phot-z_spec)/(1+z_spec))**2))

	return sigma

