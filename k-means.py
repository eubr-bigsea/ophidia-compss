#!/bin/python2

import sys, json, numpy, random, argparse

from PyOphidia import cube, client
from pycompss.api.constraint import constraint
from pycompss.api.task import task
from pycompss.api.parameter import *

#Simple function to get list of values from a cube
def extractValue(dataCube):

	data = dataCube.export_array()
	return data["measure"][0]["values"]

#Function to compute euclidean distance
@task(dataCube=IN, mu=IN, user=IN, pwd=IN, host=IN, port=IN, returns=list)
def computeDistance(dataCube, mu, user, pwd, host, port):
	cube.Cube.setclient(user, pwd, host, port)

	#Build query
	difference=""
	types=""
	for i in range(0,len(mu)):
		types=types+"OPH_DOUBLE|"
		difference=difference+",oph_sum_scalar('OPH_DOUBLE','OPH_DOUBLE',oph_get_subarray('OPH_DOUBLE','OPH_DOUBLE',measure,"+str(1+i)+",1),-"+str(mu[i])+")"

	types=types[:-1]
	concat="oph_concat('"+types+"','OPH_DOUBLE'"+difference+")"
	apply_query="oph_math('OPH_DOUBLE','OPH_DOUBLE',oph_operator('OPH_DOUBLE','OPH_DOUBLE',oph_operator_array('OPH_DOUBLE|OPH_DOUBLE','OPH_DOUBLE',"+concat+","+concat+",'OPH_MUL'),'OPH_SUM'),'OPH_MATH_SQRT')"

	#Run query
	newCube = dataCube.apply(query=apply_query, measure='Test', check_type='no', ncores=2)
	values = extractValue(newCube)
	dist = [v[0] for v in values]
	return dist

#Function to compute cluster centroids
@task(cluster=IN, centroid=IN, returns=list)
def computeCentroids(cluster, centroid):
	if not cluster:
		return centroid
	else:
		return numpy.mean(cluster, axis = 0).tolist()

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='K-means implementation with Ophidia and COMPSs')
	parser.add_argument('-P','--port', default="11732", help='Port of Ophidia instance')
	parser.add_argument('-H','--hostname', default='127.0.0.1', help='Hostname of the Ophidia instance')
	parser.add_argument('-u','--username', help='Name of user on the Ophidia instance', required=True)
	parser.add_argument('-p','--password', help='Password of the account on the Ophidia instance', required=True)
	args = parser.parse_args()

	port = args.port
	host = args.hostname
	user = args.username
	pwd = args.password

	from pycompss.api.api import compss_wait_on

	#Initialize
	cube.Cube.setclient(user, pwd, host, port)
	try:
		cube.Cube.createcontainer(container="Test",dim='points|features',dim_type='double|double',hierarchy='oph_base|oph_base')
	except:
		pass

	#N = elements, M = features, K = clusters
	N = 10000
	M = 2
	K = 8
	
	#Create NxM random cube
	dataCube = cube.Cube.randcube(ncores=2, nhost=1, container="Test",nfrag=2, ntuple=str(N/2), measure="Test", measure_type="double", exp_ndim=1, dim="points|features", concept_level='c|c',dim_size=str(N)+"|"+str(M))
	points = extractValue(dataCube)

	#Create sample for initial centroids
	centroids = random.sample([[random.random()*1000 for i in range(M)] for i in range(1000)], K)

	if M == 2:
		from matplotlib import pyplot			
		color_map = pyplot.get_cmap('rainbow')
		colors = [color_map(float(i)/float(K)) for i in range(0, K)]

	#Run Kmean until convergence of centroids
	it = 1
	while True:
		distances = [[] for m in range(0,len(centroids))]
		#Compute euclidean distances for each centroid
		for i in range(0,len(centroids)):
			distances[i].append(computeDistance(dataCube, centroids[i], user, pwd, host, port))

		distances = compss_wait_on(distances)

		#Compute indexes to assign original values to clusters
		indexes = numpy.argmin(distances, axis=0)[0]
		clusters  = [[] for m in range(0,len(centroids))]
		for i in range(0, len(indexes)):
			clusters[indexes[i]].append(points[i])

		#Recompute centroids
		newCentroids = [[] for m in range(0,len(centroids))]
		for i in range(0,len(centroids)):
			newCentroids[i] = computeCentroids(clusters[i], centroids[i])
		
		newCentroids = compss_wait_on(newCentroids)

		if M == 2:
			#Plot current cluster setup (works only for 2 features)
			pyplot.figure()
			for i, v in enumerate(clusters):
				pyplot.scatter([r[0] for r in v], [r[1] for r in v], color=colors[i], marker='+')
			for i, v in enumerate(newCentroids):
				pyplot.scatter(v[0], v[1], color=colors[i], marker='D', edgecolor='black', linewidth='2')
			pyplot.title("Iteration: "+str(it))
			pyplot.savefig('cluster'+str(it)+'.png', bbox_inches='tight')

		#Ending condition
		if set([tuple(a) for a in newCentroids]) == set([tuple(a) for a in centroids]) or it >= 20:
			break

		centroids = newCentroids
		it=it+1

	print(newCentroids)

	#Delete intermediate results
	cube.Cube.client.submit("oph_delete cube=[container=Test]")
	try:
		cube.Cube.deletecontainer(container='Test', delete_type='physical', hidden='no')
	except:
		pass

	if M == 2:
		#Create single gif and delete temporary files
		import imageio, os
		gifFrames = []
		for img in ['cluster'+str(i)+'.png' for i in range(1,it+1)]:
			gifFrames.append(imageio.imread(img))
			try:	
				os.remove(img)
			except:
				pass
		args = { 'duration': 0.5 }
		imageio.mimsave('clustering.gif', gifFrames, 'gif', **args)
	
