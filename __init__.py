"""
This file is part of the flask+d3 Hello World project.
"""
import json
import flask
from flask import request
import numpy as np
import networkx as nx

import csv
import numpy as np
import pandas as pd
import collections
from collections import OrderedDict
import glob
import simplejson as json


app = flask.Flask(__name__)

COUNT = 0

@app.route("/hello")
def index():
	"""
	When you request the root path, you'll get the index.html template.

	"""
	return flask.render_template("index.html")


@app.route("/")
def gindex():
	"""
	When you request the gaus path, you'll get the gaus.html template.

	"""
	mux = request.args.get('mux', '')
	muy = request.args.get('muy', '')
	muz = request.args.get('muz', '')
	if len(mux)==0: mux="Chief Executives"
	if len(muy)==0: muy="20"
	if len(muz)==0: muz="0.94"

	'''
	Begin backend
	''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
	'''
# coding: utf-8

	total_df = pd.DataFrame.from_csv('edge_weights.txt', index_col=False)
	sim_4code = pd.DataFrame.from_csv('sim_filtered.csv', index_col=False)
	soc_codes = list(sim_4code.columns.values)[1:]
	soc_titles = pd.DataFrame.from_csv('soc_titles.csv', index_col=False) # complete list of (soc, title)

	# 1, define dictionary: index in similarity matrix -> occupation name

	# soc: soc4 code
	# ind: index in similarity matrix
	# title: the actual text name of soc4  
	ind_title = collections.OrderedDict()
	title_ind = collections.OrderedDict()

	for i in range(len(soc_codes)):
		code = soc_codes[i]
		title = soc_titles[soc_titles['SOC_CODE']==code+'.00']['SOC_TITLE'].iloc[0]
		ind_title[i] = title
		title_ind[title] = i


	# # 2, given a threshold count, gives all its neighbors
	# # in the similarity matrix, grab a column, get top neighbors
	threshold = int(muy)
	occupation = mux # 'Chief Executives' # this will be the center of the cluster
	occupation_id = title_ind[occupation]

	sim_csv = np.genfromtxt('sim_filtered.csv', delimiter=',')
	sim_tmp = np.delete(sim_csv, (0), axis=0)
	sim_mat = np.delete(sim_tmp, (0), axis=1)


	sorted_ind = np.argsort( sim_mat[occupation_id] )
	top_neighbor_inds = sorted_ind[::-1][:threshold]
	center = occupation
	node_list = [ind_title[q] for q in top_neighbor_inds] # x is used later


	# # 3, threshold of edges between neighbors: i.e. 0.9
	# # get all the edges
	sim_thres = float(muz)
	edges = []
	for i in range(len(top_neighbor_inds)):
		for j in range(i):
			source_ind = top_neighbor_inds[i]
			target_ind = top_neighbor_inds[j]
			if sim_mat[source_ind][target_ind] >= sim_thres:
				edges.append( (ind_title[source_ind], ind_title[target_ind]) )


	# # 4, weight of each edge
	weight_df = total_df[['Occupation','Change Percent', 'Job openings']]
	weighted_edges = []
	for edge in edges:
		source_formated = edge[0][:1].upper() + edge[0][1:].lower()
		target_formated = edge[1][:1].upper() + edge[1][1:].lower()
		
		first_entry = weight_df[ weight_df['Occupation']==source_formated ]
		second_entry = weight_df[ weight_df['Occupation']==target_formated ]
				
		first_cp = first_entry['Change Percent'].iloc[0]
		second_cp = second_entry['Change Percent'].iloc[0]
		
		first_jo = first_entry['Job openings'].iloc[0]
		second_jo = second_entry['Job openings'].iloc[0]
			
		if first_cp <= second_cp:
			weight = second_cp - first_cp
			weighted_edges.append( (edge[0], edge[1], weight) )
		else:
			weight = first_cp - second_cp
			weighted_edges.append( (edge[1], edge[0], weight) )    

	# json_file = graph2json(center, node_list, weighted_edges)	

	network = {}
	netNode = [{"occupation":x, "center":center} for x in node_list]
	network["nodes"] = netNode
	nodePos = {title:ind for ind,title in enumerate(node_list)}
	posNode = {ind:title for ind,title in enumerate(node_list)}
	netLink = [ {"source":nodePos[x[0]], "target":nodePos[x[1]], "value":x[2]} for x in weighted_edges]
	network["links"] = netLink

	# Graph analysis code:
	G = nx.Graph()
	for i in range(len(node_list)):
		G.add_node(node_list[i])

	for i in range(len(netLink)):
		src = posNode[ netLink[i]["source"] ]
		tgt = posNode[ netLink[i]["target"] ]
		wgt = netLink[i]["value"]
		G.add_edge(src, tgt, weight=wgt)

	# PageRank and Diameters
	diameter = -1
	group = {}
	if G.number_of_edges()==0: # G is null graph
		ranking = dict.fromkeys(node_list, -1)
		diameter = 0
		group_values = range(len(node_list))
		group = dict(zip(node_list, group_values))

	else:	                   # G contains edges
		ranking = nx.pagerank(G)

		# update diameter & group
		if nx.number_connected_components(G.to_undirected())==1: # G has all nodes connected
			diameter = nx.diameter(G)
			group = dict.fromkeys(node_list, 0)
			print "in 1"
		else: 				   # G has some nodes connected
			print "in 2"
			maxDia = -1
			group_ind = 0
			for subG in nx.connected_component_subgraphs(G):
				group.update( dict.fromkeys(subG.nodes(), group_ind) )
				group_ind += 1
				if nx.diameter(subG)>maxDia:
					maxDia = nx.diameter(subG)
			diameter = maxDia
	
	# print ranking
	# print diameter
	print group

	appendRank_Group = network["nodes"]
	for item in appendRank_Group:
		if item["occupation"] in ranking:
			item.update( {"rank": ranking[item["occupation"]]} )
		else:
			item.update( {"rank": -1} )
		item.update( {"group": group[item["occupation"]]} )

	network["nodes"] = appendRank_Group

	'''
	End backend
	''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
	'''

	ret = json.dumps(network)
	global COUNT
	COUNT += 1
	return flask.render_template("gaus.html",mux=mux,muy=muy,muz=muz,network=ret,diameter=diameter,counter=COUNT)


@app.route("/data")
@app.route("/data/<int:ndata>")
def data(ndata=100):
	"""
	On request, this returns a list of ``ndata`` randomly made data points.

	:param ndata: (optional)
		The number of data points to return.

	:returns data:
		A JSON string of ``ndata`` data points.

	"""
	x = 10 * np.random.rand(ndata) - 5
	y = 0.5 * x + 0.5 * np.random.randn(ndata)
	A = 10. ** np.random.rand(ndata)
	c = np.random.rand(ndata)
	return json.dumps([{"_id": i, "x": x[i], "y": y[i], "area": A[i],
		"color": c[i]}
		for i in range(ndata)])

@app.route("/gdata")
@app.route("/gdata/<float:mux>/<float:muy>")
def gdata(ndata=100,mux=.5,muy=0.5): # deleted ndata=100
	"""
	On request, this returns a list of ``ndata`` randomly made data points.
	about the mean mux,muy

	:param ndata: (optional)
		The number of data points to return.

	:returns data:
		A JSON string of ``ndata`` data points.

	"""

	x = np.random.normal(mux,.5,ndata)
	y = np.random.normal(muy,.5,ndata)
	A = 10. ** np.random.rand(ndata)
	c = np.random.rand(ndata)


	# original json dumps
	return json.dumps([{"_id": i, "x": x[i], "y": y[i], "area": A[i],
		"color": c[i]}
		for i in range(ndata)])

# 5, write to json
def graph2json(center, nodes, edges):
	graph = {}
	nodeList = [{"occupation":x, "center":center} for x in nodes]
	graph["nodes"] = nodeList
	node_pos = {title:ind for ind,title in enumerate(nodes)}

	linkList = [ {"source":node_pos[x[0]], "target":node_pos[x[1]], "value":x[2]} for x in edges]
	graph["links"] = linkList
	
	return json.dumps(graph)


if __name__ == "__main__":
	import os

	port = 8100

	# Open a web browser pointing at the app.
	os.system("open http://localhost:{0}/".format(port))

	# Set up the development server on port 8000.
	app.debug = True
	app.run(port=port)
