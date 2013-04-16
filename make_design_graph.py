from lib.processors import findFaceGetPulse
import networkx as nx

"""
Simple tool to visualize the design of the real-time image analysis

Everything needed to produce the graph already exists in an instance of the
assembly.
"""

#get the component/data dependancy graph (depgraph) of the assembly
assembly = findFaceGetPulse()
graph = assembly._depgraph._graph

#prune a few unconnected nodes not related to the actual analysis
graph.remove_node("@xin")
graph.remove_node("@xout")
graph.remove_node("driver")

#plot the graph to disc as a png image
ag = nx.to_agraph(graph)
ag.layout('dot')
ag.draw('design.png')
