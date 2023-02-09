class_name Neuron
extends Node

enum LAYER {INPUT,HIDDEN,OUTPUT}

export(LAYER) var layer
export(float) var value
export(float) var bias = -10

var afferent_connections:Array = []
var afferent_connection_weights:Array = []

var efferent_connections:Array = []


#-------------------------------------------------------------------------------
func neuron_process()->void:
	var sum:float = 0.0
	for i in afferent_connections.size():
		sum += afferent_connections[i].value * afferent_connection_weights[i]
	value = internal_function(sum - bias)

func internal_function(input:float)->float:
	var result = 0.0
	var e = 2.71828
	result = 1/(1-pow(e,input))
	return result
