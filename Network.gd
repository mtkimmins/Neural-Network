class_name Network
extends Node

var input_layer_size = 24*24
var hidden_layer_size = 16
var output_layer_size = 10

onready var input_layer = $InputLayer
onready var hidden_layer = $HiddenLayer
onready var output_layer = $OutputLayer

export(NodePath) var pixel_grid
export(NodePath) var submit


func _ready():
	var pixels = get_node(pixel_grid).get_children()
	for i in input_layer_size:
		var neuron = Neuron.new()
		neuron.value = pixels[i].color.r / 255
		input_layer.add_child(neuron)
	for i in hidden_layer_size:
		var neuron = Neuron.new()
		neuron.value = 0
		neuron.afferent_connections = input_layer.get_children()
		for n in neuron.afferent_connections.size():
			neuron.afferent_connection_weights.append(0)
		hidden_layer.add_child(neuron)
	for i in output_layer_size:
		var neuron = Neuron.new()
		neuron.value = 0
		neuron.name = str(i)
		neuron.afferent_connections = hidden_layer.get_children()
		for n in neuron.afferent_connections.size():
			neuron.afferent_connection_weights.append(0)
		output_layer.add_child(neuron)


func _on_Submit_pressed()->void:
	var pixels = get_node(pixel_grid).get_children()
	for i in input_layer.get_child_count():
		input_layer.get_children()[i].value = pixels[i].color.r
	for i in hidden_layer.get_children():
		i.neuron_process()
	for i in output_layer.get_children():
		i.neuron_process()

func train()->void:
	randomize()
	var target = randi() % 10
	print(target)
	yield(get_node(submit), "pressed")
	get_node(pixel_grid).erase()
	train()


func _on_Training_toggled(button_pressed):
	if button_pressed:
		train()
