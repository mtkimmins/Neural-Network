class_name SketchBoard
extends GridContainer

func _ready():
	for i in 24*24:
		var a = ColorRect.new()
		a.size_flags_horizontal = SIZE_EXPAND
		a.size_flags_vertical = SIZE_EXPAND
		a.rect_min_size = Vector2(10,10)
		a.color = Color.black
		add_child(a)
	print("Done")

func _process(delta):
	if Input.is_mouse_button_pressed(BUTTON_LEFT):
			var tip_size = 10
			var m_pos = get_global_mouse_position()
			for i in get_children():
				var dist = (i.rect_global_position - m_pos).length()
				if dist < tip_size:
					var max_change = 50
					var percent = dist/tip_size
					var d = max_change * percent
					i.color += Color8(d,d,d,1)

func erase():
	for i in get_children():
		i.color = Color.black


func _on_Erase_pressed():
	erase()
