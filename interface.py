#TODO make sure the text scales with the actual size of the button

import math
import pygame
import NeuralNetwork as nn

#TODO
#-when click elsewhere than buttons, all buttons are disabled
#-all buttons diable after something else is clicked, but as long as its on that button right after being clicked for the first time,
#this is irrelevant of actual position on the button, can move around within the button and it still is active
#-make a consequence setter


#############
#   BUTTON  #
#############
class Button:
    def __init__(self, position:list, size:list, image=None, label:str=""):
        #the position is top-left corner
        self.position = position
        self.size = size
        self.base_color = (255,255,255)
        self.text_color = (0,0,0)
        self.label = label
        self.image = image
        self.surface = pygame.Surface(self.size)
        self.pressed = False
  
        self.surface_center = (self.surface.get_rect().size[0]/2, self.surface.get_rect().size[1]/2)

        #use an image and blit to surface, else blit a blank rect
        if image is not None:
            pass
        else:
            pygame.draw.rect(self.surface, self.base_color, self.surface.get_rect())

        #blit the text
        self.add_font(40)
        
    
    #---------------AUX FUNCS---------------
    def add_font(self, size):
        font_name = "Arial"
        font_size = size
        font = pygame.font.SysFont(font_name, font_size)
        text = font.render(self.label, True, self.text_color)
        #ensure the font fits the size specified
        while text.get_width() > self.surface.get_width() or text.get_height() > self.surface.get_height():
            #break conditions
            if font_size == 1: break

            font_size -= 1
            font = pygame.font.SysFont(font_name, font_size)
            text = font.render(self.label, True, self.text_color)
        
        #blit to surface
        text_center = (text.get_rect().size[0]/2, text.get_rect().size[1]/2)
        self.surface.blit(text, (self.surface_center[0] - text_center[0], self.surface_center[1] - text_center[1]))
        




    #------------SETTERS--------------------
    def set_size(self, new_size:list):
        self.size = new_size
    
    def set_position(self, new_position:list):
        self.position = new_position

    def set_color(self, new_color:tuple):
        self.color = new_color
    
    def set_text(self, new_text:str):
        self.text = new_text

    #---------------GETTERS-----------------
    def get_surface(self) -> pygame.Surface:
        return self.surface

    def get_position(self) -> list:
        return self.position
    
    def get_size(self) -> list:
        return self.size
    
    def is_pressed(self) -> bool:
        return self.pressed

    #-------EXTERNALS-------------------
    def is_clicked(self, event_pos) -> bool:
        if (self.position[0] + self.size[0]) > event_pos[0] > self.position[0] and (self.position[1] + self.size[1] > event_pos[1] > self.position[1]):
            return True
        return False


#############
#   NODE    #
#############
class Node:
    def __init__(self, value:float, size:list):
        self.width_margin = 5
        self.height_margin = 5


        self.value = value
        #get adjusted size for text
        font = pygame.font.SysFont("Arial", 12)
        self.text = font.render(str(self.value), True, (255,255,255))
        width = max(size[0], self.text.get_rect().width)
        height = max(size[1], self.text.get_rect().height)
        self.size = (width, height)

        self.output_nodes = []
        self.surface = pygame.Surface(self.size)

        self.set_surface()

    def set_output_nodes(self, output_nodes:list):
        self.output_nodes = output_nodes

    def set_surface(self):
        size = (self.surface.get_rect().width, self.surface.get_rect().height)
        pygame.draw.circle(self.surface, (255,0,0), (size[0]/2, size[1]/2), size[0]/2)
        self.surface.blit(self.text, self.surface.get_rect())


#############
#   CANVAS  #
#############
class Canvas:
    def __init__(self, size:tuple, position:tuple) -> None:
        self.size = size
        self.position = position
        self.surface = pygame.Surface(self.size)
        self.draw_color = (255,255,255) #White default
        self.background_color = (0,0,0) #black default
        self.drawing_radius = 0
        self.drawing = False
        self.disabled = True

    def draw(self, draw_position:tuple):
        if self.drawing:
            #use the x coord provided
            # if self.is_hovered(draw_position):
            for i in range(-self.drawing_radius, self.drawing_radius + 1):
                for j in range(-self.drawing_radius, self.drawing_radius + 1):
                    self.surface.set_at((math.floor(draw_position[0] - self.position[0]) + i, math.floor(draw_position[1] - self.position[1]) + j), self.draw_color)

    #----------------SETTERS-------------------------
    def clear(self):
        self.surface = pygame.Surface(self.size)
    
    def set_draw_color(self, new_value:tuple):
        self.draw_color = new_value

    def set_background_color(self, new_value:tuple):
        self.background_color = new_value
        self.surface.fill(self.background_color)

    def set_size(self, new_size:tuple):
        self.size = new_size
        self.surface = pygame.Surface(self.size)
    
    def set_position(self, new_position:tuple):
        self.position = new_position
    
    def set_drawing(self, new_value:bool):
        if self.disabled: return
        self.drawing = new_value
    
    #------------------GETTERS---------------------
    def get_surface(self) -> pygame.Surface:
        return self.surface

    def get_position(self) -> tuple:
        return tuple(self.position)
    
    def is_hovered(self, mouse_position:tuple):
        if (self.position[0] + self.size[0]) > mouse_position[0] > self.position[0] and (self.position[1] + self.size[1] > mouse_position[1] > self.position[1]):
            return True
        return False
    
    def get_surface_as_list(self) -> list:
        surface_value_list = []
        for i in range(self.surface.get_width()):
            for j in range(self.surface.get_height()):
                surface_value_list.append(self.surface.get_at((i,j))[0])
        return surface_value_list


    #---------------EXTERNALS-------------------


#############
#   GRAPH   #
#############
class Graph:
    def __init__(self, size, position):
        self.size = size
        self.position = position
        self.background_color = (0,0,0)
        self.draw_color = (255,255,255)
        self.axis_color = (175,175,175)
        self.points = []
        self.x_max = 100
        self.y_max = 100
        self.surface = pygame.Surface(self.size)
        self.edge_margin = 5
        self.origin = (self.edge_margin, self.size[1] - self.edge_margin)

    def draw(self):
        #draw the axes
        x_axis = pygame.draw.line(self.surface, self.axis_color, self.origin, (self.size[0] - self.edge_margin, self.origin[1]))
        y_axis = pygame.draw.line(self.surface, self.axis_color, (self.edge_margin, self.edge_margin), self.origin)
        
        #plot points
        for i in self.points:
            assert type(i) == tuple, "point not a tuple"
            if self.y_max << int(i[1]):
                self.y_max = max(round(i[1]),100)
            pos = (self.origin[0] + i[0], self.origin[1] + (self.size[1] - self.edge_margin*2)*(i[1]/self.y_max))
            pygame.draw.circle(self.surface, self.draw_color, pos, 1)

    def add_point(self, new_coord:tuple):
        self.points.append(new_coord)
        self.draw()


#########################################
#   RENDERER FOR NETWORK THRU PYGAME    #
#########################################

class NetworkRender:
    #TODO
    #1)create all the nodes
    #2)connect all the nodes
    #3)draw onto surface
    def __init__(self, network:nn.Network):
        self.network = network

        self.v_gap = 10
        self.h_gap = 20
        self.node_size = pygame.Rect(0,0,10,10)
        self.surface_size = self.calculate_surface_size()
        
        self.surface = pygame.Surface(self.surface_size)
        self.surface.set_colorkey((255,255,255))
        self.create_nodes()


    def get_surface(self)->pygame.Surface:
        return self.surface

    def create_nodes(self):
        pass
        #for each layer
        for i in range(len(self.network.layer_list)):
            #for each node in each layer
            for j in range(self.network.layer_list[i]):
                node = Node(round(self.network.matrices[i*3].matrix[j][0]), self.node_size)
                coords = (i*(self.node_size.width + self.h_gap),
                          j*(self.node_size.height + self.v_gap))
                self.surface.blit(node.surface, coords)

    def calculate_surface_size(self)->tuple:
        width = (self.network.layers * self.node_size.width) + (self.network.layers - 1) * self.h_gap
        widest_layer = max(self.network.layer_list)
        height = (widest_layer * self.node_size.height) + (widest_layer - 1) * self.v_gap
        return (width, height)

