#TODO make sure the text scales with the actual size of the button

import pygame

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
        self.effect_function = None
  
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
    def set_effect(self, new_effect_function):
        self.effect_function = new_effect_function

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
    def is_event_clicked(self, event_pos):
        if (self.position[0] + self.size[0]) > event_pos[0] > self.position[0] and (self.position[1] + self.size[1] > event_pos[1] > self.position[1]):
            print(self.effect_function)
            pressed = True
            if self.effect_function != None:
                self.effect_function


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