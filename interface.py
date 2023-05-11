#TODO make sure the text scales with the actual size of the button

import settings
import pygame

class Button:
    def __init__(self, size:list, image=None, label:str="", base_color=(255,255,255), text_color=(0,0,0)):
        #the position is top-left corner
        self.position = [0,0]
        self.base_color = base_color
        self.text_color = text_color
        self.label = label
        self.image = image
        self.pressed = False
        
        #make surface the biggest dimensions between the text and the size specified
        text = settings.font.render(self.label, True, self.text_color)
        text_center = (text.get_rect().size[0]/2, text.get_rect().size[1]/2)
        self.size_x = max(text.get_rect().size[0], size[0])
        self.size_y = max(text.get_rect().size[1], size[1])
        self.surface = pygame.Surface((self.size_x, self.size_y))
        self.surface_center = (self.surface.get_rect().size[0]/2, self.surface.get_rect().size[1]/2)

        #use an image and blit to surface, else blit a blank rect
        if image is not None:
            pass
        else:
            pygame.draw.rect(self.surface, self.base_color, self.surface.get_rect())

        #blit the text
        self.surface.blit(text, (self.surface_center[0] - text_center[0], self.surface_center[1] - text_center[1]))
    
    #------------SETTER FUNCTIONS-----------

    def set_size(self, new_size:list):
        self.size = new_size
    
    def set_position(self, new_position:list):
        self.position = new_position

    def set_color(self, new_color):
        self.color = new_color
    
    def set_text(self, new_text):
        self.text = new_text



class Node:
    def __init__(self, value:float, size:list):
        self.value = value
        #get adjusted size for text
        self.text = settings.font.render(str(self.value), True, (255,255,255))
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
