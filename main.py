#REFERENCES
import NeuralNetwork as nn
import MatrixMath as mx
import pygame
import settings
import interface
import numpy
import random
import math

#################
#   VARIABLES   #
#################
#GUI
buttons = []

#NETWORK STUFF
net = nn.Network([28**2, 16, 16, 10])


#-----------------FUNCTIONS-------------------------
#####################
#   MAIN FUNCTIONS  #
#####################
def init():
    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((settings.WIDTH, settings.HEIGHT))
    screen.fill((255,255,255))
    create_UI()
    data = divide_data(load_data())
    run(screen, data)

def run(screen:pygame.Surface, ml_data:tuple):
    #main loop
    while True:
        get_input(ml_data)
        draw(screen)
        update()

def get_input(ml_data):
    for event in pygame.event.get():
        #quit key
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
        #when keys are pressed down
        # elif event.type == pygame.KEYDOWN:
        #     match event.key:
        #         case pygame.K_END:
        #             net.train(inputs, outputs, 1000)
        #         case pygame.K_UP:
        #             net.print()
        #         case pygame.K_DOWN:
        #             for n in range(len(inputs)):
        #                 a = net.assess(inputs[n], outputs[n])
        #                 print(a)
        #when mouse clicks                
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            for button in buttons:
                assert type(button) == interface.Button, "Error: object in button list is not a button class"
                if not button.pressed:
                    if button.is_clicked(mouse_pos):
                        button.pressed = True
                        match button.label:
                            case "TRAIN":
                                train(ml_data, 100)
                                button.pressed = False

        # #when mouse moves
        # elif event.type == pygame.MOUSEMOTION:
        #     if whiteboard.drawing:
        #         mouse_pos = pygame.mouse.get_pos()
        #         whiteboard.draw(mouse_pos)
        # #when mouse unclicks
        # elif event.type == pygame.MOUSEBUTTONUP:
        #     if whiteboard.drawing:
        #         whiteboard.set_drawing(False)



def update():
    pass

def draw(screen:pygame.Surface):
    for button in buttons:
        assert type(button) == interface.Button
        screen.blit(button.get_surface(), button.get_position())

    pygame.display.update()


#--------------------------------------------------------------------------
#######################
#   GUI FUNCTIONS     #
#######################
def create_UI():   #called in init to make all the menu objects
    #Train button
    button_size = (200,100)
    button_pos = (settings.CENTRE[0] - button_size[0]/2, settings.HEIGHT - button_size[1]*2 - 10)
    assess_button = interface.Button(button_pos, button_size, None, "TRAIN")
    buttons.append(assess_button)


#########################
#   NETWORK FUNCTIONS   #
#########################
def train(data:tuple, epochs:int):
    training_data, test_data = data
    e = 0
    while e < epochs:
        e += 1
        net.train(training_data)
        numb_correct = 0
        net.test(test_data)
        x,_y = test_data
        print("Epoch {0}: {1} of {2}".format(e, numb_correct, len(x)))

def divide_data(data:tuple) -> tuple:
    inputs, outputs = data
    length = len(inputs)
    threshold = math.floor(0.8*length)
    training = [inputs[:threshold], outputs[:threshold]]
    testing = [inputs[threshold:], outputs[threshold:]]
    return (training, testing) #tuple of 2 lists

def load_data() -> tuple:
    inputs_np = numpy.load("data/new_img_array.npy")
    outputs_np = numpy.load("data/new_y_true.npy")
    #convert outputs from ints to matrices
    int_to_list = {
        0:[1,0,0,0,0,0,0,0,0,0],
        1:[0,1,0,0,0,0,0,0,0,0],
        2:[0,0,1,0,0,0,0,0,0,0],
        3:[0,0,0,1,0,0,0,0,0,0],
        4:[0,0,0,0,1,0,0,0,0,0],
        5:[0,0,0,0,0,1,0,0,0,0],
        6:[0,0,0,0,0,0,1,0,0,0],
        7:[0,0,0,0,0,0,0,1,0,0],
        8:[0,0,0,0,0,0,0,0,1,0],
        9:[0,0,0,0,0,0,0,0,0,1]
    }
    new_input_list = []
    new_output_list = []
    for i in range(inputs_np.shape[0]):
        new_matrix = mx.Matrix.from_list(inputs_np[i].tolist())
        new_input_list.append(new_matrix)
    for i in outputs_np:
        new_out = int_to_list[i]
        new_matrix = mx.Matrix.from_list(new_out)
        new_output_list.append(new_matrix)
    return (new_input_list, new_output_list)
#------------------------------------------------------------------------------
#####################
#   INIT PROGRAM    #
#####################

#start the program if run from this file
if __name__ == "__main__":
    init()
