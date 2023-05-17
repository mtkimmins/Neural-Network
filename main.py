import NeuralNetwork as nn
import pygame
import settings
import interface
import numpy

buttons = []
whiteboard = interface.Canvas((0,0),(0,0))

# net = nwp.Network([2,2,1], ml.Sigmoid, 0.1)



inputs = [
    [0,0],
    [1,1],
    [1,0],
    [0,1]
]

answers = [
    [0],
    [0],
    [1],
    [1]
]
# X,Y = numpy.load("data/new_img_array.npy"), numpy.load("data/new_y_true.npy")
# print(X.shape)
# x1 = []
# for i in range(X.shape[0]):
#     row = []
#     for j in range(X.shape[1]):
#         row.append(X[i,j])
#     x1.append(row)
# x2 = ml.Matrix.from_list(x1)

# x2.print()

# print(Y.shape)
# y1 = []
# for i in range(Y.shape[0]):
#     y1.append(Y[i])
# y2 = ml.Matrix.from_list(y1)


# y2.print()

#-----------------FUNCTIONS-------------------------
#####################
#   MAIN FUNCTIONS  #
#####################

def init():
    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((settings.WIDTH, settings.HEIGHT))
    screen.fill((255,255,255))
    create_UI(screen)
    run(screen)

def run(screen:pygame.Surface):
    #main loop
    while True:
        get_input()
        draw(screen)
        update()

def get_input():
    for event in pygame.event.get():
        #quit key
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
        #when keys are pressed down
        elif event.type == pygame.KEYDOWN:
            pass
            # match event.key:
            #     case pygame.K_END:
            #         net.train(inputs, answers, 1000)
            #     case pygame.K_UP:
            #         net.print()
            #     case pygame.K_DOWN:
            #         for n in range(len(inputs)):
            #             a = net.assess(inputs[n], answers[n])
            #             print(a)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            for button in buttons:
                assert type(button) == interface.Button, "Error: object in button list is not a button class"
                if not button.pressed:
                    button.is_event_clicked(mouse_pos)
                    # button.pressed = True
                    # if (button.position[0] < mouse_pos[0] < button.position[0] + button.size[0]) and (button.position[1] < mouse_pos[1] < button.position[1] + button.size[1]):
                    #     match button.label:
                    #         case "TRAIN":
                    #             print("TRAINING...")
                    #             net.train(inputs, answers, 1000)
                    #             print("TRAINED 1000 TIMES")
                    #             button.pressed = False
                    #         case "ASSESS":
                    #             print("ASSESSING...")
                    #             for i in range(len(inputs)):
                    #                 passed = net.assess(inputs[i], answers[i])
                    #                 if not passed:
                    #                     print("FAIL: network uncalibrated")
                    #                     button.pressed = False
                    #                     return
                    #             print("SUCCESS: network calibrated")
                    #             button.pressed = False
                    #     button.pressed = False
            #check whiteboard
            if whiteboard.is_hovered(mouse_pos):
                whiteboard.draw(mouse_pos)
                whiteboard.set_drawing(True)
        
        elif event.type == pygame.MOUSEMOTION:
            if whiteboard.drawing:
                mouse_pos = pygame.mouse.get_pos()
                whiteboard.draw(mouse_pos)

        elif event.type == pygame.MOUSEBUTTONUP:
            if whiteboard.drawing:
                whiteboard.set_drawing(False)



def update():
    pass

def draw(screen:pygame.Surface):
    for button in buttons:
        assert type(button) == interface.Button
        screen.blit(button.get_surface(), button.get_position())
    screen.blit(whiteboard.get_surface(), whiteboard.get_position())


    pygame.display.update()

#########################
#   LOCAL FUNCTIONS     #
#########################
def create_UI(screen:pygame.Surface):   #called in init to make all the menu objects
    #Canvas setup
    wb_size = (28,28)
    wb_pos = (settings.CENTRE[0] - wb_size[0]/2, 50 - wb_size[1]/2)
    whiteboard.set_size(wb_size)
    whiteboard.set_position(wb_pos)

    #Training button
    button_size = (200,100)
    button_pos = (settings.CENTRE[0] - button_size[0]/2, settings.HEIGHT - button_size[1])
    train_button = interface.Button(button_pos, button_size, None, "TRAIN")
    buttons.append(train_button)

    #Assess button
    button_size = (200,100)
    button_pos = (settings.CENTRE[0] - button_size[0]/2, settings.HEIGHT - button_size[1]*2 - 10)
    assess_button = interface.Button(button_pos, button_size, None, "ASSESS")
    buttons.append(assess_button)

#------------------------------------------------------------------------------
#####################
#   INIT PROGRAM    #
#####################

#start the program if run from this file
if __name__ == "__main__":
    init()
