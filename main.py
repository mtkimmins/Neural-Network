import NeuralNetwork as nn
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as pt


(trainX, trainY), (testX, testY) = tf.keras.datasets.mnist.load_data()

Ytoy = {
    0: [1,0,0,0,0,0,0,0,0,0],
    1: [0,1,0,0,0,0,0,0,0,0],
    2: [0,0,1,0,0,0,0,0,0,0],
    3: [0,0,0,1,0,0,0,0,0,0],
    4: [0,0,0,0,1,0,0,0,0,0],
    5: [0,0,0,0,0,1,0,0,0,0],
    6: [0,0,0,0,0,0,1,0,0,0],
    7: [0,0,0,0,0,0,0,1,0,0],
    8: [0,0,0,0,0,0,0,0,1,0],
    9: [0,0,0,0,0,0,0,0,0,1],
}
n_list = []
n_list_t = []
for i in trainY:
    n_list.append(Ytoy[i])
for i in testY:
    n_list_t.append(Ytoy[i])

trainy1 = np.asarray(n_list)
testy1 = np.asarray(n_list_t)


net = nn.Network([28*28,16,16,10])

def print_net():
    print("PRE-ACTIVATIONS")
    
    for i in net.pre_activations:
        print(i)
        print("shape = " + str(i.shape))
    print("ACTIVATIONS")
    for i in net.activations:
        print(i)
        print("shape = " + str(i.shape))
    print("WEIGHTS")
    for i in net.weights:
        print(i)
        print("shape = " + str(i.shape))
    print("BIASES")
    for i in net.biases:
        print(i)
        print("shape = " + str(i.shape))

i = 0
while i < 200:
    print("EPOCH: " + str(i))
    net.load()
    net.train(1, (trainX, trainy1),(testX, testy1))
    net.save()
    i += 1





















# #TODO keep clean until nn is refactored


# #REFERENCES
# import NeuralNetwork as nn
# import pygame
# import settings
# import interface
# import numpy as np

# #VARIABLES
# buttons = []
# net = nn.Network([28*28, 16, 10])
# X = np.load("data/new_img_array.npy")
# Y = np.load("data/new_y_true.npy")


# #-----------------FUNCTIONS-------------------------
# #####################
# #   MAIN FUNCTIONS  #
# #####################
# def init():
#     pygame.init()
#     clock = pygame.time.Clock()
#     screen = pygame.display.set_mode((settings.WIDTH, settings.HEIGHT))
#     screen.fill((255,255,255))
#     create_UI()
#     run(screen)

# def run(screen:pygame.Surface):
#     #main loop
#     while True:
#         get_input()
#         draw(screen)
#         update()

# def get_input():
#     for event in pygame.event.get():
#         #quit key
#         if event.type == pygame.QUIT:
#             pygame.quit()
#             quit()
#         #when keys are pressed down
#         elif event.type == pygame.KEYDOWN:
#             pass
#         elif event.type == pygame.MOUSEBUTTONDOWN:
#             mouse_pos = pygame.mouse.get_pos()
#             for button in buttons:
#                 assert type(button) == interface.Button, "Error: object in button list is not a button class"
#                 if not button.pressed:
#                     button.pressed = True
#                     if (button.position[0] < mouse_pos[0] < button.position[0] + button.size[0]) and (button.position[1] < mouse_pos[1] < button.position[1] + button.size[1]):
#                         match button.label:
#                             case "TRAIN":
#                                 print("TRAINING...")
#                                 button.pressed = False

# def update():
#     pass

# def draw(screen:pygame.Surface):
#     for button in buttons:
#         assert type(button) == interface.Button
#         screen.blit(button.get_surface(), button.get_position())
#     pygame.display.update()


# #------------------------------------------------------------------
# #########################
# #   LOCAL FUNCTIONS     #
# #########################
# def create_UI():   #called in init to make all the menu objects
#     #Training button
#     button_size = (200,100)
#     button_pos = (settings.CENTRE[0] - button_size[0]/2, settings.HEIGHT - button_size[1])
#     train_button = interface.Button(button_pos, button_size, None, "TRAIN")
#     buttons.append(train_button)


# #------------------------------------------------------------------------------
# #####################
# #   INIT PROGRAM    #
# #####################

# #start the program if run from this file
# if __name__ == "__main__":
#     init()
