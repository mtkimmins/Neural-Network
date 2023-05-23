#REFERENCES
import NeuralNetwork as nn
import MatrixMath as ml
import pygame
import settings
import interface
import numpy
import random

#vars
EPOCH_MAX = 5000
training_cycles = 0
epochs = 0
MANUAL_VERIFY_MAX = 10
manual_verification_counter = 0
paused = False
manual_override = False
counter = 0



#GUI
buttons = []
whiteboard = interface.Canvas((0,0),(0,0))
graph = interface.Graph((200,200), (0,0))

#load the data into list
inputs = []
outputs = []

c_inputs = list(inputs)
c_outputs = list(outputs)

inputs_np = numpy.load("data/new_img_array.npy")
outputs_np = numpy.load("data/new_y_true.npy")
# print(outputs_np.tolist())
#dict to translate output to correct array
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
#inputs
for i in range(inputs_np.shape[0]):
    i_matrix = ml.Matrix.from_list(inputs_np[i].tolist())
    inputs.append(i_matrix)
#outputs
answers = outputs_np.tolist()
for i in range(len(answers)):
    out = int_to_list[answers[i]]
    o_matrix = ml.Matrix.from_list(out)
    # print(o_matrix.matrix)
    outputs.append(o_matrix)





#NETWORK STUFF
net = nn.Network([28**2, 16, 16, 10], ml.Sigmoid, 0.01)

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
            match event.key:
                case pygame.K_END:
                    net.train(inputs, outputs, 1000)
                case pygame.K_UP:
                    net.print()
                case pygame.K_DOWN:
                    for n in range(len(inputs)):
                        a = net.assess(inputs[n], outputs[n])
                        print(a)
                        
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            for button in buttons:
                assert type(button) == interface.Button, "Error: object in button list is not a button class"
                if not button.pressed:
                    if button.is_clicked(mouse_pos):
                        button.pressed = True
                        match button.label:
                            case "TRAIN":
                                print("TRAINING...")
                                net.train(inputs, answers, 1000)
                                print("TRAINED 1000 TIMES")
                                button.pressed = False
                            case "ASSESS":
                                print("ASSESSING...")
                                for i in range(len(inputs)):
                                    passed = net.assess(inputs[i], answers[i])
                                    if not passed:
                                        print("FAIL: network uncalibrated")
                                        button.pressed = False
                                        return
                                print("SUCCESS: network calibrated")
                                button.pressed = False
                            case "PREDICT":
                                print("PREDICTING...")
                                canvas_data:list = whiteboard.get_surface_as_list()
                                net.predict(canvas_data)
                                global paused
                                if input("Is this correct? Y/N") == "N":
                                    global manual_verification_counter
                                    

                                    manual_verification_counter = 0
                                    paused = False
                                    whiteboard.clear()
                                else:
                                    print("SUCCESS!!!!")
                                    net.verified = True
                                    paused = False
                                    
                                    whiteboard.clear()
                        button.pressed = False
            #check whiteboard
            if whiteboard.is_hovered(mouse_pos):
                whiteboard.set_drawing(True)
                whiteboard.draw(mouse_pos)
                
        
        elif event.type == pygame.MOUSEMOTION:
            if whiteboard.drawing:
                mouse_pos = pygame.mouse.get_pos()
                whiteboard.draw(mouse_pos)

        elif event.type == pygame.MOUSEBUTTONUP:
            if whiteboard.drawing:
                whiteboard.set_drawing(False)



def update():
    # net.train(inputs, outputs, 1)
    # net.print()
    global counter
    global c_inputs
    global c_outputs

    if c_inputs == []:
        c_inputs = list(inputs)
        c_outputs = list(outputs)

    i = random.randint(0,len(c_inputs)-1)
    c_in = c_inputs.pop(i)
    c_out = c_outputs.pop(i)
    cost = net.train_to_cost(c_in, c_out)
    print(cost)
    # graph.add_point((counter, net.train_to_cost(c_in, c_out)))
    counter += 1

    if cost < 1:
        print("MANUAL VERIFICATION REQUIRED...\n PLEASE DRAW AND PREDICT")
        whiteboard.disabled = False
        paused = True
    
    # global paused
    # global manual_override

    # if not paused:
    #     if not net.verified:
    #         train_ai()
    #         global manual_verification_counter
    #         if manual_verification_counter < MANUAL_VERIFY_MAX:
    #             for n in range(10):
    #                 i = random.randrange(0, len(outputs))
    #                 # print(outputs[i].matrix)
    #                 if not net.assess(inputs[i], outputs[i]):
    #                     print("FAIL TO VERIFY")
    #                     break
    #                 else:
    #                     net.verified = True
    #                     print("VERIFIED")
    #                     whiteboard.disabled = False
    #             manual_verification_counter += 1
    #         else:
    #             if manual_override:
    #                 print("MANUAL VERIFICATION REQUIRED...\n PLEASE DRAW AND PREDICT")
    #                 whiteboard.disabled = False
    #                 paused = True
    #             else:
    #                 manual_verification_counter = 0

def draw(screen:pygame.Surface):
    for button in buttons:
        assert type(button) == interface.Button
        screen.blit(button.get_surface(), button.get_position())
    screen.blit(whiteboard.get_surface(), whiteboard.get_position())

    graph.draw()
    screen.blit(graph.surface, graph.position)




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

    # #Training button
    # button_size = (200,100)
    # button_pos = (settings.CENTRE[0] - button_size[0]/2, settings.HEIGHT - button_size[1])
    # train_button = interface.Button(button_pos, button_size, None, "TRAIN")
    # buttons.append(train_button)

    #Predict button
    button_size = (200,100)
    button_pos = (settings.CENTRE[0] - button_size[0]/2, settings.HEIGHT - button_size[1]*2 - 10)
    assess_button = interface.Button(button_pos, button_size, None, "PREDICT")
    buttons.append(assess_button)

def train_ai():
    e = 0
    if e < EPOCH_MAX:
        global epochs
        net.train(inputs, outputs, 1)
        e += 1
        epochs += 1
    print("EPOCH " + str(epochs) + " TRAINED...")

#------------------------------------------------------------------------------
#####################
#   INIT PROGRAM    #
#####################

#start the program if run from this file
if __name__ == "__main__":
    init()
