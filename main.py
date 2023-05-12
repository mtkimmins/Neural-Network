import networkParts as nwp
import mathLib as ml
import pygame
import settings
import interface

buttons = []


net = nwp.Network([2,2,1], ml.Sigmoid, 0.1)

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

#-----------------FUNCTIONS-------------------------
#####################
#   MAIN FUNCTIONS  #
#####################

def init():
    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((settings.WIDTH, settings.HEIGHT))
    create_UI()
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
                    net.train(inputs, answers, 1000)
                case pygame.K_UP:
                    net.print()
                case pygame.K_DOWN:
                    for n in range(len(inputs)):
                        a = net.assess(inputs[n], answers[n])
                        print(a)
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

def update():
    pass

def draw(screen:pygame.Surface):
    for button in buttons:
        assert type(button) == interface.Button
        screen.blit(button.get_surface(), button.get_position())

    pygame.display.update()

#########################
#   LOCAL FUNCTIONS     #
#########################
def create_UI():
    #called in init to make all the menu objects
    #Training button
    button_size = (200,100)
    button_pos = (settings.CENTRE[0] - button_size[0]/2, settings.HEIGHT - button_size[1])
    train_button = interface.Button(button_pos, button_size, None, "TRAIN")
    train_button.set_effect(net.train(inputs, answers, 1000))
    buttons.append(train_button)

    #Assess button
    button_size = (200,100)
    button_pos = (settings.CENTRE[0] - button_size[0]/2, settings.HEIGHT - button_size[1]*2 - 10)
    assess_button = interface.Button(button_pos, button_size, None, "ASSESS")
    assess_button.set_effect(net.print())
    buttons.append(assess_button)

#------------------------------------------------------------------------------
#####################
#   INIT PROGRAM    #
#####################

#start the program if run from this file
if __name__ == "__main__":
    init()
