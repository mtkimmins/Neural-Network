import networkParts as nwp
import mathLib as ml
import pygame
import settings #pygame init there
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
def init():
    #pygame init is in settings file as it is imported off the bat, and settings depend on pygame init, like font creation
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((settings.WIDTH, settings.HEIGHT))
    run(screen)

def run(screen):
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
                    button.pressed = True
                    if (button.position[0] < mouse_pos[0] < button.position[0] + button.size_x) and (button.position[1] < mouse_pos[1] < button.position[1] + button.size_y):
                        match button.label:
                            case "TRAIN":
                                print("TRAINING...")
                                net.train(inputs, answers, 1000)
                                print("TRAINED 1000 TIMES")
                                button.pressed = False
                                break
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
                                break

def update():
    pass

def draw(screen):
    #draw button for training
    _button_size = (1,50)
    
    train_button = interface.Button(_button_size, None, "TRAIN")
    _button_pos = (settings.WIDTH/2 - train_button.size_x/2, settings.HEIGHT - train_button.size_y - 140)
    buttons.append(train_button)
    train_button.set_position(_button_pos)
    screen.blit(train_button.surface, _button_pos)
    #draw button for assessment
    button_size = (200,100)
    
    assess_button = interface.Button(button_size, None, "ASSESS")
    button_pos = (settings.WIDTH/2 - button_size[0]/2, settings.HEIGHT - button_size[1])
    buttons.append(assess_button)
    assess_button.set_position(button_pos)
    screen.blit(assess_button.surface, button_pos)
    #draw test node
    # a = pygame.Rect((50,50,100,100))
    # b = pygame.Surface((100,100))
    # node = interface.Node(a, 1)
    # pygame.draw.circle(screen, (255,255,255), center=(50,50), radius=5.0)
    # screen.blit(b, node.rect)
    

    #draw the network
    net_width = (net.layers*settings.PERCEPTRON_RADIUS) + (net.layers-1)*settings.X_SEP
    x = settings.WIDTH/2 - net_width/2
    
    net_render = nwp.NetworkRender(net)
    screen.blit(net_render.get_surface(), screen.get_rect())
    # for i in range(net.layers):
    #     current_layer_height = net.layer_list[i]
    #     col_height = current_layer_height*settings.PERCEPTRON_RADIUS + (current_layer_height-1)*settings.Y_SEP
    #     y = settings.HEIGHT/2 - col_height/2

    #     for p in range(net.layer_list[i]):
    #         pygame.draw.circle(screen, (255,255,255), (x, y), 10)
    #         y += settings.PERCEPTRON_RADIUS*2 + settings.Y_SEP

    #     x += settings.PERCEPTRON_RADIUS*2 + settings.X_SEP

    #KEEP THIS -- ESSENTIAL
    pygame.display.update()



#####################
#   INIT PROGRAM    #
#####################

#start the program if run from this file
if __name__ == "__main__":
    init()
