import networkParts as nwp
import mathLib as ml
import pygame
import settings

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
    #pygame init
    pygame.init()
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
                    for n in inputs:
                        net.predict(n)

def update():
    pass

def draw(screen):
    net_width = (net.layers*settings.PERCEPTRON_RADIUS) + (net.layers-1)*settings.X_SEP
    x = settings.WIDTH/2 - net_width/2
    
    for i in range(net.layers):
        current_layer_height = net.layer_list[i]
        col_height = current_layer_height*settings.PERCEPTRON_RADIUS + (current_layer_height-1)*settings.Y_SEP
        y = settings.HEIGHT/2 - col_height/2

        for p in range(net.layer_list[i]):
            pygame.draw.circle(screen, (255,255,255), (x, y), 10)
            y += settings.PERCEPTRON_RADIUS*2 + settings.Y_SEP

        x += settings.PERCEPTRON_RADIUS*2 + settings.X_SEP
    pygame.display.update()



#####################
#   INIT PROGRAM    #
#####################

#start the program if run from this file
if __name__ == "__main__":
    init()