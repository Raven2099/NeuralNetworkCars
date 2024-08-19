import pygame
import math
import os
import sys
import neat

# Collision detection
collision = (255, 255, 255)
curr_gen = 0
x = 60
y = 60
# ADJUST ACCORDING TO MAP
width = 1024
height = 1024
map_use = 'Circular_map.png'
generation = 0

class Car:
    def __init__(self):
        self.car = pygame.image.load('car.png').convert_alpha()
        self.car = pygame.transform.scale(self.car, (x, y))
        self.rotatedcar = self.car

        self.position = [320.9645481762549, 722.55073210080786]

        self.speed = 0
        self.speed_set = False
        self.angle = 0
        self.distance = 0
        self.time = 0
        self.center = [self.position[0] + x / 2, self.position[1] + y / 2]

        self.radars = []
        self.corners = []
        self.alive = True

    def draw(self, screen):
        screen.blit(self.rotatedcar, self.position)
        self.draw_radars(screen)

    def draw_radars(self, screen):
        for radar in self.radars:
            position = radar[0]
            pygame.draw.line(screen, (0, 255, 0), self.center, position, 1)
            pygame.draw.circle(screen, (0, 255, 0), position, 5)

    def if_collision(self, map):
        self.alive = True
        for point in self.corners:
            if map.get_at((int(point[0]), int(point[1]))) == collision:
                self.alive = False
                break

    def check_radar(self, degree, map):
        length = 0
        x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
        y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)
        while not map.get_at((x, y)) == collision and length < 500:
            length += 1
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)
        dist_border = int(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))
        self.radars.append([(x, y), dist_border])

    def update(self, map):
        if not self.speed_set:
            self.speed = 5
            self.speed_set = True
        self.rotatedcar = self.rotate_center(self.car, self.angle)
        self.position[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        self.position[0] = max(self.position[0], 20)
        self.position[0] = min(self.position[0], width - 120)
        self.distance += self.speed
        self.time += 1
        self.position[1] += math.sin(math.radians(360 - self.angle)) * self.speed
        self.position[1] = max(self.position[1], 20)
        self.position[1] = min(self.position[1], width - 120)

        self.center = [int(self.position[0]) + x / 2, int(self.position[1]) + y / 2]
        length = 0.5 * x
        left_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 30))) * length,
                    self.center[1] + math.sin(math.radians(360 - (self.angle + 30))) * length]
        right_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 150))) * length,
                     self.center[1] + math.sin(math.radians(360 - (self.angle + 150))) * length]
        left_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 210))) * length,
                       self.center[1] + math.sin(math.radians(360 - (self.angle + 210))) * length]
        right_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 330))) * length,
                        self.center[1] + math.sin(math.radians(360 - (self.angle + 330))) * length]
        self.corners = [left_top, right_top, left_bottom, right_bottom]

        self.if_collision(map)
        self.radars.clear()

        for d in range(-90, 120, 45):
            self.check_radar(d, map)

    def data(self):
        radars = self.radars
        val = [0, 0, 0, 0, 0]
        for i, radar in enumerate(radars):
            val[i] = int(radar[1] / 30)
        return val

    def check_alive(self):
        return self.alive

    def reward(self):
        # calculate reward
        return self.distance / (x / 2)

    def rotate_center(self, image, angle):
        orig_rect = image.get_rect()
        rotated_image = pygame.transform.rotate(image, angle)
        rotated_rect = orig_rect.copy()
        rotated_rect.center = rotated_image.get_rect().center
        rotated_image = rotated_image.subsurface(rotated_rect).copy()
        return rotated_image


def run_simu(genomes, config):
    nets = []
    cars = []

    pygame.init()
    screen = pygame.display.set_mode((width, height))

    for i, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0

        cars.append(Car())

    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 30)
    map = pygame.image.load(map_use).convert()

    global curr_gen
    curr_gen += 1

    counter = 0  # change later to better time limit

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)
        for i, car in enumerate(cars):
            out = nets[i].activate(car.data())
            choice = out.index(max(out))
            if choice == 0:
                car.angle += 10
            elif choice == 1:
                car.angle -= 10
            if choice == 2:
                if car.speed - 6>= 2:
                    car.speed -= 2
            else:
                car.speed += 1
        still_alive = 0
        for i, car in enumerate(cars):
            if car.check_alive():
                still_alive += 1
                car.update(map)
                genomes[i][1].fitness += car.reward()
        if still_alive == 0:
            break

        counter += 1
        if counter == 30 * 40:
            break
        screen.blit(map, (0, 0))
        for car in cars:
            if car.check_alive():
                car.draw(screen)

        text = font.render("Generation Number: " + str(curr_gen), True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (150, 50)
        screen.blit(text, text_rect)

        text = font.render("Alive: " + str(still_alive), True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (150, 75)
        screen.blit(text, text_rect)

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    config_path = "./neat_config.txt"
    config = neat.config.Config(neat.DefaultGenome,
                                neat.DefaultReproduction,
                                neat.DefaultSpeciesSet,
                                neat.DefaultStagnation,
                                config_path)

    # Create Population And Add Reporters
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    # Run Simulation For A Maximum of 100 Generations
    population.run(run_simu, 100)
