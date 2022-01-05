import pygame, sys, random
import numpy as np
from random import randint

class Block:
    def __init__(self, x_pos, y_pos, color, width, height):
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.color = color
        self.width = width
        self.height = height
        self.rect = pygame.Rect(self.x_pos, self.y_pos, self.width, self.height)

class Player(Block):
    def __init__(self, x_pos, y_pos, color, speed = 5):
        super().__init__(x_pos, y_pos, color, width=5, height=HEIGHT/8.5)
        self.speed = speed
        self.movement = 0

    def screen_limit(self):
        if self.rect.top <= 0:  self.rect.top = 0
        if self.rect.bottom >= HEIGHT:  self.rect.bottom = HEIGHT

    def move_up(self):
        self.movement = -5

    def move_down(self):
        self.movement = 5

    def stop(self):
        self.movement = 0

    def move(self):
        self.rect.y += self.movement
        self.screen_limit()

    def draw(self):
        pygame.draw.rect(screen, self.color, self.rect)
        
    def get_y(self):
        return self.rect.centery
    
    def get_x(self):
        return self.rect.centerx

class Ball(Block):
    def __init__(self, x_pos, y_pos, color, paddles, speed_x = 4, speed_y = 4):
        super().__init__(x_pos, y_pos, color, width=10, height=10)
        self.speed_x = speed_x * random.choice((-1,1))
        self.speed_y = speed_y * random.choice((-1,1))
        self.paddles = paddles

    def move(self):
        self.rect.x += self.speed_x
        self.rect.y += self.speed_y
        self.screen_limit()
        self.collisions()
		
    def screen_limit(self):
        if self.rect.top <= 0:  self.rect.top = 0
        if self.rect.bottom >= HEIGHT:  self.rect.bottom = HEIGHT

    def collisions(self):
        if self.rect.top <= 0 or self.rect.bottom >= HEIGHT:
            self.speed_y *= -1

        for paddle in self.paddles:
            if self.rect.colliderect(paddle):
                if abs(self.rect.right - paddle.rect.left) < 10 and self.speed_x > 0:
                    self.speed_x *= -1
                if abs(self.rect.left - paddle.rect.right) < 10 and self.speed_x < 0:
                    self.speed_x *= -1
                if abs(self.rect.top - paddle.rect.bottom) < 10 and self.speed_y < 0:
                    self.rect.top = paddle.rect.bottom
                    self.speed_y *= -1
                if abs(self.rect.bottom - paddle.rect.top) < 10 and self.speed_y > 0:
                    self.rect.bottom = paddle.rect.top
                    self.speed_y *= -1

    def reset_ball(self):
        self.active = False
        self.speed_x *= random.choice((-1,1))
        self.speed_y *= random.choice((-1,1))
        self.score_time = pygame.time.get_ticks()
        self.rect.center = (WIDTH/2, HEIGHT/2)

    def draw(self):
        pygame.draw.ellipse(screen, self.color, self.rect)

    def get_y(self):
        return self.rect.centery
    
    def get_x(self):
        return self.rect.centerx

    def get_vel_direction(self):
        if  (self.speed_x > 0 and self.speed_y > 0):  return 0
        elif(self.speed_x > 0 and self.speed_y < 0):  return 1
        elif(self.speed_x < 0 and self.speed_y > 0):  return 2
        else:   return 3

class Opponent(Block):
    def __init__(self, x_pos, y_pos, color, speed = 5):
        super().__init__(x_pos, y_pos, color, width=5, height=HEIGHT/9)
        self.speed = speed

    def move(self, ball):
        if self.rect.top < ball.rect.y:
            self.rect.y += self.speed
        if self.rect.bottom > ball.rect.y:
            self.rect.y -= self.speed
        self.screen_limit()

    def screen_limit(self):
        if self.rect.top <= 0:  self.rect.top = 0
        if self.rect.bottom >= HEIGHT:  self.rect.bottom = HEIGHT

    def draw(self):
        pygame.draw.rect(screen, self.color, self.rect)

class GameManager:
    def __init__(self, ball, player, opponent):
        self.player_score = 0
        self.opponent_score = 0
        self.ball = ball
        self.player = player
        self.opponent = opponent

    def run_game(self):
        # Drawing the game objects + Updating the game objects
        self.player.draw()  
        self.ball.draw()
        self.opponent.draw()

        self.player.move()
        self.ball.move()
        self.opponent.move(self.ball)

        reward = self.get_reward()
        self.draw_score()
        return reward

    def get_reward(self):
        if self.ball.rect.right >= WIDTH:
            self.opponent_score += 1
            self.ball.reset_ball()
            return -1
        if self.ball.rect.left <= 0:
            self.player_score += 1
            self.ball.reset_ball()
            return 1
        return 0

    def draw_score(self):
        player_score = basic_font.render(str(self.player_score),True,self.ball.color)
        opponent_score = basic_font.render(str(self.opponent_score),True,self.ball.color)

        player_score_rect = player_score.get_rect(midleft = (WIDTH / 2 + 30, HEIGHT/2))
        opponent_score_rect = opponent_score.get_rect(midright = (WIDTH / 2 - 30, HEIGHT/2))

        screen.blit(player_score,player_score_rect)
        screen.blit(opponent_score,opponent_score_rect)
    
    def is_done(self):
        if self.player_score >= 11 or self.opponent_score >= 11:
            return True
        else:
            return False

# General setup
pygame.mixer.pre_init(44100,-16,2,512)
pygame.init()
clock = pygame.time.Clock()

# Main Window
WIDTH = 480
HEIGHT = 360
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Pong')

# Global Variables
bg_color = (255, 255, 255)
light_grey = (200, 200, 200)
basic_font = pygame.font.Font('freesansbold.ttf', 20)
middle_strip = pygame.Rect(WIDTH/2-1, 0, 2, HEIGHT)

if __name__ == '__main__':
    # Game objects
    tmp_color = (randint(100,255),randint(100,255),randint(100,255))
    player = Player(WIDTH - 10, HEIGHT/2, tmp_color)
    opponent = Opponent(5, HEIGHT/2, tmp_color)
    paddles = [player, opponent]
    ball = Ball(WIDTH/2, HEIGHT/2, color = tmp_color, paddles = paddles)
    game_manager = GameManager(ball=ball, player=player, opponent=opponent)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    player.movement -= player.speed
                if event.key == pygame.K_DOWN:
                    player.movement += player.speed
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_UP:
                    player.movement += player.speed
                if event.key == pygame.K_DOWN:
                    player.movement -= player.speed
        
        # Background Stuff
        screen.fill(bg_color)
        pygame.draw.rect(screen,light_grey,middle_strip)
        
        # Run the game
        game_manager.run_game()

        # Rendering
        pygame.display.flip()
        clock.tick(60)