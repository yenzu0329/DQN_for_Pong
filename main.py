import pickle
import torch
from dqn import Agent
from pong import *
import matplotlib.pyplot as plt
import numpy as np
import os

def discount_rewards(reward, gamma = 0.99):
    reward = np.array(reward)
    discounted_r = np.zeros_like(reward, dtype=float)
    running_add = 0

    for t in reversed(range(0, reward.size)):
        if reward[t] != 0:
            running_add = 0 # if the game ended (in Pong), reset the reward sum
        running_add = running_add * gamma + reward[t] # the point here is to use Horner's method to compute those rewards efficiently
        discounted_r[t] = running_add

    discounted_r = (discounted_r - np.mean(discounted_r)) / np.std(discounted_r)
    return discounted_r

UP = 0
DOWN = 1
STOP = 2

if __name__ == '__main__':
    
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=3, input_dims=[4], lr=0.001)
    scores, eps_history = [], []
    avg_scores = []

    player = Player(WIDTH - 10, HEIGHT/2, light_grey)
    opponent = Opponent(5, HEIGHT/2, light_grey)
    paddles = [player, opponent]
    n_games = 10000

    for i in range(n_games):
        # Game objects
        tmp_color = (randint(80,220),randint(80,220),randint(80,220))
        player.color = tmp_color
        opponent.color = tmp_color
        ball = Ball(WIDTH/2, HEIGHT/2, color = tmp_color, paddles = paddles)
        game_manager = GameManager(ball=ball, player=player, opponent=opponent)
        
        done = False
        score = 0.0
        observation = [abs(player.get_x()-ball.get_x()), player.get_y(), ball.get_y(), ball.get_vel_direction()]

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            action = agent.choose_action(observation=observation)
            if action == UP:            player.move_up()
            elif action == DOWN:        player.move_down()
            else:                       player.stop()
            
            # Background Stuff
            screen.fill(bg_color)
            pygame.draw.rect(screen,light_grey,middle_strip)
            score_label = basic_font.render("Episode: "+str(i), True, light_grey)
            screen.blit(score_label, (10, 10))
            
            # Run the game
            reward = game_manager.run_game()
            if reward == 0:
                score += 0.01
            elif reward == 1:
                score += 3.0
            if reward == -1:
                score -= abs(ball.get_y() - player.get_y()) * 0.1
                done = True
            #done = game_manager.is_done()
            new_observation = [abs(player.get_x()-ball.get_x()), player.get_y(), ball.get_y(), ball.get_vel_direction()]
            agent.store_transition(observation, action, reward, new_observation, done)
            agent.learn()
            observation = new_observation

            # Rendering
            pygame.display.flip()
            clock.tick(500)

        scores.append(score)
        eps_history.append(agent.epsilon)
        if(len(scores) > 50):
            avg_score = 1.0 * np.mean(scores[-50:])
        else:
            avg_score = 1.0 * np.mean(scores)
        avg_scores.append(avg_score)

        if i % 100 == 0:
            agent.Q_target.load_state_dict(agent.Q_eval.state_dict())
        print('episode%4d' % i, '-- score %5.2f' % score, 'avg score %5.2f' % avg_score,
              'epsilon %.2f' % agent.epsilon)
        if score > 200:
            n_games = i + 1
            break

    # print and save model
    print(agent.Q_eval)
    print(agent.Q_target)

    local_dir = os.path.dirname(__file__)
    policy_net_path = os.path.join(local_dir, "result/policy_net_model.pth")
    target_net_path = os.path.join(local_dir, "result/target_net_model.pth")
    memory_path = os.path.join(local_dir, "result/memory.pickle")

    torch.save(agent.Q_eval.state_dict(), policy_net_path)        
    torch.save(agent.Q_target.state_dict(), target_net_path)
    with open(memory_path, 'wb') as f:
        pickle.dump(agent.make_memory(), f)

    # draw plot
    x = [i+1 for i in range(n_games)]
    filename = os.path.join(local_dir, 'dqn_for_pong.png')
    fig = plt.figure()
    plt.title("DQN for Pong")
    plt.plot(x, scores, '-', label = 'score')
    plt.plot(x, avg_scores, '-', label = 'avg_score')
    plt.legend()
    fig.savefig(filename)
    