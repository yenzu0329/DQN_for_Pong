from main import *

def run_for_test(agent):
    # Game objects
    tmp_color = (randint(80,220),randint(80,220),randint(80,220))
    player = Player(WIDTH - 10, HEIGHT/2, tmp_color)
    opponent = Opponent(5, HEIGHT/2, tmp_color)
    paddles = [player, opponent]
    ball = Ball(WIDTH/2, HEIGHT/2, color = tmp_color, paddles = paddles)
    game_manager = GameManager(ball=ball, player=player, opponent=opponent)

    done = False
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        observation = [abs(player.get_x()-ball.get_x()), player.get_y(), ball.get_y(), ball.get_vel_direction()]
        action = agent.choose_action(observation=observation)
        if action == UP:            player.move_up()
        elif action == DOWN:        player.move_down()
        else:                       player.stop()
        
        # Background Stuff
        screen.fill(bg_color)
        pygame.draw.rect(screen,light_grey,middle_strip)
        
        # Run the game
        game_manager.run_game()
        done = game_manager.is_done()

        # Rendering
        pygame.display.flip()
        clock.tick(60)

if __name__ == '__main__':
    agent = Agent(gamma=0.99, epsilon=0.0, batch_size=64, n_actions=3, input_dims=[4], lr=0.001)

    local_dir = os.path.dirname(__file__)
    policy_net_path = os.path.join(local_dir, "result/policy_net_model.pth")
    target_net_path = os.path.join(local_dir, "result/target_net_model.pth")
    memory_path = os.path.join(local_dir, "result/memory.pickle")

    agent.Q_eval.load_state_dict(torch.load(policy_net_path))
    agent.Q_eval.eval()
    agent.Q_target.load_state_dict(torch.load(target_net_path))
    agent.Q_target.eval()
    with open(memory_path, 'rb') as f:
        memory = pickle.load(f)
    agent.load_memory(memory)
    run_for_test(agent)