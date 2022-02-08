"""
Example
"""

if __name__ == "__main__":
    import random
    random.seed(42)
    from jsonargparse import ArgumentParser, ActionConfigFile
    from atcenv import Environment
    import time
    from tqdm import tqdm

    parser = ArgumentParser(
        prog='Conflict resolution environment',
        description='Basic conflict resolution environment for training policies with reinforcement learning',
        print_config='--print_config',
        parser_mode='yaml'
    )
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--config', action=ActionConfigFile)
    parser.add_class_arguments(Environment, 'env')

    # parse arguments
    args = parser.parse_args()

    # init environment
    env = Environment(**vars(args))

    # run episodes
    for e in tqdm(range(args.episodes)):
        # reset environment
        obs = env.reset()

        # set done status to false
        done = False

        # execute one episode
        while not done:
            # perform step with dummy action
            obs, rew, done, info = env.step([])
            env.render()
            time.sleep(0.05)

        # close rendering
        env.close()
