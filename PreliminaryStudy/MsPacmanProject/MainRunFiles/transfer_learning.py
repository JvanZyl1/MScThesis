import gymnasium as gym
import torch

from MainRunFiles.Preprocessing import MsPacmanReducedActionSpaceWrapper, MsPacmanFullActionSpaceWrapper
from MainRunFiles.Preprocessing import PacmanWrapper
from MainRunFiles.Preprocessing import AlienVeryReducedActionSpaceWrapper, AlienReducedActionSpaceWrapper, AlienSemiReducedActionSpaceWrapper, AlienFullActionSpaceWrapper
from MainRunFiles.TrainingLoop import DQNTrainer
from MainRunFiles.QNetwork import PNNFlexibleQNetwork, BaseFeatureExtractor


def generate_config_name(configuration):
    # (eg) DQN or Double first
    string = ""
    if configuration["double_q_learning_bool"]:
        string += "Double"
    else:
        string += "DQN"

    if configuration["PER_bool"] and configuration["DuelingQNetwork"]:
        string += "_PER_Dueling"
    elif configuration["PER_bool"] and not configuration["DuelingQNetwork"]:
        string += "_PER"
    elif not configuration["PER_bool"] and configuration["DuelingQNetwork"]:
        string += "__Dueling"
    else:
        string += ""

    return string

def write_rewards(environment_name, config_name, rewards):
    with open(f"Results/{environment_name}/{config_name}/Rewards.txt", "w") as file:
        for reward in rewards:
            file.write(str(reward) + "\n")

def transfer_learning(DQN_params_Pacman,
                      DQN_params_MsPacmanReduced,
                      DQN_params_MsPacmanFull,
                      DQN_params_AlienVeryReduced,
                      DQN_params_AlienReduced,
                      DQN_params_AlienSemiReduced,
                      DQN_params_AlienFull,
                      configuration_all,
                      num_episodes = (1000, 500, 500, 500, 500, 500, 500),# (Pacman, MsPacmanReduced, MsPacmanFull, Alien)
                      render_frequency = (2, 2, 2, 2, 2, 2, 2),
                      model_save_interval = (2, 2, 2, 2, 2, 2, 2),
                      freeze_feature_extractor = (False, True, True, False, True, True, True),
                      pnn_bool = False,
                      plot_bool = False):
    
    num_tasks = len(render_frequency)
    print("Number of tasks: ", num_tasks)

    task_architectures = []
    for key in configuration_all.keys():
        if configuration_all[key]["DuelingQNetwork"]:
            task_architectures.append("dueling")
        else:
            task_architectures.append("normal")

    input_shape = (1, 80, 80)  # dep. pre-processing
    ###### Pacman ######
    env_Pacman = gym.make('ALE/Pacman-v5', full_action_space=False)
    wrapped_env_Pacman = PacmanWrapper(env_Pacman)
    num_actions_Pacman = wrapped_env_Pacman.action_space.n
    configuration_Pacman = configuration_all["Pacman"]
    config_name_Pacman = generate_config_name(configuration_Pacman)
    ###### Ms.Pacman : Reduced Action Space ######
    env_MsPacmanReduced = gym.make('ALE/MsPacman-v5', full_action_space=False)
    wrapped_env_MsPacmanReduced = MsPacmanReducedActionSpaceWrapper(env_MsPacmanReduced)
    num_actions_MsPacmanReduced = wrapped_env_MsPacmanReduced.action_space.n
    configuration_MsPacmanReduced = configuration_all["MsPacmanReduced"]
    config_name_MsPacmanReduced = generate_config_name(configuration_MsPacmanReduced)
    ###### Ms.Pacman : Full Action Space ######
    env_MsPacmanFull = gym.make('ALE/MsPacman-v5', full_action_space=False)
    wrapped_env_MsPacmanFull = MsPacmanFullActionSpaceWrapper(env_MsPacmanFull)
    num_actions_MsPacmanFull = wrapped_env_MsPacmanFull.action_space.n
    configuration_MsPacmanFull = configuration_all["MsPacmanFull"]
    config_name_MsPacmanFull = generate_config_name(configuration_MsPacmanFull)
    ###### Alien : Very Reduced Action Space ######
    env_AlienVeryReduced = gym.make('ALE/Alien-v5', full_action_space=False)
    wrapped_env_AlienVeryReduced = AlienVeryReducedActionSpaceWrapper(env_AlienVeryReduced)
    num_actions_AlienVeryReduced = wrapped_env_AlienVeryReduced.action_space.n
    configuration_AlienVeryReduced = configuration_all["AlienVeryReduced"]
    config_name_AlienVeryReduced = generate_config_name(configuration_AlienVeryReduced)
    ###### Alien : Reduced Action Space ######
    env_AlienReduced = gym.make('ALE/Alien-v5', full_action_space=False)
    wrapped_env_AlienReduced = AlienReducedActionSpaceWrapper(env_AlienReduced)
    num_actions_AlienReduced = wrapped_env_AlienReduced.action_space.n
    configuration_AlienReduced = configuration_all["AlienReduced"]
    config_name_AlienReduced = generate_config_name(configuration_AlienReduced)
    ###### Alien : Semi Action Space ######
    env_AlienSemiReduced = gym.make('ALE/Alien-v5', full_action_space=False)
    wrapped_env_AlienSemiReduced = AlienSemiReducedActionSpaceWrapper(env_AlienSemiReduced)
    num_actions_AlienSemiReduced = wrapped_env_AlienSemiReduced.action_space.n
    configuration_AlienSemiReduced = configuration_all["AlienSemiReduced"]
    config_name_AlienSemiReduced = generate_config_name(configuration_AlienSemiReduced)
    ###### Alien : Full Action Space ######
    env_AlienFull = gym.make('ALE/Alien-v5', full_action_space=True)
    wrapped_env_AlienFull = AlienFullActionSpaceWrapper(env_AlienFull)
    num_actions_AlienFull = wrapped_env_AlienFull.action_space.n
    configuration_AlienFull = configuration_all["AlienFull"]
    config_name_AlienFull = generate_config_name(configuration_AlienFull)

    flexible_pnn_network = PNNFlexibleQNetwork(
        feature_extractor= BaseFeatureExtractor(input_shape),
        num_tasks=num_tasks,
        task_architectures=task_architectures,
        task_num_actions=[num_actions_Pacman,
                  num_actions_MsPacmanReduced,
                  num_actions_MsPacmanFull,
                  num_actions_AlienVeryReduced,
                  num_actions_AlienReduced,
                  num_actions_AlienSemiReduced,
                  num_actions_AlienFull],
    )
    ###### Pacman ######
    print("Training Pacman :)")
    train_Pacman = DQNTrainer(input_shape,                                      # input shape
                        num_actions_Pacman,                                     # number of actions
                        DQN_params_Pacman,                                      # DQN parameters
                        wrapped_env_Pacman,                                     # environment
                        num_episodes[0],                                        # number of episodes
                        render_frequency[0],                                    # render frequency
                        file_path="Results",                                    # file path
                        environment_name = 'Pacman',                            # environment name
                        Wrapper= PacmanWrapper,                                 # wrapper
                        model_save_interval=model_save_interval[0],             # model save interval
                        configuration = configuration_Pacman,                   # configuration
                        config_name = config_name_Pacman,                       # configuration name
                        plot_bool=plot_bool,                                    # plot enabled
                        task_id= 1,                                             # task id
                        pnn_bool = pnn_bool,                                    # PNN enabled
                        num_tasks=num_tasks,                                            # number of tasks
                        flexible_pnn_network=flexible_pnn_network)              # flexible PNN network
    
    train_Pacman.training_loop()

    task_specific_layers_names_list_PACMAN = train_Pacman.task_specific_layers_names_list

    rewards_Pacman = train_Pacman.episode_rewards
    write_rewards('Pacman', config_name_Pacman, rewards_Pacman)

    path_to_Pacman_model = f'Results/Pacman/{config_name_Pacman}/Episode_{num_episodes[0]}.pt'
    
    ###### Ms.Pacman : Reduced Action Space ######    
    print("Training Ms.Pacman Reduced Action Space:)")
    train_MsPacmanReduced = DQNTrainer(input_shape,                             # input shape
                        num_actions_MsPacmanReduced,                            # number of actions
                        DQN_params_MsPacmanReduced,                             # DQN parameters
                        wrapped_env_MsPacmanReduced,                            # environment
                        num_episodes[1],                                        # number of episodes
                        render_frequency[1],                                    # render frequency
                        file_path="Results",                                    # file path
                        environment_name = 'MsPacman-Reduced',                  # environment name
                        Wrapper= MsPacmanReducedActionSpaceWrapper,             # wrapper
                        model_save_interval=model_save_interval[1],             # model save interval
                        configuration = configuration_MsPacmanReduced,          # configuration
                        config_name = config_name_MsPacmanReduced,              # configuration name
                        plot_bool=plot_bool,                                    # plot enabled
                        task_id= 2,                                             # task id
                        pnn_bool=pnn_bool,                                      # PNN enabled
                        num_tasks=num_tasks,                                            # number of tasks
                        task_specific_layers_names_list = task_specific_layers_names_list_PACMAN, # task specific layers names list
                        flexible_pnn_network=flexible_pnn_network)              # flexible PNN network

    train_MsPacmanReduced.load_pretrained_weights_transfer_learning(path_to_Pacman_model, freeze_feature_extractor=freeze_feature_extractor[1])

    train_MsPacmanReduced.training_loop()

    task_specific_layers_names_list_MSPACMANREDUCED = train_MsPacmanReduced.task_specific_layers_names_list

    rewards_MsPacmanReduced = train_MsPacmanReduced.episode_rewards
    write_rewards('MsPacman-Reduced', config_name_MsPacmanReduced, rewards_MsPacmanReduced)

    path_to_MsPacmanReduced_model = f'Results/MsPacman-Reduced/{config_name_MsPacmanReduced}/Episode_{num_episodes[1]}.pt'

    ###### Ms.Pacman : Full Action Space ######   
    print("Training Ms.Pacman Full Action Space:)")
    train_MsPacmanFull = DQNTrainer(input_shape,                                # input shape
                        num_actions_MsPacmanFull,                               # number of actions
                        DQN_params_MsPacmanFull,                                # DQN parameters
                        wrapped_env_MsPacmanFull,                               # environment
                        num_episodes[2],                                        # number of episodes
                        render_frequency[2],                                    # render frequency
                        file_path="Results",                                    # file path
                        environment_name = 'MsPacman-Full',                     # environment name
                        Wrapper= MsPacmanReducedActionSpaceWrapper,             # wrapper
                        model_save_interval=model_save_interval[2],             # model save interval
                        configuration = configuration_MsPacmanFull,             # configuration
                        config_name = config_name_MsPacmanFull,                 # configuration name
                        plot_bool=plot_bool,                                     # plot enabled
                        task_id= 3,                                              # task id
                        pnn_bool=pnn_bool,                                       # PNN enabled
                        num_tasks=num_tasks,                                             # number of tasks
                        task_specific_layers_names_list = task_specific_layers_names_list_MSPACMANREDUCED, # task specific layers names list
                        flexible_pnn_network=flexible_pnn_network)              # flexible PNN network

    train_MsPacmanFull.load_pretrained_weights_transfer_learning(path_to_MsPacmanReduced_model, freeze_feature_extractor=freeze_feature_extractor[2])

    train_MsPacmanFull.training_loop()

    task_specific_layers_names_list_MSPACMANFULL = train_MsPacmanFull.task_specific_layers_names_list

    rewards_MsPacmanFull = train_MsPacmanFull.episode_rewards
    write_rewards('MsPacman-Full', config_name_MsPacmanFull, rewards_MsPacmanFull)

    path_to_MsPacmanFull_model = f'Results/MsPacman-Full/{config_name_MsPacmanFull}/Episode_{num_episodes[2]}.pt'

    ###### Alien : Very Reduced Action Space ######   
    print("Training Alien Very Reduced Action Space:)")
    train_AlienVeryReduced = DQNTrainer(input_shape,                                # input shape
                        num_actions_AlienVeryReduced,                        # number of actions
                        DQN_params_AlienVeryReduced,                         # DQN parameters
                        wrapped_env_AlienVeryReduced,                        # environment
                        num_episodes[3],                                     # number of episodes
                        render_frequency[3],                                 # render frequency
                        file_path="Results",                                 # file path
                        environment_name='Alien-VeryReduced',                       # environment name
                        Wrapper=AlienVeryReducedActionSpaceWrapper,          # wrapper
                        model_save_interval=model_save_interval[3],          # model save interval
                        configuration=configuration_AlienVeryReduced,        # configuration
                        config_name=config_name_AlienVeryReduced,            # configuration name
                        plot_bool=plot_bool,                                 # plot enabled
                        task_id=4,                                           # task id
                        pnn_bool=pnn_bool,                                   # PNN enabled
                        num_tasks=num_tasks,                                         # number of tasks
                        task_specific_layers_names_list=task_specific_layers_names_list_MSPACMANFULL, # task specific layers names list
                        flexible_pnn_network=flexible_pnn_network)           # flexible PNN network

    train_AlienVeryReduced.load_pretrained_weights_transfer_learning(path_to_MsPacmanFull_model, freeze_feature_extractor=freeze_feature_extractor[3])

    train_AlienVeryReduced.training_loop()

    task_specific_layers_names_list_ALIENVERYREDUCED = train_AlienVeryReduced.task_specific_layers_names_list

    rewards_AlienVeryReduced = train_AlienVeryReduced.episode_rewards
    write_rewards('Alien-VeryReduced', config_name_AlienVeryReduced, rewards_AlienVeryReduced)
    path_to_AlienVeryReduced_model = f'Results/Alien-VeryReduced/{config_name_AlienVeryReduced}/Episode_{num_episodes[3]}.pt'

    ###### Alien : Reduced Action Space ######   
    print("Training Alien Reduced Action Space:)")
    train_AlienReduced = DQNTrainer(input_shape,                                # input shape
                        num_actions_AlienReduced,                        # number of actions
                        DQN_params_AlienReduced,                         # DQN parameters
                        wrapped_env_AlienReduced,                        # environment
                        num_episodes[4],                                     # number of episodes
                        render_frequency[4],                                 # render frequency
                        file_path="Results",                                 # file path
                        environment_name='Alien-Reduced',                       # environment name
                        Wrapper=AlienReducedActionSpaceWrapper,          # wrapper
                        model_save_interval=model_save_interval[4],          # model save interval
                        configuration=configuration_AlienReduced,        # configuration
                        config_name=config_name_AlienReduced,            # configuration name
                        plot_bool=plot_bool,                                 # plot enabled
                        task_id=5,                                           # task id
                        pnn_bool=pnn_bool,                                   # PNN enabled
                        num_tasks=num_tasks,                                         # number of tasks
                        task_specific_layers_names_list=task_specific_layers_names_list_ALIENVERYREDUCED, # task specific layers names list
                        flexible_pnn_network=flexible_pnn_network)           # flexible PNN network

    train_AlienReduced.load_pretrained_weights_transfer_learning(path_to_AlienVeryReduced_model, freeze_feature_extractor=freeze_feature_extractor[4])

    train_AlienReduced.training_loop()

    task_specific_layers_names_list_ALIENREDUCED = train_AlienReduced.task_specific_layers_names_list

    rewards_AlienReduced = train_AlienReduced.episode_rewards
    write_rewards('Alien-Reduced', config_name_AlienReduced, rewards_AlienReduced)

    path_to_AlienReduced_model = f'Results/Alien-Reduced/{config_name_AlienReduced}/Episode_{num_episodes[4]}.pt'

    ###### Alien : Semi Reduced Action Space ######   
    print("Training Alien Semi Reduced Action Space:)")
    train_AlienSemiReduced = DQNTrainer(input_shape,                                # input shape
                        num_actions_AlienSemiReduced,                        # number of actions
                        DQN_params_AlienSemiReduced,                         # DQN parameters
                        wrapped_env_AlienSemiReduced,                        # environment
                        num_episodes[5],                                     # number of episodes
                        render_frequency[5],                                 # render frequency
                        file_path="Results",                                 # file path
                        environment_name='Alien-SemiReduced',                       # environment name
                        Wrapper=AlienSemiReducedActionSpaceWrapper,          # wrapper
                        model_save_interval=model_save_interval[5],          # model save interval
                        configuration=configuration_AlienSemiReduced,        # configuration
                        config_name=config_name_AlienSemiReduced,            # configuration name
                        plot_bool=plot_bool,                                 # plot enabled
                        task_id=6,                                           # task id
                        pnn_bool=pnn_bool,                                   # PNN enabled
                        num_tasks=num_tasks,                                         # number of tasks
                        task_specific_layers_names_list=task_specific_layers_names_list_ALIENREDUCED, # task specific layers names list
                        flexible_pnn_network=flexible_pnn_network)           # flexible PNN network

    train_AlienSemiReduced.load_pretrained_weights_transfer_learning(path_to_AlienReduced_model, freeze_feature_extractor=freeze_feature_extractor[5])

    train_AlienSemiReduced.training_loop()

    task_specific_layers_names_list_ALIENSEMIREDUCED = train_AlienSemiReduced.task_specific_layers_names_list

    rewards_AlienSemiReduced = train_AlienSemiReduced.episode_rewards
    write_rewards('Alien-SemiReduced', config_name_AlienSemiReduced, rewards_AlienSemiReduced)

    path_to_AlienSemiReduced_model = f'Results/Alien-SemiReduced/{config_name_AlienSemiReduced}/Episode_{num_episodes[5]}.pt'

    ###### Alien : Full Action Space ######   
    print("Training Alien Full Action Space:)")
    train_AlienFull = DQNTrainer(input_shape,                                # input shape
                        num_actions_AlienFull,                        # number of actions
                        DQN_params_AlienFull,                         # DQN parameters
                        wrapped_env_AlienFull,                        # environment
                        num_episodes[6],                                     # number of episodes
                        render_frequency[6],                                 # render frequency
                        file_path="Results",                                 # file path
                        environment_name='Alien-Full',                       # environment name
                        Wrapper=AlienFullActionSpaceWrapper,          # wrapper
                        model_save_interval=model_save_interval[6],          # model save interval
                        configuration=configuration_AlienFull,        # configuration
                        config_name=config_name_AlienFull,            # configuration name
                        plot_bool=plot_bool,                                 # plot enabled
                        task_id=7,                                           # task id
                        pnn_bool=pnn_bool,                                   # PNN enabled
                        num_tasks=num_tasks,                                         # number of tasks
                        task_specific_layers_names_list=task_specific_layers_names_list_ALIENSEMIREDUCED, # task specific layers names list
                        flexible_pnn_network=flexible_pnn_network)           # flexible PNN network

    train_AlienFull.load_pretrained_weights_transfer_learning(path_to_AlienSemiReduced_model, freeze_feature_extractor=freeze_feature_extractor[6])

    train_AlienFull.training_loop()

    task_specific_layers_names_list_ALIENFULL = train_AlienFull.task_specific_layers_names_list

    rewards_AlienFull = train_AlienFull.episode_rewards
    write_rewards('Alien-Full', config_name_AlienFull, rewards_AlienFull)

    path_to_AlienFull_model = f'Results/Alien-Full/{config_name_AlienFull}/Episode_{num_episodes[6]}.pt'

    print("Finished training all tasks!")
    print('Task specific layers names list: ', task_specific_layers_names_list_ALIENFULL)