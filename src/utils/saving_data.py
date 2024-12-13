import pickle
from datetime import datetime

def save_agent(agent, agent_name: str, base_dir: str = "data/agents", note: str = ""):
    '''
    Save the trained model to a file.

    params:
    base_dir: Base directory to save the MPO model [str]. Default: "data/agents".
    note: Optional note to add to the model save file [str].
    '''
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{agent_name}_agent_{timestamp}.pkl"
    filepath = f"{base_dir}/{filename}"

    # Create a copy of the agent without non-picklable attributes
    agent_copy = agent.__dict__.copy()
    non_picklable_keys = ['rng_key', 'replay_buffer']  # Add other non-picklable attributes if needed
    for key in non_picklable_keys:
        if key in agent_copy:
            del agent_copy[key]

    with open(filepath, 'wb') as f:
        pickle.dump({"mpo": agent_copy, "note": note}, f)

    print(f"{agent_name} model saved to {filepath} with note: {note}")
def save_losses(critic_losses: list,
                actor_losses: list,
                base_dir: str = "data/losses",
                note: str = ""):
    '''
    Save the critic and actor losses to a file.

    params:
    critic_losses: List of critic losses [list].
    actor_losses: List of actor losses [list].
    base_dir: Base directory to save the losses [str]. Default: "data/losses".
    note: Optional note to add to the losses save file [str].
    '''
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"losses_{timestamp}.pkl"
    filepath = f"{base_dir}/{filename}"

    with open(filepath, 'wb') as f:
        pickle.dump({"critic_losses": critic_losses, "actor_losses": actor_losses, "note": note}, f)

    print(f"Losses saved to {filepath} with note: {note}")
