from pprint import pprint

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.connectors.env_to_module import FlattenObservations

# Configure the algorithm.
config = (
    PPOConfig()
    .environment("Taxi-v3")
    .env_runners(
        num_env_runners=2,
        # Observations are discrete (ints) -> We need to flatten (one-hot) them.
        env_to_module_connector=lambda env: FlattenObservations(),
    )
    .evaluation(evaluation_num_env_runners=1)
)


def main():
    # Build the algorithm.
    algo = config.build_algo()

    # Train it for 5 iterations ...
    for _ in range(5):
        pprint(algo.train())
        print("-" * 100)

    pprint(algo.evaluate())

    # Release the algo's resources (remote actors, like EnvRunners and Learners).
    algo.stop()


if __name__ == "__main__":
    main()
