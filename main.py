import argparse
import glob
import os

from models import Parser
from models.initial_solution import InitialSolution

INPUT_INSTANCES_DIR = 'input'
OUTPUT_INSTANCES_DIR = 'output'

MINUTES_TO_RUN = 10


def main(version: str, initial_solution_method: str) -> None:
    output_sub_dir = os.path.join(OUTPUT_INSTANCES_DIR, version)
    os.makedirs(output_sub_dir, exist_ok=True)

    instance_paths = glob.glob(f'{INPUT_INSTANCES_DIR}/*.txt')

    for instance_path in instance_paths:
        instance = Parser(instance_path).parse()

        if initial_solution_method == 'generate_initial_solution_elite_threshold':
            initial_solution = InitialSolution.generate_initial_solution_elite_threshold(instance, 0.03)
        elif initial_solution_method == 'generate_initial_solution_ordered_list':
            initial_solution = InitialSolution.generate_initial_solution_ordered_list(instance)
        elif initial_solution_method == 'generate_initial_solution_static_greedy':
            initial_solution = InitialSolution.generate_initial_solution_static_greedy(instance)
        elif initial_solution_method == 'generate_initial_solution_weighted_efficiency':
            initial_solution = InitialSolution.generate_initial_solution_weighted_efficiency(instance)
        elif initial_solution_method == 'generate_initial_solution_adaptive_heap':
            initial_solution = InitialSolution.generate_initial_solution_adaptive_heap(instance)
        else:
            raise ValueError(f'Wrong initial solution method: {initial_solution_method}')

        score = initial_solution.fitness_score
        instance_name = os.path.basename(instance_path)
        print(instance_name, score)
        output_file = os.path.join(output_sub_dir, instance_name)
        initial_solution.export(output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--version', type=str, required=True)
    parser.add_argument('-i', '--initial_solution_method', type=str, required=True)

    args = parser.parse_args()
    main(args.version, args.initial_solution_method)
