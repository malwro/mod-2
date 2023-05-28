import yaml
import argparse

from model import ProblemModel
from ga import GeneticAlgorithm


def main():
    args = parse_args()

    with open(args.ga_config, "r") as file:
        try:
            ga_config = yaml.safe_load(file)
        except yaml.YAMLError as e:
            print(e)

    with open(args.data, "r") as file:
        try:
            data = yaml.safe_load(file)
        except yaml.YAMLError as e:
            print(e)

        p = ProblemModel(data)

        if args.use_genetic:
            a = GeneticAlgorithm(ga_config, data)
            a.run()

            solution, _ = a.get_solution()
            gen_d_bin_vars = a.regroup_solution(solution)

            a.first_condition(gen_d_bin_vars)
            print(f"gen_d_bin_vars: {gen_d_bin_vars}")

            p.mod_set_d_bin_vars(gen_d_bin_vars)
            print(
                f"Checked #: {a.checked_num}, Punished #: {a.punished_num}, Not meeting constraints #: {a.constr_not_met_num}, Cplexed #: {a.cplexed_num}"
            )

        p.mod_setup()
        p.mod_set_objective()

        p.mod_solve(log=True)
        print(p.mod_get_objective())

    exit(1)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ga_config",
        "-g",
        dest="ga_config",
        metavar="FILE",
        help="path to the GA config file",
        default="ga_config.yaml",
    )

    parser.add_argument(
        "--data",
        "-d",
        dest="data",
        metavar="FILE",
        help="path to data file",
        default="data.yaml",
    )

    parser.add_argument(
        "--use_genetic",
        "-gen",
        action="store_true",
        help="Pass flag to run genetic algorithm",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
