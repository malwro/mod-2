import pygad
import numpy as np
from model import ProblemModel

np.set_printoptions(linewidth=np.nan)


class GeneticAlgorithm:
    def __init__(self, ga_config, data) -> None:
        self.config = ga_config
        self.model_data = data
        self.O_num = 5
        self.fitness_func = self.get_fitness_function()
        self.ga_instance = pygad.GA(
            num_generations=self.config["num_generations"],
            num_parents_mating=self.config["num_parents_mating"],
            sol_per_pop=self.config["sol_per_pop"],
            num_genes=self.config["num_genes"],
            fitness_func=self.fitness_func,
            gene_space=[0, 1],
            mutation_num_genes=self.config["mutation_num_genes"],
            parent_selection_type="rank",
            gene_type=int,
            stop_criteria=self.config["stop_criteria"],
            keep_parents=self.config["num_parents_mating"],
            on_generation=lambda alg: print("Function: " + str(alg.best_solution()[1])),
        )

        self.punishment = self.config["punishment"]
        self.punished_num = 0
        self.constr_not_met_num = 0
        self.cplexed_num = 0
        self.checked_num = 0

    def run(self):
        self.ga_instance.run()
        print(f"Iterations #: {self.ga_instance.generations_completed}")

    def get_solution(self):
        solution, solution_fitness, solution_idx = self.ga_instance.best_solution()
        print(f"Solution: {solution}")
        print(f"Fitness: {solution_fitness}")
        return solution, solution_fitness

    def regroup_solution(self, solution):
        # Regroups solution list into list of lists corresponding to months
        list_of_lists = []
        list_of_lists = np.array(solution).reshape(-1, self.O_num)
        return list_of_lists

    def first_condition(self, new_solution):
        for m in new_solution:
            m[4] = min(1, m[0] + m[1])

    def second_condition(self, solution):
        punishment = 0

        for m in solution:
            s = np.sum(m)
            if s <= self.model_data["O_max"]:
                continue
            else:
                punishment += self.punishment * (s - self.model_data["O_max"])
        return punishment

    def get_fitness_function(self):
        def fitness_function(ga_instance, solution, solution_idx):
            punishment = 0
            self.checked_num += 1

            new_solution = self.regroup_solution(solution)
            self.first_condition(new_solution)
            punishment -= self.second_condition(new_solution)

            if punishment == 0:
                self.cplexed_num += 1
                cplex_model = ProblemModel(self.model_data)
                cplex_model.mod_set_d_bin_vars(new_solution)
                cplex_model.mod_set_variables()
                cplex_model.mod_set_constraints()
                cplex_model.mod_set_objective()
                cplex_model.mod_solve()
                try:
                    return cplex_model.mod_get_objective()
                except Exception as e:
                    self.constr_not_met_num += 1
                    return 0
            self.punished_num += 1
            return punishment

        return fitness_function
