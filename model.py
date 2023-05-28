from docplex.mp.model import Model
from docplex.mp.linear import Var
from docplex.mp.progress import TextProgressListener
from typing import List, Optional


class ProblemModel(Model):
    def __init__(self, data) -> None:
        super(ProblemModel, self).__init__()
        self.data = data
        self.O_num = 5
        self.gen_d_bin_vars: Optional[List[List[bool]]] = None
        # P_total - całkowity zysk ze sprzedaży gotowych mieszanek we wszystkich miesiącach
        self.P_total: Var
        # C_total - całkowity koszt zakupu surowych olejów wszystkich typów we wszystkich miesiącach
        self.C_total: Var
        # W_total - całkowity koszt magazynowania surowych olejów wszystkich typów we wszystkich miesiącach
        self.W_total: Var
        self.variable_types = {
            "x": "x_var_",
            "y": "y_var_",
            "s": "s_var_",
            "d": "d_bin_var_",
        }

    def get_variable_name(self, type, month, product) -> str:
        return (
            self.variable_types.get(type)
            + self._get_month(month)
            + "_"
            + self._get_prod_name(product)
        )

    def set_continous_var_matrix(
        self, type: str, monthId: int = 0, prodId: int = 1
    ) -> None:
        self.continuous_var_matrix(
            [i for i in range(self.data["M"])],
            [i for i in range(self.O_num)],
            name=lambda key: self.variable_types.get(type)
            + self._get_month(key[monthId])
            + "_"
            + self._get_prod_name(key[prodId]),
            lb=0.0,
        )

    # x_OM - ilość zakupionego oleju typu o w miesiącu m [T]
    def set_purcharsed_amount(self) -> None:
        self.set_continous_var_matrix("x")

    # y_OM - ilość oleju typu o wykorzystanego do produkcji w danym miesiącu m [T]
    def set_used_amount(self) -> None:
        self.set_continous_var_matrix("y")

    # d_OM - zmienna binarna określająca decyzję o wykorzystaniu oleju typu o do produkcji w danym miesiącu m [T]
    def set_used_decision(self) -> None:
        if self.gen_d_bin_vars is None:
            self.set_continous_var_matrix("d")

    # s_OM - ilość magazynowana oleju typu o na koniec miesiąca m [T]
    def set_stored_amount(self) -> None:
        self.continuous_var_matrix(
            [i for i in range(self.data["M"])],
            [i for i in range(self.O_num)],
            name=lambda key: "s_var_"
            + self._get_month(key[0])
            + "_"
            + self._get_prod_name(key[1]),
            lb=0.0,
            ub=self.data["So_max"],
        )

    def prepare_total_sum(self) -> None:
        self.P_total = self.continuous_var(name="P_total")
        self.C_total = self.continuous_var(name="C_total")
        self.W_total = self.continuous_var(name="W_total")

    def mod_set_variables(self):
        self.set_purcharsed_amount()
        self.set_used_amount()
        self.set_used_decision()
        self.set_stored_amount()
        self.prepare_total_sum()

    def st_total_sum(self) -> None:
        self.add_constraint(
            self.P_total
            == self.sum(self.find_matching_vars("y_var_", True)) * self.data["P_mix"]
        )
        self.add_constraint(
            self.W_total
            == self.sum(self.find_matching_vars("s_var_", True)) * self.data["C_store"]
        )
        self.add_constraint(
            self.C_total
            == self.sum(
                [
                    self.get_variable("x", m, o) * self.data["C"][m][o]
                    for m in range(self.data["M"])
                    for o in range(self.O_num)
                ]
            )
        )

    def st_max_amount_produced(self) -> None:
        # Maksymalna wielkość rafinacji oleju roślinnego/nieroślinnego w danym miesiącu
        for m in range(0, self.data["M"]):
            self.add_constraint(
                self.sum([self.get_variable("y", m, t) for t in [0, 1]])
                <= self.data["Yv_max"]
            )
            self.add_constraint(
                self.sum([self.get_variable("y", m, t) for t in [2, 3, 4]])
                <= self.data["Yo_max"]
            )

    def st_start_amount(self) -> None:
        # Początkowa ilość magazynowana (Jan)
        for o in range(self.O_num):
            self.add_constraint(
                self.get_variable("s", 0, o)
                == self.data["So_start"]
                - self.get_variable("y", 0, o)
                + self.get_variable("x", 0, o)
            )

    def st_final_stored_amount(self) -> None:
        # Ilość magazynowana w miesiącach Feb-Jun
        for m in range(1, self.data["M"]):
            for o in range(self.O_num):
                self.add_constraint(
                    self.get_variable("s", m, o)
                    == self.get_variable("s", m - 1, o)
                    - self.get_variable("y", m, o)
                    + self.get_variable("x", m, o)
                )

    def st_stored_amount(self) -> None:
        # Końcowa ilość magazynowana
        for o in range(self.O_num):
            self.add_constraint(
                self.get_variable("s", self.data["M"] - 1, o) == self.data["So_stop"]
            )

    def st_final_stored_amount_2(self) -> None:
        # Końcowa ilość magazynowana
        for o in range(self.O_num):
            self.add_constraint(
                self.get_variable("s", self.data["M"] - 1, o) == self.data["So_stop"]
            )

    def st_min_hardness(self) -> None:
        # minimalna twardość
        for m in range(0, self.data["M"]):
            self.add_constraint(
                self.sum(
                    [
                        self.get_variable("y", m, o)
                        * (self.data["H"][o] - self.data["H_min"])
                        for o in [*range(self.O_num)]
                    ]
                )
                >= 0.0
            )

    def st_max_hardness(self) -> None:
        # maksymalna twardość
        for m in range(0, self.data["M"]):
            self.add_constraint(
                self.sum(
                    [
                        self.get_variable("y", m, o)
                        * (self.data["H_max"] - self.data["H"][o])
                        for o in [*range(self.O_num)]
                    ]
                )
                >= 0.0
            )

    def st_binary_variables(self) -> None:
        if self.gen_d_bin_vars is None:
            # MIP problem
            # Zabezpieczenie podjętej decyzji binarnej
            for m in range(0, self.data["M"]):
                for o in range(self.O_num):
                    self.add_constraint(
                        self.get_variable("y", m, o)
                        <= self.data["A"] * self.get_variable("d", m, o)
                    )
            # Maksymalna ilość olejów wykorzystanych do produkcji
            for m in range(0, self.data["M"]):
                self.add_constraint(
                    self.sum(
                        [self.get_variable("d", m, o) for o in [*range(self.O_num)]]
                    )
                    <= self.data["O_max"]
                )
            # Minimalna wielkość zużycia oleju typu O w danym miesiący
            for m in range(0, self.data["M"]):
                for o in range(self.O_num):
                    self.add_constraint(
                        self.get_variable("y", m, o)
                        >= self.data["Y_min"] * self.get_variable("d", m, o)
                    )
            # Wymaganie dot. jednoczesnego użycia oleju typu VEG1/VEG2 oraz OIL3
            for m in range(0, self.data["M"]):
                self.add_constraint(
                    self.get_variable("d", m, 0) + self.get_variable("d", m, 1)
                    <= 2 * self.get_variable("d", m, 4)
                )

        else:
            # d_bin_var constraints introduced by Genetic algorithm
            for m in range(0, self.data["M"]):
                for o in range(self.O_num):
                    if self.gen_d_bin_vars[m, o]:
                        self.add_constraint(
                            self.get_variable("y", m, o) >= self.data["Y_min"]
                        )
                    else:
                        self.add_constraint(self.get_variable("y", m, o) == 0)

    def mod_set_constraints(self):
        self.st_total_sum()
        self.st_max_amount_produced()
        self.st_start_amount()
        self.st_final_stored_amount()
        self.st_stored_amount()
        self.st_final_stored_amount_2()
        self.st_min_hardness()
        self.st_max_hardness()
        self.st_binary_variables()

    def mod_set_d_bin_vars(self, vars: List[List[bool]]):
        self.gen_d_bin_vars = vars

    def mod_set_objective(self):
        self.set_objective(
            "max",
            self.P_total - self.C_total - self.W_total,
        )

    def mod_setup(self):
        self.mod_set_variables()
        self.mod_set_constraints()

    def mod_solve(self, log=False):
        if log:
            self.print_information()

        if self.gen_d_bin_vars is None:
            # progress listener only for MIP problem
            self.add_progress_listener(TextProgressListener())

        try:
            self.solve()
            if log:
                self.print_solution()
        except:
            print("Solution not found")

    def mod_get_objective(self):
        return self.objective_value

    def get_variable(self, type: str, month_id: int, prod_id: int):
        key = (
            self.variable_types.get(type, "nothing")
            + self._get_month(month_id)
            + "_"
            + self._get_prod_name(prod_id)
        )
        return self.get_var_by_name(key)

    @staticmethod
    def _get_month(n: int):
        return ["Jan", "Feb", "Mar", "Apr", "May", "Jun"][n]

    @staticmethod
    def _get_prod_name(n: int):
        return "VEG" + str(n + 1) if n < 2 else ("OIL" + str(n - 1))
