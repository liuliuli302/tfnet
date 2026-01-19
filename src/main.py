from src.solver.solver02 import Solver02, Solver02Config
import warnings
warnings.filterwarnings(
    "ignore",
    message=".*copying from a non-meta parameter.*"
)

if __name__ == "__main__":
    solver_config = Solver02Config.load_config_from_file(
        "configs/solver/solver02.yaml")
    Solver02(solver_config).run()
