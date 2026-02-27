from src.solver.solver03 import Solver03, Solver03Config
import warnings
warnings.filterwarnings(
    "ignore",
    message=".*copying from a non-meta parameter.*"
)

if __name__ == "__main__":
    solver_config = Solver03Config.load_config_from_file(
        "configs/solver/solver03.yaml")
    Solver03(solver_config).run()
