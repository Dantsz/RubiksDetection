import numpy as np
from . import cube_state
from . import color
import kociemba

def solve(state: cube_state.CubeState) -> list[str]:
    """Solves the state of the cube using the Kociemba solver."""
    solve_str = kociemba.solve(state.to_solver_string())
    solve_steps = [ color.spatial_symbol_to_color(move).name[0] for move in solve_str.split(" ")]
    return solve_steps
