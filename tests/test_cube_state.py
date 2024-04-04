import rpd.cube_state as cube_state
import numpy as np

SOLVED_STATE_ARRAY = np.array([
    [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ],
    [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ],
    [
        [2, 2, 2],
        [2, 2, 2],
        [2, 2, 2],
    ],
    [
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
    ],
    [
        [4, 4, 4],
        [4, 4, 4],
        [4, 4, 4],
    ],
    [
        [5, 5, 5],
        [5, 5, 5],
        [5, 5, 5],
    ],
])

W_STATE_ARRAY = np.array([
    [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ],
    [
        [2, 2, 2],
        [1, 1, 1],
        [1, 1, 1],
    ],
    [
        [4, 4, 4],
        [2, 2, 2],
        [2, 2, 2],
    ],
    [
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
    ],
    [
        [5, 5 ,5],
        [4, 4, 4],
        [4, 4, 4],
    ],
    [
        [1, 1, 1],
        [5, 5, 5],
        [5, 5, 5],
    ],
])

def test_cube_to_solver_string():
    state = cube_state.CubeState(SOLVED_STATE_ARRAY)
    assert state.to_solver_string() == "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"

def test_cube_face_line_api():
    state = cube_state.CubeState(SOLVED_STATE_ARRAY)
    assert np.array_equal(state.get_face_line(0, 0, None), np.array([0, 0, 0]))


def test_cube_solved():
    state = cube_state.CubeState(SOLVED_STATE_ARRAY)
    assert state.is_solved() == True
    state = cube_state.CubeState(W_STATE_ARRAY)
    assert state.is_solved() == False

def test_cube_rotate_face():
    state = cube_state.CubeState(SOLVED_STATE_ARRAY).rotate_clockwise_once(cube_state.SquareColor.WHITE)
    expected = cube_state.CubeState(W_STATE_ARRAY)
    return np.array_equal(state, expected)