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
        [5, 1, 1],
        [5, 1, 1],
        [5, 1, 1],
    ],
    [
        [1, 2, 2],
        [1, 2, 2],
        [1, 2, 2],
    ],
    [
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
    ],
    [
        [2, 4 ,4],
        [2, 4, 4],
        [2, 4, 4],
    ],
    [
        [4, 5, 5],
        [4, 5, 5],
        [4, 5, 5],
    ],
])

R_STATE_ARRAY = np.array([
    [
        [0, 0, 0],
        [0, 0, 0],
        [2, 2, 2],
    ],
    [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ],
    [
        [2, 2, 2],
        [2, 2, 2],
        [3, 3, 3],
    ],
    [
        [3, 3, 3],
        [3, 3, 3],
        [5, 5, 5],
    ],
    [
        [4, 4, 4],
        [4, 4, 4],
        [4, 4, 4],
    ],
    [
        [5, 5, 5],
        [5, 5, 5],
        [0, 0, 0],
    ],
])

G_STATE_ARRAY = np.array([
    [
        [0, 0, 4],
        [0, 0, 4],
        [0, 0, 4],
    ],
    [
        [0, 0, 0],
        [1, 1, 1],
        [1, 1, 1],
    ],
    [
        [2, 2, 2],
        [2, 2, 2],
        [2, 2, 2],
    ],
    [
        [1, 3, 3],
        [1, 3, 3],
        [1, 3, 3],
    ],
    [
        [4, 4, 4],
        [4, 4, 4],
        [3, 3, 3],
    ],
    [
        [5, 5, 5],
        [5, 5, 5],
        [5, 5, 5],
    ],
])

Y_STATE_ARRAY = np.array([
    [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ],
    [
        [1, 1, 2],
        [1, 1, 2],
        [1, 1, 2],
    ],
    [
        [2, 2, 4],
        [2, 2, 4],
        [2, 2, 4],
    ],
    [
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
    ],
    [
        [4, 4, 5],
        [4, 4, 5],
        [4, 4, 5],
    ],
    [
        [5, 5, 1],
        [5, 5, 1],
        [5, 5, 1],
    ],
])

O_STATE_ARRAY = np.array([
    [
        [5, 5, 5],
        [0, 0, 0],
        [0, 0, 0],
    ],
    [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ],
    [
        [0, 0, 0],
        [2, 2, 2],
        [2, 2, 2],
    ],
    [
        [2, 2, 2],
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
        [3, 3, 3],
    ],
])

B_STATE_ARRAY = np.array([
    [
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
    ],
    [
        [1, 1, 1],
        [1, 1, 1],
        [3, 3, 3],
    ],
    [
        [2, 2, 2],
        [2, 2, 2],
        [2, 2, 2],
    ],
    [
        [3, 3, 4],
        [3, 3, 4],
        [3, 3, 4],
    ],
    [
        [0, 0, 0],
        [4, 4, 4],
        [4, 4, 4],
    ],
    [
        [5, 5, 5],
        [5, 5, 5],
        [5, 5, 5],
    ],
])

def test_cube_to_solver_string():
    state = cube_state.CubeState(SOLVED_STATE_ARRAY)
    assert state.to_solver_string() == "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"

def test_cube_face_line_api():
    state = cube_state.CubeState(W_STATE_ARRAY)
    assert np.array_equal(state.get_face_line(0, 0, None), np.array([0, 0, 0]))
    assert np.array_equal(state.get_face_line(0, None, 0), np.array([0, 0, 0]))
    assert np.array_equal(state.get_face_line(1, 0, None), np.array([5, 5, 5]))
    assert np.array_equal(state.get_face_line(1, None, 0), np.array([5, 1, 1]))

def test_cube_solved():
    state = cube_state.CubeState(SOLVED_STATE_ARRAY)
    assert state.is_solved() == True
    state = cube_state.CubeState(W_STATE_ARRAY)
    assert state.is_solved() == False

def test_cube_rotate_face_W():
    state = cube_state.CubeState(SOLVED_STATE_ARRAY).rotate_clockwise_once(cube_state.SquareColor.WHITE)
    expected = cube_state.CubeState(W_STATE_ARRAY)
    assert np.array_equal(state.state, expected.state), "Expected: \n{}\nGot: \n{}".format(expected.state, state.state)

def test_cube_rotate_face_R():
    state = cube_state.CubeState(SOLVED_STATE_ARRAY).rotate_clockwise_once(cube_state.SquareColor.RED)
    expected = cube_state.CubeState(R_STATE_ARRAY)
    assert np.array_equal(state.state, expected.state), "Expected: \n{}\nGot: \n{}".format(expected.state, state.state)

def test_cube_rotate_face_G():
    state = cube_state.CubeState(SOLVED_STATE_ARRAY).rotate_clockwise_once(cube_state.SquareColor.GREEN)
    expected = cube_state.CubeState(G_STATE_ARRAY)
    assert np.array_equal(state.state, expected.state), "Expected: \n{}\nGot: \n{}".format(expected.state, state.state)


def test_cube_rotate_face_Y():
    state = cube_state.CubeState(SOLVED_STATE_ARRAY).rotate_clockwise_once(cube_state.SquareColor.YELLOW)
    expected = cube_state.CubeState(Y_STATE_ARRAY)
    assert np.array_equal(state.state, expected.state), "Expected: \n{}\nGot: \n{}".format(expected.state, state.state)

def test_cube_rotate_face_O():
    state = cube_state.CubeState(SOLVED_STATE_ARRAY).rotate_clockwise_once(cube_state.SquareColor.ORANGE)
    expected = cube_state.CubeState(O_STATE_ARRAY)
    assert np.array_equal(state.state, expected.state), "Expected: \n{}\nGot: \n{}".format(expected.state, state.state)

def test_cube_rotate_face_B():
    state = cube_state.CubeState(SOLVED_STATE_ARRAY).rotate_clockwise_once(cube_state.SquareColor.BLUE)
    expected = cube_state.CubeState(B_STATE_ARRAY)
    assert np.array_equal(state.state, expected.state), "Expected: \n{}\nGot: \n{}".format(expected.state, state.state)
