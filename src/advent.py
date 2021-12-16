import argparse
import copy
import importlib
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple

import networkx as netx
import numpy as np


class AdventProblem(ABC):
    """
    An abstraction of an advent code problem.
    """

    def __init__(self, name: str):
        """
        :param name: a name useful for logging and debugging
        """

        self.name = name

    @abstractmethod
    def solve_part1(self) -> str:
        """
        Solve the advent problem part 1.
        """

    @abstractmethod
    def solve_part2(self) -> str:
        """
        Solve the advent problem part 2.
        """


class Day1(AdventProblem):

    def __init__(self, test: bool):

        super().__init__('do the submarine soundings increase?')
        soundings_file = 'day1_test.txt' if test else 'day1_soundings.txt'
        self.sounding_lines = open(soundings_file, 'r').readlines()

    def solve_part1(self) -> str:

        growing, last_sounding = 0, None

        for line in self.sounding_lines:
            sounding = int(line)
            if last_sounding is not None and last_sounding < sounding:
                growing += 1
            last_sounding = sounding

        return f'number growing: {growing}'

    def solve_part2(self) -> None:
        growing = 0
        last_sounding_1, last_sounding_2, last_sum3 = None, None, None

        for line in self.sounding_lines:
            sounding = int(line)
            if None not in (last_sounding_1, last_sounding_2):
                sum3 = sounding + last_sounding_1 + last_sounding_2
                if last_sum3 is not None and sum3 > last_sum3:
                    growing = growing + 1
                last_sum3 = sum3

            last_sounding_2, last_sounding_1 = last_sounding_1, sounding

        return f'number growing: {growing}'


class Day2(AdventProblem):

    def __init__(self, test: bool):

        super().__init__('find the submarine path')
        directions_file = 'day2_test.txt' if test else 'day2_directions.txt'
        self.direction_lines = open(directions_file, 'r').readlines()

    def solve_part1(self) -> None:
        horz, vert = 0, 0
        for line in self.direction_lines:
            tokens = line.split(' ')
            if tokens[0] == 'forward':
                horz = horz + float(tokens[1])
            elif tokens[0] == 'down':
                vert = vert + float(tokens[1])
            elif tokens[0] == 'up':
                vert = vert - float(tokens[1])
            else:
                raise Exception(f'Unknown direction {tokens[0]}')

        return f'vert * horz: {vert * horz}'

    def solve_part2(self) -> None:
        horz, aim, depth = 0, 0, 0
        for line in self.direction_lines:
            tokens = line.split(' ')
            if tokens[0] == 'forward':
                horz = horz + float(tokens[1])
                depth = depth + aim * float(tokens[1])
            elif tokens[0] == 'down':
                aim = aim + float(tokens[1])
            elif tokens[0] == 'up':
                aim = aim - float(tokens[1])
            else:
                raise Exception(f'Unknown direction {tokens[0]}')

        return f'depth * horz: {depth * horz}'


class Day3(AdventProblem):

    def __init__(self, test: bool):
        super().__init__('')
        bits_file = 'day3_test.txt' if test else 'day3_binary.txt'
        self.bits_lines = open(bits_file, 'r').readlines()
        self.nlines, self.nbits = len(self.bits_lines), len(self.bits_lines[0].strip())
        self.bits = np.zeros((self.nbits, self.nlines), dtype=int)
        for line_num, line in enumerate(self.bits_lines):
            for num in range(self.nbits):
                self.bits[num, line_num] = line[num].strip()

    def solve_part1(self) -> None:
        num0, num1 = np.sum(self.bits == 0, axis=1), np.sum(self.bits == 1, axis=1)
        gamma = np.array([0 if num0[i] > num1[i] else 1 for i in range(self.nbits)])
        epsilon = np.array([0 if num0[i] <= num1[i] else 1 for i in range(self.nbits)])

        gamma_num = gamma.dot(2 ** np.arange(gamma.size)[::-1])
        epsilon_num = epsilon.dot(2 ** np.arange(epsilon.size)[::-1])

        return f'gamma_num * epsilon_num: {gamma_num * epsilon_num}'

    def count_bins(self, i, check):
        return np.sum(self.bits[i, check] == 0), np.sum(self.bits[i, check] == 1)

    def solve_part2(self) -> None:
        o2_i, co2_i = np.ones(self.nlines, dtype=bool), np.ones(self.nlines, dtype=bool)

        for i in range(self.nbits):
            num0_o2, num1_o2 = self.count_bins(i, o2_i)
            num0_co2, num1_co2 = self.count_bins(i, co2_i)
            if sum(o2_i) > 1:
                o2_i[o2_i] = self.bits[i, o2_i] == (1 if num1_o2 >= num0_o2 else 0)
            if sum(co2_i) > 1:
                co2_i[co2_i] = self.bits[i, co2_i] == (0 if num0_co2 <= num1_co2 else 1)

        oxygen_rating = self.bits[:, o2_i].transpose()
        o2_rating_num = oxygen_rating.dot(2 ** np.arange(oxygen_rating.size)[::-1])
        co2_rating = self.bits[:, co2_i].transpose()
        co2_rating_num = co2_rating.dot(2 ** np.arange(co2_rating.size)[::-1])

        return f'o2_rating * co2_rating: {o2_rating_num * co2_rating_num}'


class BingoBoard:
    def __init__(self, board_lines: List[str]):
        self.board = np.zeros((5, 5), dtype=int)
        self.marks = np.zeros((5, 5), dtype=bool)
        self.already_won = False
        for i in range(5):
            line_tokens = board_lines[i].split()
            for j, line_token in enumerate(line_tokens):
                self.board[i, j] = int(line_token)

    def mark_number(self, number: int):
        if not self.already_won:
            self.marks[np.where(self.board == number)] = True

    def check_for_bingo(self):
        if self.already_won:
            return False
        winner = False
        for i in range(5):
            winner = winner | np.all(self.marks[i, :])
            winner = winner | np.all(self.marks[:, i])

        return winner

    def get_score(self, number: int):
        return np.sum(self.board[~self.marks]) * number

    def set_already_won(self):
        self.already_won = True

    def reset(self):
        self.marks[:] = False


class Day4(AdventProblem):

    def __init__(self, test: bool):
        super().__init__('Whale bingo')
        bingo_file = 'day4_test.txt' if test else 'day4_bingo.txt'
        self.bingo_lines = open(bingo_file, 'r').readlines()
        self.bingo = [int(x) for x in self.bingo_lines[0].split(',')]
        cursor, self.boards, board_lines = 2, [], [None] * 5

        while True:
            if cursor > len(self.bingo_lines):
                break
            for i in range(5):
                board_lines[i] = self.bingo_lines[cursor]
                cursor += 1

            self.boards.append(BingoBoard(board_lines))
            cursor += 1  # skip blank line

    def solve_part1(self) -> str:
        for number in self.bingo:
            for i, board in enumerate(self.boards):
                board.mark_number(number)
                if board.check_for_bingo():
                    return f'Board score: {board.get_score(number)}'

    def solve_part2(self) -> str:
        for board in self.boards:
            board.reset()

        for number in self.bingo:
            for board in self.boards:
                board.mark_number(number)
                if board.check_for_bingo():
                    last_board, last_number = board, number
                    board.already_won = True

        return f'Last board score: {last_board.get_score(last_number)}'


class Day5(AdventProblem):

    def __init__(self, test: bool):
        super().__init__('Hydrothermal vents')
        lines_file = 'day5_test.txt' if test else 'day5_lines.txt'
        lines = open(lines_file, 'r').readlines()
        nx, ny = 1000, 1000
        self.points1, self.points2 = np.zeros((nx, ny), dtype=int), np.zeros((nx, ny), dtype=int)
        for line in lines:
            start, stop = [token.split(',') for token in line.split('->')]
            start, stop = np.array([int(x) for x in start]), np.array([int(x) for x in stop])
            delta = stop - start
            if np.any(delta == 0):
                smin, smax = np.minimum(start, stop), np.maximum(start, stop)
                xs, xe = smin[0], smax[0] + 1
                ys, ye = smin[1], smax[1] + 1
                self.points1[xs:xe, ys:ye] += 1
            else:
                sign = np.sign(delta)
                assert np.abs(delta)[0] == np.abs(delta)[1]
                for i in range(np.abs(delta)[0] + 1):
                    pt = start + sign * i
                    self.points2[pt[0], pt[1]] += 1

    def solve_part1(self) -> str:
        return f'number of non-diagonal intersections: {np.count_nonzero(self.points1 > 1)}'

    def solve_part2(self) -> str:
        return f'number of all intersections: {np.count_nonzero(self.points1 + self.points2 > 1)}'


class Day6(AdventProblem):

    def __init__(self, test: bool):
        super().__init__('Lanternfish')
        counts_file = 'day6_test.txt' if test else 'day6_counts.txt'
        counts = open(counts_file, 'r').readlines()
        self.fishes = np.zeros(9, dtype=int)
        for age in counts[0].split(','):
            self.fishes[int(age)] += 1

    def get_number_at_day(self, days: int) -> int:
        fishes = copy.deepcopy(self.fishes)
        for day_num in range(days):
            day_zero_pop = fishes[0]
            fishes[:-1] = fishes[1:]
            fishes[8] = day_zero_pop
            fishes[6] += day_zero_pop

        return np.sum(fishes)

    def solve_part1(self) -> str:
        return f'{self.get_number_at_day(80)}'

    def solve_part2(self) -> str:
        return f'{self.get_number_at_day(256)}'


class Day7(AdventProblem):

    def __init__(self, test: bool):
        super().__init__('Position of crabs')
        pos_file = 'day7_test.txt' if test else 'day7_pos.txt'
        pos_lines = open(pos_file, 'r').readlines()
        pos_split = pos_lines[0].split(',')
        self.pos = np.zeros(len(pos_split), dtype=int)
        for i, pos in enumerate(pos_split):
            self.pos[i] = int(pos)

    def solve_part1(self) -> str:
        maxv = np.max(self.pos)
        min_fuel = np.inf
        for i in range(maxv):
            min_fuel = min(min_fuel, np.sum(np.abs(self.pos - i)))
        return f'{min_fuel}'

    def solve_part2(self) -> str:
        maxv = np.max(self.pos)
        min_fuel = np.inf
        for i in range(maxv):
            diff = np.abs(self.pos - i)
            min_fuel = min(min_fuel, np.round(np.sum(diff * (diff + 1) // 2)))
        return f'{min_fuel}'


class Day8(AdventProblem):

    def __init__(self, test: bool):
        super().__init__('Scrambled digits')
        input_file = 'day8_test.txt' if test else 'day8_input.txt'
        input_lines = open(input_file, 'r').readlines()
        self.in_digit_sets_list, self.out_digit_sets_list = [], []
        for input_line in input_lines:
            in_str, out_str = input_line.split('|')
            self.in_digit_sets_list.append([set(x) for x in in_str.strip().split()])
            self.out_digit_sets_list.append([set(x) for x in out_str.strip().split()])

        self.len_to_digit = dict({
            2: 1,
            3: 7,
            4: 4,
            7: 8
        })

    def solve_part1(self) -> str:
        num = 0
        for i in range(len(self.out_digit_sets_list)):
            for out_digit_set in self.out_digit_sets_list[i]:
                sl = len(out_digit_set)
                if sl in (2, 3, 4, 7):
                    num = num + 1
        return f'{num}'

    def solve_part2(self) -> str:
        num = 0

        for i in range(len(self.out_digit_sets_list)):
            digit_sets = {}

            for pass_num in range(3):
                # find 1, 4, 7, 8
                for in_digit_set in self.in_digit_sets_list[i]:
                    in_len = len(in_digit_set)
                    if in_len in self.len_to_digit:
                        digit_int = self.len_to_digit[in_len]
                        digit_sets[digit_int] = in_digit_set

                # find 2, 3, 5
                for in_digit_set in self.in_digit_sets_list[i]:
                    in_len = len(in_digit_set)
                    if in_len == 5:
                        if in_digit_set.intersection(digit_sets[1]) == digit_sets[1]:
                            digit_sets[3] = in_digit_set
                        elif len(in_digit_set.intersection(digit_sets[4])) == 3:
                            digit_sets[5] = in_digit_set
                        else:
                            digit_sets[2] = in_digit_set

                # to find 0, 6, 9
                for in_digit_set in self.in_digit_sets_list[i]:
                    in_len = len(in_digit_set)
                    if in_len == 6:
                        if in_digit_set.intersection(digit_sets[3]) == digit_sets[3]:
                            digit_sets[9] = in_digit_set
                        elif in_digit_set.intersection(digit_sets[1]) == digit_sets[1]:
                            digit_sets[0] = in_digit_set
                        else:
                            digit_sets[6] = in_digit_set

            # flip the keys and tokens
            str_digits = {}
            for digit_int, digit_set in digit_sets.items():
                str_digits[''.join(sorted(digit_set))] = digit_int

            out_num_str = ''
            for out_digit_str in self.out_digit_sets_list[i]:
                out_num_str += str(str_digits[''.join(sorted(out_digit_str))])

            num += int(out_num_str)

        return f'{num}'


class Day9(AdventProblem):

    def __init__(self, test: bool):
        super().__init__('Cave map')
        map_file = 'day9_test.txt' if test else 'day9_map.txt'
        map_lines = open(map_file, 'r').readlines()
        self.cave_map = None
        self.ny = len(map_lines)
        for i, map_line in enumerate(map_lines):
            if self.cave_map is None:
                self.nx = len(map_line.strip())
                self.cave_map = np.zeros((self.nx, self.ny), dtype=int)
            self.cave_map[:, i] = [x for x in list(map_line.strip())]

    def get_lower_comparison(self) -> np.ndarray:
        lower = np.ones((self.nx, self.ny), dtype=bool)
        lower[1:, :] = lower[1:, :] & (self.cave_map[1:, :] < self.cave_map[:-1, :])
        lower[:-1, :] = lower[:-1, :] & (self.cave_map[:-1, :] < self.cave_map[1:, :])
        lower[:, 1:] = lower[:, 1:] & (self.cave_map[:, 1:] < self.cave_map[:, :-1])
        lower[:, :-1] = lower[:, :-1] & (self.cave_map[:, :-1] < self.cave_map[:, 1:])

        return lower

    def solve_part1(self) -> str:
        return f'{np.sum(self.cave_map[self.get_lower_comparison()] + 1)}'

    def mark_basin(self, ind, x, y, basins):
        if x < 0 or x >= self.nx or y < 0 or y >= self.ny or basins[x, y] == -1 or basins[
            x, y] == ind:
            return

        basins[x, y] = ind
        self.mark_basin(ind, x + 1, y, basins)
        self.mark_basin(ind, x - 1, y, basins)
        self.mark_basin(ind, x, y + 1, basins)
        self.mark_basin(ind, x, y - 1, basins)

    def solve_part2(self) -> str:
        lower = self.get_lower_comparison()
        lowest_pts = self.cave_map[lower]
        lowest_inds = np.where(lower)
        basins = -2 * np.ones((self.nx, self.ny), dtype=int)
        basins[self.cave_map == 9] = -1
        nbasins = np.zeros(len(lowest_pts), dtype=int)

        for ind in range(len(lowest_pts)):
            self.mark_basin(ind, lowest_inds[0][ind], lowest_inds[1][ind], basins)
            nbasins[ind] = np.sum(basins == ind)

        num = np.prod(np.sort(nbasins)[-3:])
        return f'{num}'


class Day10(AdventProblem):

    def __init__(self, test: bool):
        super().__init__('Parser')
        code_file = 'day10_test.txt' if test else 'day10_code.txt'
        self.code_lines = open(code_file, 'r').readlines()
        self.openings = {
            '(': 3,
            '[': 57,
            '{': 1197,
            '<': 25137
        }
        self.closings = {
            ')': 3,
            ']': 57,
            '}': 1197,
            '>': 25137
        }
        self.incomplete_scores = {
            '(': 1,
            '[': 2,
            '{': 3,
            '<': 4
        }

    def solve_part1(self) -> str:
        stack, score = [], 0
        for code_line in self.code_lines:
            for token in list(code_line.strip()):
                if token in self.openings:
                    stack.append(token)
                elif token in self.closings:
                    open_token = stack.pop()
                    if self.closings[token] != self.openings[open_token]:
                        score += self.closings[token]
                        break
                else:
                    raise ValueError(f'Invalid token {token}')
        return f'{score}'

    def solve_part2(self) -> str:
        scores = []
        for code_line in self.code_lines:
            stack = []
            invalid_line = False
            for token in list(code_line.strip()):
                if token in self.openings:
                    stack.append(token)
                elif token in self.closings:
                    open_token = stack.pop()
                    if self.closings[token] != self.openings[open_token]:
                        invalid_line = True
                        break
            if not invalid_line:
                score = 0
                for bad_token in stack[::-1]:
                    score = 5 * score + self.incomplete_scores[bad_token]
                scores.append(score)

        scores_sorted = sorted(scores)
        return f'{scores_sorted[len(scores) // 2]}'


class Octopus:
    def __init__(self, energy: int):
        self.energy = energy
        self.neighbors: List[Octopus] = []
        self.flashed = False

    def check_for_flash(self):
        if self.flashed:
            return

        if self.energy > 9:
            # flash
            self.flashed = True
            for neighbor in self.neighbors:
                if not neighbor.flashed:
                    neighbor.energy += 1
                    neighbor.check_for_flash()

    def roll_over(self):
        if self.flashed:
            self.energy = 0
        self.flashed = False

    def __repr__(self):
        return f'{self.energy}'


class Day11(AdventProblem):
    def __init__(self, test: bool):
        super().__init__('Octopus flashes')
        code_file = 'day11_test.txt' if test else 'day11_energies.txt'
        self.energy_lines = open(code_file, 'r').readlines()
        self.nx, self.ny, self.nturns = 10, 10, 100
        self.octopuses, self.octopi_mat = [], np.zeros((self.nx, self.ny), dtype=int)
        self.stencil = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        self.octopus_map: Dict[Tuple[int, int], Octopus] = {}
        for i, energy_line in enumerate(self.energy_lines):
            for j, energy in enumerate(list(energy_line.strip())):
                self.octopi_mat[i, j] = int(energy)
                self.octopus_map[(i, j)] = Octopus(int(self.octopi_mat[i, j]))

        for ind, octopus in self.octopus_map.items():
            for x, y in self.stencil:
                ix, jx = ind[0] + x, ind[1] + y
                # if 0 <= i+x < self.nx & 0 <= j+y < self.ny:
                if (ix, jx) in self.octopus_map:
                    octopus.neighbors.append(self.octopus_map[(ix, jx)])

    def solve_part1(self) -> str:
        nflashed = 0

        for turn in range(self.nturns):
            for ind, octopus in self.octopus_map.items():
                octopus.flashed = False
                octopus.energy += 1

            for ind, octopus in self.octopus_map.items():
                octopus.check_for_flash()

            for octopus in self.octopus_map.values():
                if octopus.flashed:
                    nflashed += 1

            for ind, octopus in self.octopus_map.items():
                octopus.roll_over()

        return f'{nflashed}'

    def solve_part2(self) -> str:
        turn = 0

        while turn < 1000:  # turn in range(self.nturns):
            nflashed = 0
            for ind, octopus in self.octopus_map.items():
                octopus.flashed = False
                octopus.energy += 1

            for ind, octopus in self.octopus_map.items():
                octopus.check_for_flash()

            for octopus in self.octopus_map.values():
                if octopus.flashed:
                    nflashed += 1

            turn += 1

            if nflashed == self.nx * self.ny:
                break

            for ind, octopus in self.octopus_map.items():
                octopus.roll_over()

        return f'{turn}'


class Cave:
    def __init__(self, name: str):
        self.name = name
        self.big = name[0] == name[0].upper()
        self.connections = set()

    def walk_path(self, destination: 'Cave', current_path: List['Cave'],
                  all_paths: List[List['Cave']], max_small: int):

        for connection in self.connections:
            if connection == destination:
                current_path.append(destination)
                all_paths.append(current_path.copy())
                current_path.pop()
                continue

            can_visit = True
            if connection.big or connection not in current_path:
                temp_max_small = max_small
            elif max_small >= 1 and connection.name not in ('start', 'end'):
                temp_max_small = max_small - 1
            else:
                can_visit, temp_max_small = False, 0

            if can_visit:
                current_path.append(connection)
                connection.walk_path(destination, current_path, all_paths, temp_max_small)
                current_path.pop()

    def __repr__(self):
        return self.name


class Day12(AdventProblem):
    def __init__(self, test: bool):
        super().__init__('Cave graph')
        cave_file = 'day12_test.txt' if test else 'day12_graph.txt'
        self.cave_lines = open(cave_file, 'r').readlines()
        self.cave_maps = {}
        # load the cave names
        for cave_line in self.cave_lines:
            cave1_str, cave2_str = cave_line.split('-')
            cave1_name, cave2_name = cave1_str.strip(), cave2_str.strip()
            if cave1_name not in self.cave_maps:
                self.cave_maps[cave1_name] = Cave(cave1_name)
            if cave2_name not in self.cave_maps:
                self.cave_maps[cave2_name] = Cave(cave2_name)

        # load the cave connection
        for cave_line in self.cave_lines:
            cave1_str, cave2_str = cave_line.split('-')
            cave1, cave2 = self.cave_maps[cave1_str.strip()], self.cave_maps[cave2_str.strip()]
            cave1.connections |= {cave2}
            cave2.connections |= {cave1}

    def solve_part1(self) -> str:
        start_cave, end_cave = self.cave_maps['start'], self.cave_maps['end']
        all_paths, current_path = [], [start_cave]
        start_cave.walk_path(end_cave, current_path, all_paths, 0)
        return f'{len(all_paths)}'

    def solve_part2(self) -> str:
        start_cave, end_cave = self.cave_maps['start'], self.cave_maps['end']
        all_paths, current_path = [], [start_cave]
        start_cave.walk_path(end_cave, current_path, all_paths, 1)
        return f'{len(all_paths)}'


class Paper:
    def __init__(self, nx: int, ny: int):
        self.nx, self.ny = nx, ny
        self.dots = np.zeros((ny, nx), dtype=bool)

    def add_dot(self, x: int, y: int):
        self.dots[y, x] = True

    def __repr__(self):
        ret_val = ''
        dots = np.where(self.dots)
        max_x, max_y = np.max(dots[1]) + 1, np.max(dots[0]) + 1
        for j in range(max_y):
            for i in range(max_x):
                ret_val += '█' if self.dots[j, i] else ' '
            ret_val += '\n'

        return ret_val

    def fold(self, direction: str, location: int):
        assert direction in ('x', 'y')
        if direction == 'y':
            self.dots = self.dots[:location, :] | self.dots[2 * location:location:-1, :]
        else:  # if direction == 'x':
            self.dots = self.dots[:, :location] | self.dots[:, 2 * location:location:-1]


class Day13(AdventProblem):
    def __init__(self, test: bool):
        super().__init__('Origami code')
        origami_file = 'day13_test.txt' if test else 'day13_origami.txt'
        self.origami_lines = open(origami_file, 'r').readlines()
        dots_mode = True
        self.paper, self.folds = Paper(2000, 2000), []
        for origami_line in self.origami_lines:
            if len(origami_line.strip()) == 0:
                dots_mode = False
                continue

            if dots_mode:
                x, y = origami_line.strip().split(',')
                self.paper.add_dot(int(x), int(y))
            else:
                _, fold_str = origami_line.strip().split('fold along ')
                direction, location = fold_str.split('=')
                self.folds.append((direction, int(location)))

    def solve_part1(self) -> str:
        fold1 = self.folds[0]
        self.paper.fold(fold1[0], fold1[1])

        return f'{np.sum(self.paper.dots)}'

    def solve_part2(self) -> str:
        for fold in self.folds[1:]:
            self.paper.fold(fold[0], fold[1])

        return f'\n{self.paper}'


class Day14(AdventProblem):
    def __init__(self, test: bool):
        super().__init__('Polymerization')
        polymer_file = 'day14_test.txt' if test else 'day14_blah.txt'
        polymer_lines = open(polymer_file, 'r').readlines()
        template_mode, self.insertions = True, {}
        for polymer_line in polymer_lines:
            if len(polymer_line.strip()) == 0:
                template_mode = False
                continue

            if template_mode:
                self.template = polymer_line.strip()
            else:
                left, right = polymer_line.strip().split(' -> ')
                self.insertions[left] = right

    def solve_part1(self) -> str:
        cursor = list(self.template)
        for step in range(10):
            tokens, new_cursor = [], ''
            for token_num in range((len(cursor) - 1)):
                tokens.append(''.join(cursor[token_num:token_num + 2]))
            for token in tokens[::-1]:
                if token in self.insertions:
                    new_token = token[0] + self.insertions[token] + token[1]
                    new_cursor = new_token[1:] + new_cursor
                else:
                    new_cursor = token[1:] + new_cursor
            cursor = cursor[0] + new_cursor

        cursor_set = set(cursor)
        counts = np.zeros(len(cursor_set), dtype=int)

        for i, letter in enumerate(cursor_set):
            counts[i] = cursor.count(letter)

        counts = sorted(counts)

        return f'{counts[-1] - counts[0]}'

    @staticmethod
    def add_to_counts(add_where, add_what, count):
        if add_what not in add_where:
            add_where[add_what] = 0
        add_where[add_what] += count

    def solve_part2(self) -> str:
        cursor, letter_counts, token_counts = list(self.template), {}, {}
        for letter in ''.join(set(cursor)):
            letter_counts[letter] = self.template.count(letter)

        for token_num in range((len(self.template) - 1)):
            self.add_to_counts(token_counts, ''.join(cursor[token_num:token_num + 2]), 1)

        for step in range(40):
            token_counts_ = token_counts.copy()
            for token, token_count in token_counts.items():
                if token in self.insertions:
                    ins = self.insertions[token]
                    token_counts_[token] -= token_count
                    self.add_to_counts(token_counts_, token[0] + ins, token_count)
                    self.add_to_counts(token_counts_, ins + token[1], token_count)
                    self.add_to_counts(letter_counts, ins, token_count)
            token_counts = token_counts_

        counts = sorted(letter_counts.values())

        return f'{counts[-1] - counts[0]}'


class Day15(AdventProblem):
    def __init__(self, test: bool):
        super().__init__('Chiton path')
        graph_file = 'day15_test.txt' if test else 'day15_graph.txt'
        graph_lines = open(graph_file, 'r').readlines()
        self.graph = None
        self.stencil = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        for i, graph_line in enumerate(graph_lines):
            graph_line = graph_line.strip()
            if self.graph is None:
                self.nx = len(graph_line)
                self.graph = np.zeros((self.nx,self.nx), dtype=int)

            for j, graph_val in enumerate(graph_line):
                self.graph[i, j] = int(graph_val)

    def solve_part1(self) -> str:
        graph = netx.DiGraph()
        for i in range(self.nx):
            for j in range(self.nx):
                for stencil in self.stencil:
                    i_, j_ = i + stencil[0], j + stencil[1]
                    if 0 <= i_ < self.nx and 0 <= j_ < self.nx:
                        graph.add_edge((i,j), (i_,j_), weight=self.graph[i_, j_])

        dist = netx.shortest_path_length(graph, source=(0,0), target=(self.nx-1, self.nx-1),
                                         weight = 'weight')

        return f'{dist}'

    def solve_part2(self) -> str:
        ntile = 5
        nxp = ntile*self.nx
        graph_tiled = np.zeros((nxp, nxp), dtype=int)
        graph = netx.DiGraph()
        for xtile in range(ntile):
            for ytile in range(ntile):
                xs, ys = xtile * self.nx, ytile * self.nx
                xe, ye = (xtile+1)* self.nx, (ytile+1) * self.nx
                z_offset = xtile + ytile
                graph_tiled[xs:xe, ys:ye] = (self.graph + z_offset - 1) % 9 + 1

        for i in range(nxp):
            for j in range(nxp):
                for stencil in self.stencil:
                    i_, j_ = i + stencil[0], j + stencil[1]
                    if 0 <= i_ < nxp and 0 <= j_ < nxp:
                        graph.add_edge((i, j), (i_, j_), weight=graph_tiled[i_, j_])

        dist = netx.shortest_path_length(graph, source=(0,0), target=(nxp-1, nxp-1),
                                         weight = 'weight')

        return f'{dist}'


class BitsTree:
    def __init__(self):
        self.children: List['BitsTree'] = []
        self.literal: int = None
        self.version: int = None
        self.type: int = None

    def parse(self, bits: str) -> int:
        self.version, self.type = int(bits[0:3],2), int(bits[3:6],2)

        cursor = 6
        if self.type == 4: # literal string
            lit_str, has_more = '', True
            while has_more:
                next_five = bits[cursor:cursor+5]
                has_more = next_five[0] == '1'
                lit_str += next_five[1:]
                cursor = cursor + 5
            self.literal = int(lit_str,2)
        else: # operator
            length_type = int(bits[cursor:cursor+1], 2)
            cursor += 1
            if length_type == 0:
                num_bits = int(bits[cursor:cursor+15],2)
                cursor += 15
                tot_parsed = 0
                while tot_parsed != num_bits:
                    sub = BitsTree()
                    num_parsed = sub.parse(bits[cursor:])
                    tot_parsed += num_parsed
                    cursor += num_parsed
                    self.children.append(sub)
            else:
                num_packets, num_parsed = int(bits[cursor:cursor + 11], 2), 0
                cursor += 11
                while num_parsed != num_packets:
                    sub = BitsTree()
                    cursor += sub.parse(bits[cursor:])
                    self.children.append(sub)
                    num_parsed += 1

        return cursor

    def operate(self) -> int:
        if self.type == 4:
            retval = self.literal
        elif self.type == 0:
            retval = 0
            for child in self.children:
                retval += child.operate()
        elif self.type == 1:
            retval = 1
            for child in self.children:
                retval *= child.operate()
        elif self.type == 2:
            retval = np.inf
            for child in self.children:
                retval = min(retval, child.operate())
        elif self.type == 3:
            retval = -np.inf
            for child in self.children:
                retval = max(retval, child.operate())
        elif self.type == 5:
            assert len(self.children) == 2
            retval = 1 if self.children[0].operate() > self.children[1].operate() else 0
        elif self.type == 6:
            assert len(self.children) == 2
            retval = 1 if self.children[0].operate() < self.children[1].operate() else 0
        elif self.type == 7:
            assert len(self.children) == 2
            retval = 1 if self.children[0].operate() == self.children[1].operate() else 0
        else:
            raise ValueError(f'Unknown type {self.type}')

        return retval

    @property
    def version_sum(self) -> int:
        vsum = self.version
        for child in self.children:
            vsum += child.version_sum
        return vsum

class Day16(AdventProblem):
    def __init__(self, test: bool):
        super().__init__('BITS')
        bits_file = 'day16_test.txt' if test else 'day16_bits.txt'
        bits_hex = open(bits_file, 'r').readlines()[0].strip()
        bits_bin = ''.join([bin(int(x,16))[2:].zfill(4) for x in bits_hex])
        self.root = BitsTree()
        self.root.parse(bits_bin)

    def solve_part1(self) -> str:
        return f'{self.root.version_sum}'

    def solve_part2(self) -> str:
        return f'{self.root.operate()}'


def run_day(day: int, test: bool) -> None:
    module = importlib.import_module('advent')
    class_ = getattr(module, f'Day{day}')
    time1 = time.time()
    instance: AdventProblem = class_(test)
    time2 = time.time()
    print(f'Creating the class took {time2 - time1:.4f} seconds')
    print(f'Now solving Day {day} "{instance.name}":')
    part1, time3 = instance.solve_part1(), time.time()
    print(f'Part 1 ({time3 - time2:.4f} s) - {part1}')
    part2, time4 = instance.solve_part2(), time.time()
    print(f'Part 2 ({time4 - time3:.4f} s) - {part2}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("advent.py")

    parser.add_argument(dest='day', help='which day to run')
    parser.add_argument('-t', '--test', dest='test', help='run the output on the test data',
                        action="store_true")
    parser.set_defaults(test=False)

    args = parser.parse_args()

    run_day(args.day, args.test)
