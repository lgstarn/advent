import argparse
import copy
import importlib
import itertools
import re
import time
from abc import ABC, abstractmethod
from collections import namedtuple, Counter
from math import floor, ceil, inf
from typing import List, Dict, Tuple, Optional, Union, Set, Any
import heapq

import networkx as netx
import numpy as np
from scipy.signal import convolve2d
from scipy.spatial import KDTree


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

        last_board, last_number = None, None

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
    def __init__(self, depth: int = 0):
        self.children: List['BitsTree'] = []
        self.literal: int = None,
        self.version: int = None
        self.type: int = None
        self.depth = depth

    def parse(self, bits: str) -> Tuple[int, int]:
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
                    sub = BitsTree(self.depth+1)
                    num_parsed = sub.parse(bits[cursor:])
                    tot_parsed += num_parsed
                    cursor += num_parsed
                    self.children.append(sub)
            else:
                num_packets, num_parsed = int(bits[cursor:cursor + 11], 2), 0
                cursor += 11
                while num_parsed != num_packets:
                    sub = BitsTree(self.depth+1)
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

    @property
    def max_depth(self) -> int:
        vmax = self.depth
        for child in self.children:
            vmax = max(vmax, child.max_depth)
        return vmax


class Day16(AdventProblem):
    def __init__(self, test: bool):
        super().__init__('BITS')
        bits_file = 'day16_test.txt' if test else 'day16_bits.txt'
        bits_hex = open(bits_file, 'r').readlines()[0].strip()
        bits_bin = ''.join([bin(int(x,16))[2:].zfill(4) for x in bits_hex])
        self.root = BitsTree()
        self.root.parse(bits_bin)

    def solve_part1(self) -> str:
        return f'{self.root.version_sum} (max depth: {self.root.max_depth})'

    def solve_part2(self) -> str:
        return f'{self.root.operate()}'


class Day17(AdventProblem):
    def __init__(self, test: bool):
        super().__init__('Trick shot')
        target_file = 'day17_test.txt' if test else 'day17_target.txt'
        target_area = open(target_file, 'r').readlines()[0].strip()
        _, xy_range = target_area.split('target area: x=')
        x_range, y_range = xy_range.split(', y=')
        x_lower, x_upper = x_range.split('..')
        y_lower, y_upper = y_range.split('..')
        self.x_range = (int(x_lower), int(x_upper))
        self.y_range = (int(y_lower), int(y_upper))

    def valid_vxs(self) -> Tuple[int, int]:
        lower, upper = self.x_range
        vx, min_vx = 0, -np.inf
        for vx in range(lower + 1):
            x = vx * (vx + 1) / 2
            if lower <= x <= upper:
                min_vx = vx
                break

        max_vx = upper + 1  # guaranteed to be too fast
        return min_vx, max_vx

    def valid_vys(self):
        lower, upper = self.y_range
        return lower, abs(lower)

    def ends_in_target(self, vx: int, vy: int) -> Tuple[bool, int]:
        impossibru = False
        x, y = 0, 0
        xstart, xend = self.x_range
        ystart, yend = self.y_range
        max_y = -np.inf
        while not impossibru:
            x += vx
            y += vy

            max_y = max(max_y, y)
            if xstart <= x <= xend and ystart <= y <= yend:
                return True, max_y

            vx = max(vx - 1, 0)
            vy -= 1

            if x > xend or y < ystart:
                impossibru = True
        return False, max_y

    def solve_part1(self) -> str:
        min_vx, max_vx = self.valid_vxs()
        min_vy, max_vy = self.valid_vys()
        max_y = -np.inf
        for vy in range(min_vy, max_vy):
            for vx in range(min_vx, max_vx):
                valid, max_y_ = self.ends_in_target(vx, vy)
                if valid:
                    if max_y_ > max_y:
                        max_y = max_y_

        return f'{max_y}'

    def solve_part2(self) -> str:
        min_vx, max_vx = self.valid_vxs()
        min_vy, max_vy = self.valid_vys()
        valid_vxvys = 0
        for vy in range(min_vy, max_vy):
            for vx in range(min_vx, max_vx):
                valid, _ = self.ends_in_target(vx, vy)
                if valid:
                    valid_vxvys += 1

        return f'{valid_vxvys}'


class SnailfishTree:
    def __init__(self, val: Optional[int] = None, depth: int = 0):
        self.left, self.right, self.parent = None, None, None
        self.val, self.depth = val, depth

    def parse(self, instr: Union[List, int], depth: int = 0,
              parent: Optional['SnailfishTree'] = None) -> None:
        self.depth = depth
        self.parent = parent
        if isinstance(instr, List):
            self.left = SnailfishTree()
            self.left.parse(instr[0], depth + 1, self)
            self.right = SnailfishTree()
            self.right.parse(instr[1], depth + 1, self)
        else:
            self.val = instr

    def increment_depth(self) -> None:
        self.depth += 1
        if self.left is not None:
            self.left.increment_depth()
        if self.right is not None:
            self.right.increment_depth()

    def flatten_to_list(self, to_add: List['SnailfishTree']):
        if self.val is not None:
            to_add.append(self)
        else:
            self.left.flatten_to_list(to_add)
            self.right.flatten_to_list(to_add)

    def get_right(self, list_of_nodes: List['SnailfishTree']) -> 'SnailfishTree':
        ind = list_of_nodes.index(self)
        return list_of_nodes[ind + 1] if ind + 1 < len(list_of_nodes) else None

    def get_left(self, list_of_nodes: List['SnailfishTree']) -> 'SnailfishTree':
        ind = list_of_nodes.index(self)
        return list_of_nodes[ind - 1] if ind > 0 else None

    def needs_explosion(self) -> bool:
        return self.depth > 4

    def needs_split(self) -> bool:
        return self.val is not None and self.val >= 10

    def explode(self, list_of_nodes: List['SnailfishTree']) -> None:
        l, r = self.left.val, self.right.val

        right_neighbor = self.right.get_right(list_of_nodes)
        if right_neighbor is not None:
            right_neighbor.val += r

        left_neighbor = self.left.get_left(list_of_nodes)
        if left_neighbor is not None:
            left_neighbor.val += l

        self.val, self.left, self.right = 0, None, None

    def split(self):
        assert self.val is not None
        assert self.right is None
        assert self.left is None
        cur_val = self.val
        self.val = None
        self.left = SnailfishTree(floor(cur_val / 2))
        self.right = SnailfishTree(ceil(cur_val / 2))
        self.left.parent = self
        self.right.parent = self
        self.left.depth = self.depth + 1
        self.right.depth = self.depth + 1
        # fin

    @property
    def magnitude(self) -> int:
        # The magnitude of a pair is 3 times the magnitude of its left element plus
        # 2 times the magnitude of its right element. The magnitude of a regular
        # number is just that number.
        if self.val is not None:
            return self.val
        else:
            return 3 * self.left.magnitude + 2 * self.right.magnitude

    def __repr__(self) -> str:
        if self.val is not None:
            return f'{self.val}'
        else:
            return f'[{self.left},{self.right}]'

    def __add__(self, other: 'SnailfishTree') -> 'SnailfishTree':
        assert self.depth == other.depth == 0
        parent = SnailfishTree()
        parent.depth = self.depth
        parent.left = self
        parent.right = other
        parent.left.increment_depth()
        parent.right.increment_depth()
        self.parent = other.parent = parent
        return parent

    def to_list(self) -> List[int]:
        if self.parent is None:
            return self.left.to_list() + self.right.to_list()
        elif self.val is None:
            return [self.left.to_list() + self.right.to_list()]
        else:
            return [self.val]

    def get_max_depth(self, maxd: int) -> int:
        if self.val is not None:
            return max(maxd, self.depth)
        else:
            return max(self.left.get_max_depth(maxd), self.right.get_max_depth(maxd))

    def reduce(self) -> bool:
        flattened_list = []
        self.flatten_to_list(flattened_list)
        # max_depth = self.get_max_depth(-np.inf)
        for leaf in flattened_list:
            if leaf.depth > 4:
                leaf.parent.explode(flattened_list)
                return True

        for leaf in flattened_list:
            if leaf.val >= 10:
                leaf.split()
                return True

        return False


class Day18(AdventProblem):
    def __init__(self, test: bool):
        super().__init__('Snailfish tree/list')
        data_file = 'day18_test.txt' if test else 'day18_trees.txt'
        data_lines = open(data_file, 'r').readlines()
        self.root_trees = []
        for data_line in data_lines:
            arr = eval(data_line.strip())
            root_tree = SnailfishTree()
            root_tree.parse(arr)
            self.root_trees.append(root_tree)

    def solve_part1(self) -> str:
        end_tree = copy.deepcopy(self.root_trees[0])
        for root_tree in self.root_trees[1:]:
            while end_tree.reduce():
                pass
            while root_tree.reduce():
                pass
            end_tree = end_tree + copy.deepcopy(root_tree)

        while end_tree.reduce():
            pass

        return f'{end_tree.magnitude}'

    def solve_part2(self) -> str:
        max_mag = -np.inf
        for i in range(len(self.root_trees)):
            for j in range(len(self.root_trees)):
                if i != j:
                    tree1 = copy.deepcopy(self.root_trees[i])
                    tree2 = copy.deepcopy(self.root_trees[j])
                    out = tree1 + tree2
                    while out.reduce():
                        pass

                    max_mag = max(max_mag, out.magnitude)

        return f'{max_mag}'


class ScannerRotation:
    def __init__(self, x_flip: bool, y_flip: bool, z_flip: bool, permutation: Tuple[int,int,int]):
        self.x_sign = -1 if x_flip else 1
        self.y_sign = -1 if y_flip else 1
        self.z_sign = -1 if z_flip else 1
        self.permutation = permutation

    def rotate(self, points: np.ndarray) -> np.ndarray:
        # permute the columns
        new_array = points[:, self.permutation]
        # flip the signs
        new_array[:,0] *= self.x_sign
        new_array[:,1] *= self.y_sign
        new_array[:,2] *= self.z_sign
        return new_array

    def compose(self, other: 'ScannerRotation') -> 'ScannerRotation':
        x_flip = self.x_sign * other.x_sign == -1
        y_flip = self.y_sign * other.y_sign == -1
        z_flip = self.z_sign * other.z_sign == -1
        other_permutation = list(other.permutation)
        permutation = tuple(other_permutation[i] for i in list(self.permutation))
        return ScannerRotation(x_flip, y_flip, z_flip, permutation)


class Scanner:
    def __init__(self, scanner_number: int, list_of_points: List[List[int]]):
        self.scanner_number = scanner_number
        self.beacons = np.array(list_of_points)
        # self.dist_mat = distance_matrix(self.beacons, self.beacons)
        self.offset = np.array([0,0,0])

    def check_for_consistent_offset(self, other: 'Scanner', rotation: np.ndarray) -> Tuple[bool, np.ndarray]:
        rot_beacons = (rotation @ other.beacons.T).T
        offset_map = {}

        for i in range(self.beacons.shape[0]):
            offsets = rot_beacons - self.beacons[i, :]
            for offset_num in range(offsets.shape[0]):
                offset = tuple(offsets[offset_num, :])
                if offset not in offset_map:
                    offset_map[offset] = 0
                offset_map[offset] += 1

        # get out the counts, see if any are more than 12
        list_vals = list(offset_map.values())
        list_inds = list(offset_map)
        correct_inds = np.where(np.array(list_vals) >= 12)
        if len(correct_inds) > 1:
            raise Exception('double inds!')
        correct_ind = correct_inds[0]
        found_offset = len(correct_ind) > 0
        correct_offset = list_inds[correct_ind[0]] if found_offset else None
        if found_offset:
            other.offset = np.array(correct_offset)
        return found_offset, correct_offset

    def add_new_points(self, other: 'Scanner', offset: np.ndarray, rotation: np.ndarray):
        rot_and_trans = (rotation @ other.beacons.T).T - offset
        to_include = []
        for ii,point in enumerate(rot_and_trans):
            contains = list(point) in self.beacons.tolist()
            to_include.append(not contains)
        self.beacons = np.append(self.beacons, rot_and_trans[to_include,:], axis=0)

    def __repr__(self):
        ret_val = f'--- scanner {self.scanner_number} ---\n'
        for i in range(self.beacons.shape[0]):
            for j in range(self.beacons.shape[1]):
                ret_val += (',' if j > 0 else '') + str(self.beacons[i,j])
            ret_val += '\n'
        return ret_val


class Day19(AdventProblem):
    def __init__(self, test: bool):
        super().__init__('Scanners and beacons')
        sensor_file = 'day19_test.txt' if test else 'day19_sensors.txt'
        sensor_lines = open(sensor_file, 'r').readlines()
        self.scanners = []
        scanner_beacons: List[List[int]] = []
        ii_scan = -1
        for sensor_line in sensor_lines:
            if '---' in sensor_line:
                ii_scan += 1
                scanner_beacons.append([])
            elif len(sensor_line.strip()) != 0:
                scanner_beacons[ii_scan].append([int(x) for x in sensor_line.split(",")])
        for scanner in scanner_beacons:
            self.scanners.append(Scanner(len(self.scanners), scanner))
        self.to_check = None

    def largest_distance(self):
        maxdist = -np.inf
        for ii,x in enumerate(self.scanners):
            for y in self.scanners[ii:]:
                maxdist = max(np.linalg.norm(x.offset-y.offset,ord=1),maxdist)
        return maxdist

    @classmethod
    def run_icp(cls, target: Scanner, other: Scanner) -> Tuple[bool, np.ndarray, np.ndarray]:
        steps, converged = 0, False
        target_kdt = KDTree(target.beacons)
        offset, rotation = np.array([0, 0, 0]), ScannerRotation(False, False, False, [0,1,2])
        rot_trans_beacons = other.beacons
        while not converged and steps < 5:
            _, inds = target_kdt.query(rot_trans_beacons, p=1)
            # centroid = np.round(np.mean(diff, axis=0))
            for x_flip, y_flip, z_flip in itertools.product([False, True], repeat=3):
                for permutation in itertools.permutations(range(3)):
                    new_rotation = ScannerRotation(x_flip, y_flip, z_flip, permutation)
                    # new_beacons = new_rotation.rotate(rot_trans_beacons) - centroid
                    # new_diff = target.beacons[inds, :] - new_beacons
            # find the rotation that minimizes the distance
            steps += 1
        pass

    @classmethod
    def scan_list(cls, target: Scanner, to_be_scanned: Set[Scanner]) -> Set[Scanner]:
        to_remove = set()

        for scanner in to_be_scanned:
            converged, offset, rotation = cls.run_icp(target, scanner)
            if converged:
                target.add_new_points(scanner, offset, rotation)
                to_remove |= {scanner}

        return to_be_scanned - to_remove

    def solve_part1(self) -> str:
        to_check = copy.deepcopy(self.scanners[0])
        to_scan = set(self.scanners[1:])

        while len(to_scan) > 0:
            to_scan = self.scan_list(to_check, to_scan)

        self.to_check = to_check

        return f'{to_check.beacons.shape[0]}'

    def solve_part2(self) -> str:
        return f'{self.largest_distance()}'


class InfiniteImageConvolver:
    def __init__(self, state: np.ndarray, rule: np.ndarray):
        self.state, self.rule, self.boundary_value = state, rule, 0
        self.vision = 2 ** (np.arange(9).reshape((3, 3)))
        self.vfunc = np.vectorize(lambda x: self.rule[x])

    def __repr__(self):
        ret_val = ''
        for i in range(self.state.shape[0]):
            for j in range(self.state.shape[1]):
                ret_val += '#' if self.state[i, j] else '.'
            ret_val += '\n'
        return ret_val

    def update(self):
        self.state = self.vfunc(convolve2d(self.state, self.vision, mode='full', boundary='fill',
                                           fillvalue=self.boundary_value))
        self.boundary_value = int(self.rule[self.boundary_value * 511])


class Day20(AdventProblem):
    def __init__(self, test: bool):
        super().__init__('Infinite images')
        image_file = 'day20_test.txt' if test else 'day20_image.txt'
        image_lines = open(image_file, 'r').readlines()

        algo_mode = True

        self.algorithm: List[int] = None
        self.image: np.ndarray = None
        image_list: List[List[int]] = []

        for image_line in image_lines:
            if algo_mode:
                self.algorithm = [True if x == '#' else False for x in image_line.strip()]
                algo_mode = False
            elif len(image_line.strip()) > 0:
                image_list.append([True if x == '#' else False for x in image_line.strip()])

        self.image = np.array(image_list, dtype=bool)
        self.twerker = InfiniteImageConvolver(self.image, self.algorithm)

    def solve_part1(self) -> str:
        for _ in range(2):
            self.twerker.update()

        return f'{np.sum(self.twerker.state)}'

    def solve_part2(self) -> str:
        for _ in range(2, 50):
            self.twerker.update()

        return f'{np.sum(self.twerker.state)}'


class Die(ABC):
    def __init__(self):
        self.roll_count = 0

    def do_roll(self) -> int:
        self.roll_count += 1
        return self.get_roll()

    @abstractmethod
    def get_roll(self) -> int:

        """
        Get the value of the roll
        """


class DeterministicDie(Die):
    def __init__(self):
        super().__init__()
        self.num = 0

    def get_roll(self) -> int:
        self.num += 1
        if self.num > 100:
            self.num = 1
        return self.num


class Player:
    def __init__(self, num: int, location: int):
        self.num = num
        self.location = location
        self.score = 0

    def move_roll(self, die):
        rolls: List[int] = []
        for _ in range(3):
            rolls.append(die.do_roll())

        result = sum(rolls)
        new_location = ((self.location + result - 1) % 10) + 1
        self.score += new_location
        self.location = new_location

    def check_for_win(self):
        return self.score >= 1000


WorldState = namedtuple('WorldState', ['p1_loc', 'p1_score', 'p2_loc', 'p2_score'])


class Day21(AdventProblem):
    def __init__(self, test: bool):
        super().__init__('Dirac Dice')
        sensor_file = 'day21_test.txt' if test else 'day21_positions.txt'
        sensor_lines = open(sensor_file, 'r').read()
        (_, p1_start), (_, p2_start) = re.findall("Player (\d+) starting position: (\d+)",
                                                  sensor_lines)
        self.players = [Player(1, location=int(p1_start)), Player(2, location=int(p2_start))]

    def solve_part1(self) -> str:
        die, players, winner = DeterministicDie(), copy.deepcopy(self.players), None

        for player in itertools.cycle(players):
            player.move_roll(die)
            # print(f"player {player.num} score: {player.score}")
            if player.check_for_win():
                winner = player
                break

        loser = players[winner.num % 2]

        return f'{loser.score * die.roll_count}'

    @classmethod
    def prune_winners(cls, state_dict: Counter, winning_count: Tuple[int, int]) -> Counter:
        # state_dict_copy = state_dict.copy()
        to_remove = []
        for state in state_dict:
            if state.p1_score >= 21:
                winning_count[0] += state_dict[state]
                to_remove.append(state)
            elif state.p2_score >= 21:
                winning_count[1] += state_dict[state]
                to_remove.append(state)

        for state in to_remove:
            del state_dict[state]
        return state_dict

    @classmethod
    def transition_state(cls, state_dict: Counter, pnum: int):
        state_dict_new = Counter()
        for state in state_dict:
            sent = state_dict[state]

            for r1, r2, r3 in itertools.product(range(1, 3 + 1), repeat=3):
                rsum = r1 + r2 + r3
                location1, score1, location2, score2 = state
                if pnum == 0:
                    location1 = (-1 + rsum + location1) % 10 + 1
                    score1 += location1
                else:
                    location2 = (-1 + rsum + location2) % 10 + 1
                    score2 += location2

                new_state = WorldState(location1, score1, location2, score2)
                state_dict_new[new_state] += sent
        return state_dict_new

    def solve_part2(self) -> str:
        p1, p2 = self.players[0], self.players[1]
        initial_state = WorldState(p1.location, p1.score, p2.location, p2.score)
        state_dict: Counter[WorldState, int] = Counter()
        state_dict[initial_state] = 1
        winning_count: List[int] = [0, 0]

        for turn in range(14):
            for player_turn in range(2):
                state_dict = self.transition_state(state_dict, player_turn)
                self.prune_winners(state_dict, winning_count)

        return f'{max(winning_count)}'


class NiceSlice:
    """A class implementing the algebra of left-closed integer intervals [a, b)."""
    def __init__(self, start: int = -inf, stop: int = inf):
        self.__check_start_stop(start, stop)
        self.__start = start
        self.__stop = stop

    @classmethod
    def from_slice(cls, slc: slice) -> "NiceSlice":
        start = slc.start if slice.start is not None else -inf
        stop = slc.stop if slice.stop is not None else inf
        return NiceSlice(start, stop)

    @classmethod
    def __check_start_stop(cls, start: Optional[int], stop: Optional[int]) -> None:
        if stop < start:
            raise ValueError(f"Stop ({stop}) should not be less than start ({start}).")


    @property
    def start(self):
        return self.__start

    @start.setter
    def start(self, start: int):
        self.__check_start_stop(start, self.stop)
        self.__start = start

    @property
    def stop(self):
        return self.__stop

    @stop.setter
    def stop(self, stop: int):
        self.__check_start_stop(self.start, stop)
        self.__stop = stop

    @property
    def is_empty(self):
        return self.stop == self.start

    @property
    def is_singleton(self):
        return self.start == self.stop - 1

    def shift_cp(self, offset: int) -> "NiceSlice":
        return NiceSlice(self.__start + offset, self.__stop + offset)

    def disjoint(self, other: "NiceSlice") -> bool:
        if self.stop <= other.start or other.stop <= self.start:
            return True
        return False

    def adjacent(self, other: "NiceSlice") -> bool:
        return self.stop == other.start or self.start == other.stop

    def __or__(self, other: "NiceSlice") -> "NiceSlice":
        if self.start <= other.start <= self.stop <= other.stop:
            return NiceSlice(self.start, other.stop)
        if other.start <= self.start <= other.stop <= self.stop:
            return NiceSlice(other.start, self.stop)
        if self <= other:
            return other
        if other <= self:
            return self
        if self.is_empty:
            return other
        if other.is_empty:
            return self
        if self.disjoint(other):
            raise NotImplementedError("A union of disjoint NiceSlices is not a NiceSlice.")

    def union(self, other: "NiceSlice") -> "NiceSlice":
        return self | other

    def __and__(self, other: "NiceSlice") -> "NiceSlice":
        if self.start <= other.start < self.stop <= other.stop:
            return NiceSlice(other.start, self.stop)
        if other.start <= self.start < other.stop <= self.stop:
            return NiceSlice(self.start, other.stop)
        if self <= other:
            return self
        if other <= self:
            return other
        return NiceSlice(0, 0)

    def intersect(self, other: "NiceSlice") -> "NiceSlice":
        return self & other

    def __eq__(self, other: "NiceSlice") -> bool:
        return self.start == other.start and self.stop == other.stop

    def __lt__(self, other: "NiceSlice") -> bool:
        return (other.start <= self.start <= self.stop < other.stop) or (other.start < self.start <= self.stop <= other.stop)

    def __le__(self, other: "NiceSlice") -> bool:
        return other.start <= self.start <= self.stop <= other.stop

    def __contains__(self, i: int) -> bool:
        return self.start <= i < self.stop

    def combine(self, other: "NiceSlice") -> "NiceSlice":
        if not self.adjacent(other):
            raise NotImplementedError("Can't combine non-adjacent NiceSlices.")
        return self | other

    def to_slice(self) -> slice:
        start = None if abs(self.start) == inf else self.start
        stop = None if abs(self.stop) == inf else self.stop
        return slice(start, stop)

    def __repr__(self) -> str:
        return f'[{self.start} to {self.stop-1}]'


class CubeSlice:
    def __init__(self, on: bool, x: NiceSlice, y: NiceSlice, z: NiceSlice):
        self.on = on
        self.x, self.y, self.z = x, y, z

    def intersects(self, other: 'CubeSlice') -> bool:
        if self.x.disjoint(other.x) or self.y.disjoint(other.y) or self.z.disjoint(other.z):
            return False
        else:
            return True

    def intersect(self, other: 'CubeSlice') -> 'CubeSlice':
        new_x = self.x.intersect(other.x)
        new_y = self.y.intersect(other.y)
        new_z = self.z.intersect(other.z)
        return CubeSlice(True, new_x, new_y, new_z)

    def combine(self, other: 'CubeSlice') -> bool:
        me = [self.x, self.y, self.z]
        them = [other.x, other.y, other.z]
        eqs = [False]*3
        nequal = 0
        last_neq = None
        for i in range(3):
            eqs[i] = me[i] == them[i]
            nequal += 1 if eqs[i] else 0
            if not eqs[i]:
                last_neq = i
        if nequal < 2:
            return False
        elif nequal == 2:
            assert last_neq is not None
            if me[last_neq].adjacent(them[last_neq]):
                me[last_neq] = me[last_neq].combine(them[last_neq])
                self.x, self.y, self.z = me
                return True
            else:
                return False
        else:
            # do nothing - they are already the same
            return True

    def __repr__(self) -> str:
        return f"{'on' if self.on else 'off'} " \
               f'x={self.x.start}..{self.x.stop-1},' \
               f'y={self.y.start}..{self.y.stop-1},' \
               f'z={self.z.start}..{self.z.stop-1}'

    def __eq__(self, other: "NiceSlice") -> bool:
        return self.on == other.on and \
               self.x == other.x and self.y == other.y and self.z == other.z


class Cube:
    def __init__(self):
        self.regions: List[CubeSlice] = []

    @classmethod
    def add_or_combine(cls, test_region: CubeSlice, regions: List[CubeSlice]):
        for region in regions:
            if region.combine(test_region):
                return

        regions.append(test_region)

    @classmethod
    def get_regions_product(cls, test_region: CubeSlice, interx: CubeSlice):
        ix, iy, iz = interx.x, interx.y, interx.z
        xr, yr, zr = [], [], []
        if test_region.x.start < ix.start:
            xr.append(NiceSlice(test_region.x.start, ix.start))
        if test_region.y.start < iy.start:
            yr.append(NiceSlice(test_region.y.start, iy.start))
        if test_region.z.start < iz.start:
            zr.append(NiceSlice(test_region.z.start, iz.start))
        xr.append(ix)
        yr.append(iy)
        zr.append(iz)
        if ix.stop < test_region.x.stop:
            xr.append(NiceSlice(ix.stop, test_region.x.stop))
        if iy.stop < test_region.y.stop:
            yr.append(NiceSlice(iy.stop, test_region.y.stop))
        if iz.stop < test_region.z.stop:
            zr.append(NiceSlice(iz.stop, test_region.z.stop))

        return itertools.product(xr, yr, zr)

    def add_or_split(self, test_regions: List[CubeSlice]) -> Tuple[List[CubeSlice],List[CubeSlice]]:
        new_tests = []
        to_add = []
        for test_region in test_regions:
            disjoint = True
            for region in self.regions:
                if region.intersects(test_region):
                    disjoint = False
                    interx = region.intersect(test_region)
                    if test_region == interx:
                        # if same as the intersect, this is a proper subset - break
                        break

                    # otherwise, check for parts hanging off and add them
                    for x, y, z in self.get_regions_product(test_region, interx):
                        split_cube = CubeSlice(True, x, y, z)
                        if not split_cube.intersects(region):
                            self.add_or_combine(split_cube, new_tests)
                    break

            if disjoint:
                self.add_or_combine(test_region, to_add)

        return new_tests, to_add

    def add_region(self, to_add: CubeSlice) -> None:
        if to_add.on:
            test_regions = [to_add]
            while len(test_regions) > 0:
                test_regions, add_list = self.add_or_split(test_regions)
                self.regions.extend(add_list)
        else:
            new_regions = []
            for region in self.regions:
                if region.intersects(to_add):
                    interx = region.intersect(to_add)
                    if region == interx:
                        # we are turning the whole thing off, i.e. don't add it to the new
                        continue

                    for x, y, z in self.get_regions_product(region, interx):
                        split_cube = CubeSlice(True, x, y, z)
                        if split_cube != interx:
                            self.add_or_combine(split_cube, new_regions)
                else:
                    new_regions.append(region)

            self.regions = new_regions

    def get_total_volume(self) -> int:
        tsum = 0
        for region in self.regions:
            # TODO: add this to NiceSlice
            xd = region.x.stop - region.x.start
            yd = region.y.stop - region.y.start
            zd = region.z.stop - region.z.start
            tsum += xd*yd*zd

        return tsum

class Day22(AdventProblem):
    def __init__(self, test: bool):
        super().__init__('')
        reactor_file = 'day22_test.txt' if test else 'day22_reactor.txt'
        reactor_lines = open(reactor_file, 'r').readlines()

        self.slices = []

        for reactor_line in reactor_lines:
            on_str, slices_str = reactor_line.strip().split(' ')
            x_str, y_str, z_str = slices_str.split(',')
            x_strs = x_str.split('=')[1].split('..')
            y_strs = y_str.split('=')[1].split('..')
            z_strs = z_str.split('=')[1].split('..')
            x = NiceSlice(int(x_strs[0]), int(x_strs[1])+1)
            y = NiceSlice(int(y_strs[0]), int(y_strs[1])+1)
            z = NiceSlice(int(z_strs[0]), int(z_strs[1])+1)
            self.slices.append(CubeSlice(on_str == 'on', x, y, z))

    def solve_part1(self) -> str:
        region = np.zeros((101, 101, 101), dtype=bool)
        for slc in self.slices:
            xo, yo, zo = slc.x.shift_cp(50), slc.y.shift_cp(50), slc.z.shift_cp(50)
            region[xo.to_slice(), yo.to_slice(), zo.to_slice()] = slc.on
        return f'{np.sum(region)}'

    def solve_part2(self) -> str:
        cube = Cube()
        for test_region in self.slices:
            cube.add_region(test_region)

        return f'{cube.get_total_volume()}'

class Amphipod:
    def __init__(self, atype: str, row: int, column: int):
        assert 'A' <= atype <= 'D'
        self.ntype = ord(atype) - ord('A')
        self.row = row
        self.column = column

    def get_correct_room_col(self) -> int:
        return self.ntype * 2 + 3

    def in_correct_room(self) -> bool:
        return self.row != 1 and self.column == self.get_correct_room_col()

    def __repr__(self) -> str:
        return f"{chr(self.ntype + ord('A'))} [{self.row},{self.column}]"

    def __eq__(self, other: 'Amphipod') -> bool:
        return self.ntype == other.ntype & self.row == other.row & self.column == other.column

    def __hash__(self):
        return hash((self.ntype, self.row, self.column))

class AmphipodState:
    def __init__(self, amphipods: List[Amphipod], cost: int = 0, nrows: int = 2):
        self.amphipods = amphipods
        self.cost = cost
        self.nrows = nrows
        self.moves: Set['AmphipodState'] = None
        self.cell_repr = np.zeros((nrows+3,13),dtype=int)
        self.cell_repr[1, 1:12] = 1
        self.cell_repr[3:3+nrows, 0:2] = -1
        self.cell_repr[3:3+nrows, 11:13] = -1
        for i in range(4):
            self.cell_repr[2:2+nrows, 2 * i + 3] = 1
        for amphipod in self.amphipods:
            self.cell_repr[amphipod.row, amphipod.column] = amphipod.ntype + 2

    def check_for_win(self) -> bool:
        won = True
        for amphipod in self.amphipods:
            if not amphipod.in_correct_room():
                won = False
                break
        return won

    def get_cell_repr(self) -> np.ndarray:
        return self.cell_repr

    def construct_moves(self) -> Set['AmphipodState']:
        moves = set()
        hallway = self.cell_repr[1, :]

        for anum, amphipod in enumerate(self.amphipods):
            room_above = self.cell_repr[2:amphipod.row, amphipod.column]
            my_room_col = amphipod.get_correct_room_col()
            my_room_vals = self.cell_repr[2:2+self.nrows, my_room_col]
            if not np.all(room_above == 1):
                # can't get out
                continue

            full_spaces = my_room_vals != 1
            wrong_spaces = np.logical_and(my_room_vals != amphipod.ntype + 2, full_spaces)

            if amphipod.column == my_room_col and not np.any(wrong_spaces):
                # we are looking good - no need to keep going
                continue

            # check if we can move back into our room
            if not np.any(wrong_spaces):
                # maybe we can!
                # first check if the hallway on the way there is empty
                start_from = amphipod.column + np.sign(my_room_col - amphipod.column)
                slc = slice(min(start_from, my_room_col), max(start_from, my_room_col) + 1)

                if np.all(hallway[slc] == 1):
                    # the hallway IS empty! now let's go into the deepest place we can
                    my_room_depth = np.max(np.where(np.logical_not(full_spaces)))+1
                    slclen = slc.stop - slc.start
                    new_cost = self.cost + (slclen + my_room_depth) * 10 ** amphipod.ntype
                    if amphipod.row != 1:  # in a room; add cost to get to hallway
                        to_hallway = amphipod.row - 1
                        new_cost += to_hallway * 10 ** amphipod.ntype
                    new_pods = copy.deepcopy(self.amphipods)
                    mover = new_pods[anum]
                    mover.row = 1 + my_room_depth
                    mover.column = my_room_col
                    moves.add(AmphipodState(new_pods, new_cost, self.nrows))
                    continue

            if amphipod.row != 1:  # room->hallway
                for hall_spot in [1, 2, 4, 6, 8, 10, 11]:
                    slc = slice(min(amphipod.column, hall_spot), max(amphipod.column,
                                                                     hall_spot) + 1)

                    # invalid_spot = False
                    # if hall_spot in [4, 6, 8]:
                    #     # we would go in the hallway which can block other moves. If any of those
                    #     # would ultimately prevent us from getting into our room, we should not move
                    #     # there
                    #     for squatter in my_room_vals:
                    #         if squatter != 1:
                    #             correct_column = (squatter - 2) * 2 + 3
                    #             slc2 = slice(min(correct_column, my_room_col),
                    #                          max(correct_column, my_room_col) + 1)
                    #
                    #             if slc2.start <= hall_spot < slc2.stop:
                    #                 invalid_spot = True
                    #                 break
                    #
                    # if invalid_spot:
                    #     continue

                    # okay, we can proceed. check if the hallway is open
                    if np.all(hallway[slc] == 1):
                        # it is open!
                        slclen = slc.stop - slc.start
                        new_cost = self.cost + (amphipod.row - 2 + slclen) * 10 ** amphipod.ntype
                        new_pods = copy.deepcopy(self.amphipods)
                        mover = new_pods[anum]
                        mover.row = 1
                        mover.column = hall_spot
                        moves.add(AmphipodState(new_pods, new_cost, self.nrows))

        return moves

    def get_moves(self) -> Set['AmphipodState']:
        if self.moves is None:
            self.moves = self.construct_moves()
        return self.moves

    def __lt__(self, other: 'AmphipodState'):
        return self.cost < other.cost

    def __eq__(self, other: 'AmphipodState'):
        return np.all(self.cell_repr == other.cell_repr)

    def __hash__(self):
        return hash(self.cost) ^ hash(tuple(self.amphipods))

    def __repr__(self):
        ret_val = ''
        plotter = {0: '#', -1: ' ', 1: '.', 2: 'A', 3: 'B', 4: 'C', 5:'D'}
        for i in range(self.cell_repr.shape[0]):
            for j in range(self.cell_repr.shape[1]):
                ret_val += plotter[self.cell_repr[i,j]]
            ret_val += '\n'
        return ret_val

    @classmethod
    def from_file(cls, amphipod_file) -> 'AmphipodState':
        amphipod_lines = open(amphipod_file, 'r').readlines()

        amphipods = []
        for line_num, amphipod_line in enumerate(amphipod_lines):
            amphipod_line = amphipod_line.strip()
            if line_num in [2,3]:
                types = [x for x in amphipod_line.split('#') if len(x) > 0]
                for i, atype in enumerate(types):
                    amphipods.append(Amphipod(atype, line_num, 2*i+3))

        return AmphipodState(amphipods)

    def heuristic_distance_to_go(self):
        ncorrect = [0] * 4

        dist = 0

        # count the number of correct per room
        for ntype in range(0, 4):
            room_vals = self.cell_repr[2:2+self.nrows, 2 * ntype + 3] - 2
            for room_val in room_vals[::-1]:
                if room_val == ntype:
                    ncorrect[ntype] += 1
                else:
                    break

        min_moves = [0]*4

        # assume no one is blocking each other, and just sum up the distance to go
        for amphipod in self.amphipods:
            my_room_col = amphipod.get_correct_room_col()
            my_room_vals = self.cell_repr[2:2+self.nrows, my_room_col]

            full_spaces = my_room_vals != 1
            wrong_spaces = np.logical_and(my_room_vals != amphipod.ntype + 2, full_spaces)

            diff = abs(amphipod.column - amphipod.get_correct_room_col())

            if amphipod.row == 1:
                min_moves[amphipod.ntype] += diff + 1
            elif diff != 0:
                # in a room but the wrong one - need row - 1 + diff + 1 spaces to get to my room
                min_moves[amphipod.ntype] += amphipod.row - 1 + diff + 1
            elif np.any(wrong_spaces):
                room_below = self.cell_repr[amphipod.row+1:2+self.nrows, amphipod.column]
                if np.any(room_below != amphipod.ntype + 2):
                    # in our own room but someone else is below us
                    # we'll need to at a minimum step aside, let them out, then come back in
                    min_moves[amphipod.ntype] += amphipod.row - 1 + 3

        # now update the min_moves to reflect the number wrong
        for ntype in range(0,4):
            if ncorrect[ntype] < self.nrows - 1:
                for i in range(self.nrows - ncorrect[ntype] - 1):
                    min_moves[ntype] += i+1
            dist += min_moves[ntype]*10**ntype

        return dist


class AmphipodBurrow:
    def __init__(self):
        self.winning_costs = set()

    def transition(self, old_states: Set['AmphipodState']) -> Tuple[Set['AmphipodState'], int]:

        new_states = set()
        for old_state in old_states:
            if old_state.check_for_win():
                self.winning_costs.add(old_state.cost)
                continue
            else:
                new_states |= old_state.get_moves()

        return new_states

    def get_minimum_path_cost(self, start: AmphipodState, end: AmphipodState) -> int:
        state = start
        priority_queue = []
        best_cost: Dict[AmphipodState, int] = dict()

        niter = 0
        while state != end:
            for state_ in state.get_moves():
                cost_ = state_.cost + state_.heuristic_distance_to_go()
                if state_ in best_cost and best_cost[state_] <= cost_:
                    continue
                best_cost[state_] = cost_ # min(best_cost.get(state, cost_), cost_)
                heapq.heappush(priority_queue, (cost_, state_))
            cost, state = heapq.heappop(priority_queue)
            niter += 1
            if niter % 10000 == 0:
                print(f'Iter {niter}, fringe size currently {len(priority_queue)}, cost: {cost}, State:\n{state}')

        return state.cost


class Day23(AdventProblem):
    def __init__(self, test: bool):
        super().__init__('Amphipod burrow solver')
        amphipod_file = 'day23_test.txt' if test else 'day23_amphipod.txt'
        self.start_state = AmphipodState.from_file(amphipod_file)
        self.end_state = AmphipodState.from_file('day23_end.txt')

    def solve_part1(self) -> str:
        burrow = AmphipodBurrow()
        cost = burrow.get_minimum_path_cost(self.start_state, self.end_state)

        return f'{cost}'

    def solve_part2(self) -> str:
        burrow = AmphipodBurrow()
        start_pods = copy.deepcopy(self.start_state.amphipods)
        for amphipod in start_pods:
            if amphipod.row > 2:
                amphipod.row += 2

        end_pods = copy.deepcopy(self.end_state.amphipods)
        for amphipod in end_pods:
            if amphipod.row > 2:
                amphipod.row += 2

        new_vals = [['D','C','B','A'],['D','B','A','C']]

        for i in range(4):
            for j in range(2):
                start_pods.append(Amphipod(new_vals[j][i], j+3, 2*i + 3))
                end_pods.append(Amphipod(chr(i + ord('A')), j+3, 2*i + 3))

        start_state = AmphipodState(start_pods, nrows=4)
        end_state = AmphipodState(end_pods, nrows=4)

        cost = burrow.get_minimum_path_cost(start_state, end_state)

        return f'{cost}'


ALU = namedtuple('ALU', ['w', 'x', 'y', 'z'])


class MonadOp:
    def __init(self, left: Optional['MonadOp'], right: Optional['MonadOp'], op: str,
               val: Optional[int]):

        self.left = left
        self.right = right
        self.op = op
        self.val = val

    def operate(self, values: Dict[str, Any]) -> int:
        if self.op == 'literal':
            assert self.val is not None
            return eval(self.val, values)
        else:
            assert self.left is not None and self.right is not None
            if self.op == 'add':
                return self.left.operate(values) + self.right.operate(values)
            elif self.op == 'mul':
                return self.left.operate(values) * self.right.operate(values)
            elif self.op == 'div':
                return self.left.operate(values) // self.right.operate(values)
            elif self.op == 'mod':
                return self.left.operate(values) % self.right.operate(values)
            elif self.op == 'eql':
                lval, rval = self.left.operate(values), self.right.operate(values)
                return 1 if lval == rval else 0


class MonadBlock:
    def __init__(self, block_num: int, input_var: str):
        self.lines = []
        self.num = block_num
        self.input_var = input_var
        self.depends = {}
        self.consts = []

    def parse(self):
        for num, line in enumerate(self.lines):
            tok = line.split(' ')
            if num in [3, 4, 14]:
                self.consts.append(int(tok[2]))

    @classmethod
    def check_int(cls, tok: str) -> bool:
        if tok is None or len(tok) == 0:
            return False
        if tok[0] in ('-', '+'):
            return tok[1:].isdigit()
        return tok.isdigit()

    @classmethod
    def get_val(cls, tok: str, localvals: Dict[str, int]) -> int:
        if cls.check_int(tok):
            return int(tok)
        else:
            return localvals[tok]

    def compute_forward(self, w_val: int, z_val: int) -> int:
         localvals = {'w': w_val, 'x': 0, 'y': 0, 'z': z_val}
         for line in self.lines:
             tok = line.split(' ')
             if tok[0] == 'add':
                 a, b = tok[1:3]
                 localvals[a] = localvals[a] + self.get_val(b, localvals)
             elif tok[0] == 'mul':
                 a, b = tok[1:3]
                 localvals[a] = localvals[a] * self.get_val(b, localvals)
             elif tok[0] == 'div':
                 a, b = tok[1:3]
                 localvals[a] = localvals[a] // self.get_val(b, localvals)
             elif tok[0] == 'mod':
                 a, b = tok[1:3]
                 localvals[a] = localvals[a] % self.get_val(b, localvals)
             elif tok[0] == 'eql':
                 a, b = tok[1:3]
                 localvals[a] = 1 if localvals[a] == self.get_val(b, localvals) else 0
             else:
                 raise ValueError(f'Invalid token {tok[0]}')

         return localvals['z']

    def compute_simplified(self, w: int, z: int) -> int:
        # compute the simplified version of the MONAD tree (based on staring at it for a long time)
        a, b, c = self.consts[0:3]
        if z % 26 + b == w:
            return z // a
        else:
            return 26*(z // a) + w + c


class Day24(AdventProblem):
    def __init__(self, test: bool):
        super().__init__('')
        # load the monad file
        monad_file = 'day24_test.txt' if test else 'day24_monad.txt'
        monad_lines = open(monad_file, 'r').readlines()
        block_num = -1
        self.monad_blocks = []
        current_block = None
        for monad_line in monad_lines:
            tok = monad_line.strip().split(' ')
            if tok[0] == 'inp':
                block_num += 1
                current_block = MonadBlock(block_num, tok[1])
                self.monad_blocks.append(current_block)
            else:
                current_block.lines.append(monad_line.strip())

        # parse the blocks to yoink the 3 constants: a, b, and c
        for monad_block in self.monad_blocks:
            monad_block.parse()

        nblocks = len(self.monad_blocks)

        # the z values that are possible during each step
        self.possible_zs_dict: Dict[int, Set[int]] = {nblocks: [0]}

        # run through in reverse order and get the z values that are possible
        last_possible_zs = {0}
        for block in self.monad_blocks[::-1]:
            a, b, c = block.consts[0:3]
            next_possible_zs = set()
            if a == 26 and b < 10:
                # need to hit case 1 whenever possible to ensure start/end z = 0
                # new_z needs to be such that new_z // 26 in possible_zs and new_z % 26 = w - b
                for possible_z in last_possible_zs:
                    for w in range(1,9+1):
                        z_p = w - b
                        if 0 <= z_p <= 25:
                            next_possible_zs.add(z_p + 26*possible_z)
            else:
                assert a == 1 and b >= 10
                # will hit case 2
                #    return 26 * z + w + c # case 2
                # new_z needs to be such that 26*new_z + w + c in possible_zs
                for possible_z in last_possible_zs:
                    for w in range(1, 9+1):
                        # 26*new_z + w + c == possible_z
                        new_z = (possible_z - w - c)//26
                        if 26*new_z + w + c == possible_z: # this is necessary (?)
                            next_possible_zs.add(new_z)

            print(f'At step {block.num}, there are {len(next_possible_zs)} z possibilities')
            self.possible_zs_dict[block.num] = next_possible_zs
            last_possible_zs = next_possible_zs

    def solve_part1(self) -> str:
        # now walk the path with the largest possible w such that the corresponding z is possible
        last_z = 0
        max_wzs = []
        for block in self.monad_blocks:
            max_wz_candidate = (0,0)
            valid_zs = self.possible_zs_dict[block.num+1]
            for w in range(1,10):
                new_z = block.compute_simplified(w, last_z)
                if new_z in valid_zs:
                    max_wz_candidate = (w, new_z)
            last_z = max_wz_candidate[1]
            max_wzs.append(max_wz_candidate)

        return f"{''.join([str(x[0]) for x in max_wzs])}"

    def solve_part2(self) -> str:
        last_z = 0
        min_wzs = []
        for block in self.monad_blocks:
            min_wz_candidate = (0,0)
            valid_zs = self.possible_zs_dict[block.num+1]
            for w in range(9,0,-1):
                new_z = block.compute_simplified(w, last_z)
                if new_z in valid_zs:
                    min_wz_candidate = (w, new_z)
            last_z = min_wz_candidate[1]
            min_wzs.append(min_wz_candidate)

        return f"{''.join([str(x[0]) for x in min_wzs])}"


class Day25(AdventProblem):
    def __init__(self, test: bool):
        super().__init__('')

        pass

    def solve_part1(self) -> str:
        return f''

    def solve_part2(self) -> str:
        return f''


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