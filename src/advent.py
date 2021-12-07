import copy
from abc import ABC, abstractmethod
import importlib
import argparse
from typing import List, Optional

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
                growing = growing + 1
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
        self.bits = np.zeros((self.nbits,self.nlines),dtype=int)
        for line_num, line in enumerate(self.bits_lines):
            for num in range(self.nbits):
                self.bits[num,line_num] = line[num].strip()

    def solve_part1(self) -> None:
        num0, num1 = np.sum(self.bits == 0, axis=1), np.sum(self.bits == 1, axis=1)
        gamma = np.array([0 if num0[i] > num1[i] else 1 for i in range(self.nbits)])
        epsilon = np.array([0 if num0[i] <= num1[i] else 1 for i in range(self.nbits)])

        gamma_num = gamma.dot(2**np.arange(gamma.size)[::-1])
        epsilon_num = epsilon.dot(2**np.arange(epsilon.size)[::-1])

        return f'{gamma_num*epsilon_num}'

    def count_bins(self, i, check):
        return np.sum(self.bits[i, check] == 0), np.sum(self.bits[i, check] == 1)

    def solve_part2(self) -> None:
        o2_i, co2_i = np.ones(self.nlines, dtype=bool), np.ones(self.nlines, dtype=bool)

        for i in range(self.nbits):
            num0_o2, num1_o2 = self.count_bins(i, o2_i)
            num0_co2, num1_co2 = self.count_bins(i, co2_i)
            if sum(o2_i) > 1:
                o2_i[o2_i] = self.bits[i,o2_i] == (1 if num1_o2 >= num0_o2 else 0)
            if sum(co2_i) > 1:
                co2_i[co2_i] = self.bits[i,co2_i] == (0 if num0_co2 <= num1_co2 else 1)

        oxygen_rating = self.bits[:,o2_i].transpose()
        o2_rating_num = oxygen_rating.dot(2**np.arange(oxygen_rating.size)[::-1])
        co2_rating = self.bits[:,co2_i].transpose()
        co2_rating_num = co2_rating.dot(2**np.arange(co2_rating.size)[::-1])

        return f'{o2_rating_num*co2_rating_num}'


class BingoBoard:
    def __init__(self, board_lines: List[str]):
        self.board = np.zeros((5,5), dtype=int)
        self.marks = np.zeros((5,5), dtype=bool)
        self.already_won = False
        for i in range(5):
            line_tokens = board_lines[i].split()
            for j, line_token in enumerate(line_tokens):
                self.board[i,j] = int(line_token)

    def mark_number(self, number: int):
        if not self.already_won:
            self.marks[np.where(self.board == number)] = True

    def check_for_bingo(self):
        if self.already_won:
            return False
        winner = False
        for i in range(5):
            winner = winner | np.all(self.marks[i,:])
            winner = winner | np.all(self.marks[:,i])

        return winner

    def get_score(self, number: int):
        return np.sum(self.board[~self.marks])*number

    def set_already_won(self):
        self.already_won = True

    def reset(self):
        self.marks[:] = False


class Day4(AdventProblem):

    def __init__(self, test: bool):
        super().__init__('')
        bingo_file = 'day4_test.txt' if test else 'day4_bingo.txt'
        self.bingo_lines = open(bingo_file, 'r').readlines()
        self.bingo = [int(x) for x in self.bingo_lines[0].split(',')]
        cursor, self.boards, board_lines = 2, [], [None]*5

        while True:
            if cursor > len(self.bingo_lines):
                break
            for i in range(5):
                board_lines[i] = self.bingo_lines[cursor]
                cursor += 1

            self.boards.append(BingoBoard(board_lines))
            cursor += 1 # skip blank line

    def solve_part1(self) -> str:
        for number in self.bingo:
            for i, board in enumerate(self.boards):
                board.mark_number(number)
                if board.check_for_bingo():
                    return f'{board.get_score(number)}'

    def solve_part2(self) -> str:
        for board in self.boards:
            board.reset()

        for number in self.bingo:
            for board in self.boards:
                board.mark_number(number)
                if board.check_for_bingo():
                    last_board, last_number = board, number
                    board.already_won = True

        return f'{last_board.get_score(last_number)}'


class Day5(AdventProblem):

    def __init__(self, test: bool):
        super().__init__('')
        lines_file = 'day5_test.txt' if test else 'day5_lines.txt'
        lines = open(lines_file, 'r').readlines()
        nx, ny = 1000, 1000
        self.points1, self.points2 = np.zeros((nx,ny), dtype=int), np.zeros((nx,ny), dtype=int)
        for line in lines:
            start, stop = [token.split(',') for token in line.split('->')]
            start, stop = np.array([int(x) for x in start]), np.array([int(x) for x in stop])
            delta = stop - start
            if np.any(delta == 0):
                smin, smax = np.minimum(start, stop), np.maximum(start, stop)
                xs, xe = smin[0], smax[0] + 1
                ys, ye = smin[1], smax[1] + 1
                self.points1[xs:xe,ys:ye] += 1
            else:
                sign = np.sign(delta)
                assert np.abs(delta)[0] == np.abs(delta)[1]
                for i in range(np.abs(delta)[0]+1):
                    pt = start + sign*i
                    self.points2[pt[0],pt[1]] += 1

    def solve_part1(self) -> str:
        return f'{np.count_nonzero(self.points1 > 1)}'

    def solve_part2(self) -> str:
        return f'{np.count_nonzero(self.points1 + self.points2 > 1)}'


class Day6(AdventProblem):

    def __init__(self, test: bool):
        super().__init__('')
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
            min_fuel = min(min_fuel, np.round(np.sum(diff*(diff+1)//2)))
        return f'{min_fuel}'


def run_day(day: int, test: bool) -> None:
    module = importlib.import_module('advent')
    class_ = getattr(module, f'Day{day}')
    instance: AdventProblem = class_(test)
    print(f'Now solving Day {day} "{instance.name}":')
    print(f'Part 1 - {instance.solve_part1()}')
    print(f'Part 2 - {instance.solve_part2()}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser("advent.py")

    parser.add_argument(dest='day', help='which day to run')
    parser.add_argument('-t', '--test', dest='test', help='run the output on the test data',
                        action="store_true")
    parser.set_defaults(test=False)

    args = parser.parse_args()

    run_day(args.day, args.test)
