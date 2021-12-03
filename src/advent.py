from abc import ABC, abstractmethod
import importlib
import argparse

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
        self.nlines = len(self.bits_lines)
        self.nbits = len(self.bits_lines[0].strip())
        self.bits = np.zeros((self.nbits,self.nlines),dtype=int)
        for line_num, line in enumerate(self.bits_lines):
            for num in range(self.nbits):
                self.bits[num,line_num] = line[num].strip()

        self.num0 = np.sum(self.bits == 0, axis=1)
        self.num1 = np.sum(self.bits == 1, axis=1)
        self.gamma = np.zeros(self.nbits, dtype=int)
        self.epsilon = np.zeros(self.nbits, dtype=int)
        for i in range(self.nbits):
            self.gamma[i] = 0 if self.num0[i] > self.num1[i] else 1
            self.epsilon[i] = 0 if self.num0[i] <= self.num1[i] else 1

    def solve_part1(self) -> None:
        gamma_num = self.gamma.dot(2**np.arange(self.gamma.size)[::-1])
        epsilon_num = self.epsilon.dot(2**np.arange(self.gamma.size)[::-1])

        return f'{gamma_num*epsilon_num}'

    def solve_part2(self) -> None:
        o2_possibles = np.ones(self.nlines, dtype=bool)
        co2_possibles = np.ones(self.nlines, dtype=bool)
        for i in range(self.nbits):
            num0_o2 = np.sum(self.bits[i,o2_possibles] == 0)
            num1_o2 = np.sum(self.bits[i,o2_possibles] == 1)
            num0_co2 = np.sum(self.bits[i,co2_possibles] == 0)
            num1_co2 = np.sum(self.bits[i,co2_possibles] == 1)
            o2_val = 1 if num1_o2 >= num0_o2 else 0
            co2_val = 0 if num0_co2 <= num1_co2 else 1
            if sum(o2_possibles) > 1:
                o2_possibles[o2_possibles] = \
                    o2_possibles[o2_possibles] & self.bits[i,o2_possibles] == o2_val
            if sum(co2_possibles) > 1:
                co2_possibles[co2_possibles] = \
                    co2_possibles[co2_possibles] & self.bits[i,co2_possibles] == co2_val

        oxygen_rating = self.bits[:,o2_possibles].transpose()
        o2_rating_num = oxygen_rating.dot(2**np.arange(oxygen_rating.size)[::-1])
        co2_rating = self.bits[:,co2_possibles].transpose()
        co2_rating_num = co2_rating.dot(2**np.arange(co2_rating.size)[::-1])

        return f'{o2_rating_num*co2_rating_num}'

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