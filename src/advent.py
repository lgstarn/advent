from abc import ABC, abstractmethod
import importlib
import argparse

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

    def __init__(self):

        super().__init__('do the submarine soundings increase?')
        self.sounding_lines = open('day1_soundings.txt', 'r').readlines()

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

    def __init__(self):

        super().__init__('find the submarine path')
        self.direction_lines = open('day2_directions.txt', 'r').readlines()

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


def run_day(day: int) -> None:
    module = importlib.import_module('advent')
    class_ = getattr(module, f'Day{day}')
    instance: AdventProblem = class_()
    print(f'Now solving Day {day} "{instance.name}":')
    print(f'Part 1 - {instance.solve_part1()}')
    print(f'Part 2 - {instance.solve_part2()}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser("advent.py")

    parser.add_argument(dest='day', help='which day to run')

    args = parser.parse_args()

    run_day(args.day)