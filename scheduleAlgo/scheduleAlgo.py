from typing import List
from datetime import datetime, timedelta
from enum import Enum
from constraint import Problem, AllDifferentConstraint
import json

class TaskType(Enum):
    WORK = "work"
    STUDY = "study"
    EXERCISE = "exercise"
    CHORES = "chores"
    HOBBY = "hobby"
    OTHER = "other"


class Task:
    def __init__(self, id: int, name: str, priority: str, length: int,
                 deadline: datetime, isRepeat: bool, isSeparable: bool, optionalDays: List[str],
                 optionalHours: List[int], rankListHistory: List[int], type: TaskType, description: str, start :datetime= "2023-03-28 9:00:00"):
        self.id = id
        self.name = name
        self.priority = priority
        self.start = start
        self.length = length
        self.deadline = deadline
        self.isRepeat = isRepeat
        self.isSeparable = isSeparable
        self.optionalDays = optionalDays
        self.optionalHours = optionalHours
        self.rankListHistory = rankListHistory
        self.type = type
        self.description = description


class ScheduleSettings:
    def __init__(self, startHour: int, endHour: int, minGap: int, maxHoursPerDay: int,
                 maxHoursPerTypePerDay: dict, minTimeFrame: int = 15):
        self.startHour = startHour
        self.endHour = endHour
        self.minGap = minGap
        self.maxHoursPerDay = maxHoursPerDay
        self.maxHoursPerTypePerDay = maxHoursPerTypePerDay
        self.minTimeFrame = minTimeFrame


def generate_schedule(tasks: List[Task], settings: ScheduleSettings) -> dict:
    schedule = {}
    task_variables = {}

    # Create time slots for the week
    start = datetime.now().replace(hour=settings.startHour, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=7)
    time_slots = []
    while start < end:
        time_slots.append(start)
        start += timedelta(minutes=settings.minTimeFrame)

    # Create variables for each task and time slot
    for task in tasks:
        task_variables[task.id] = []
        for slot in time_slots:
            task_variables[task.id].append(f"{task.id}-{slot.strftime('%Y-%m-%d %H:%M:%S')}")

    # Create CSP problem
    problem = Problem()

    # Create time slots for the week
    start = datetime.now().replace(hour=settings.startHour, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=7)
    time_slots = []
    while start < end:
        time_slots.append(start)
        start += timedelta(minutes=settings.minTimeFrame)

    # Create variables for each task and time slot
    for task in tasks:
        task_variables[task.id] = []
        for slot in time_slots:
            task_variables[task.id].append(f"{task.id}-{slot.strftime('%Y-%m-%d %H:%M:%S')}")

    # Create CSP problem
    problem = Problem()

    # Add variables and domains to the problem
    for task in tasks:
        for var in task_variables[task.id]:
            problem.addVariable(var, time_slots)

    # Add constraints to the problem
    for task in tasks:
        for i, slot1 in enumerate(task_variables[task.id]):
            for slot2 in task_variables[task.id][i + 1:]:
                problem.addConstraint(
                    lambda x, y: x + timedelta(minutes=task.length) + timedelta(minutes=settings.minGap) <= y,
                    (slot1, slot2))

        if not task.isSeparable:
            problem.addConstraint(AllDifferentConstraint(), task_variables[task.id])

    # Solve the problem
    solutions = problem.getSolutions()
    for sol in solutions:
        for task in tasks:
            schedule[task.id] = []
            for var in task_variables[task.id]:
                if sol[var] not in schedule.values():
                    schedule[task.id].append(sol[var])

    return schedule


if __name__ == "__main__":


    scheduleSettings = {
        "id": 1,
        "startHour": "9:00:00",
        "endHour": "18:00:00",
        "minGap": 15,
        "maxHoursPerDay": 5,
        "maxHoursPerTypePerDay": 3,
        "minTimeFrame": 15

    }


    Tasks = [
        {
            "id": 1,
            "startHour": "9:00:00",
            "endHour": "18:00:00",
            "minGap": 15,
            "maxHoursPerDay": 5,
            "maxHoursPerTypePerDay": 3,
            "minTimeFrame": 15
        },
        {
            "id": 1,
            "startHour": "9:00:00",
            "endHour": "18:00:00",
            "minGap": 15,
            "maxHoursPerDay": 5,
            "maxHoursPerTypePerDay": 3,
            "minTimeFrame": 15
        }
    ]

    # convert into JSON:
    scheduleSettingsObject = json.dumps(scheduleSettings)
    tasksObject = json.dumps(Tasks)
    # the result is a JSON string:
    generate_schedule(())
    print(tasksObject)