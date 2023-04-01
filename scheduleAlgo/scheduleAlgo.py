from typing import List
from datetime import datetime, timedelta
from enum import Enum
from ortools.sat.python import cp_model
from typing import List
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
                 deadline: datetime, isRepeat: bool, optionalDays: List[str],
                 optionalHours: List[float], rankListHistory: List[int], type: str, description: str,
                 start: datetime = None):
        self.id = id
        self.name = name
        self.priority = priority
        self.start = start if start else datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)
        self.length = length / 15
        self.deadline = deadline
        self.isRepeat = isRepeat
        self.optionalDays = optionalDays
        self.optionalHours = optionalHours                                   # two integers - start and end hours
        self.rankListHistory = rankListHistory
        self.type = type
        self.description = description


class ScheduleSettings:
    def __init__(self, startHour: str, endHour: str, minGap: int, maxHoursPerDay: int,
                 maxHoursPerTypePerDay: dict, minTimeFrame: int = 15):
        self.startHour = datetime.strptime(startHour, "%H:%M:%S").time()
        self.endHour = datetime.strptime(endHour, "%H:%M:%S").time()
        self.minGap = minGap
        self.maxHoursPerDay = maxHoursPerDay
        self.maxHoursPerTypePerDay = maxHoursPerTypePerDay
        self.minTimeFrame = minTimeFrame
        self.slotLength = int((datetime.combine(datetime.min, datetime.strptime('00:15:00', '%H:%M:%S').time()) - datetime.min).total_seconds() / 60)


    def numSlots(self) -> int:
        return int((datetime.combine(datetime.now(), self.endHour) - datetime.combine(datetime.now(), self.startHour)).total_seconds() / (60 * self.minTimeFrame))

    def numDays(self) -> int:
        return 7  # assuming weekly schedule

    def startDate(self) -> datetime:
        # assuming weekly schedule starting on Monday
        today = datetime.today()
        return today - timedelta(days=today.weekday())

    def last_day_of_week(self,date):
        days_until_end_of_week = 6 - date.weekday()
        last_day = date + timedelta(days=days_until_end_of_week)
        return last_day

    def next_day(self,date) -> datetime:
        tomorrow = date + timedelta(days=1)
        tomorrow = datetime(tomorrow.year, tomorrow.month, tomorrow.day, hour=self.startHour.hour)
        return tomorrow

    def slotLength(self) -> int:
        return self.minTimeFrame

    def number_slots_aday(self,allslots) -> int:
        return len(allslots) / 7
def overlaps(start1: int, end1: int, start2: int, end2: int) -> bool:
    return end1 > start2 and end2 > start1

def generate_schedule(tasks: List[Task], settings: ScheduleSettings, variables=None) -> dict:
    schedule = {}
    task_variables = {}
    variables = {}
    weekdays = {
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday"
    }
    # Create time slots for the week
    start = datetime.combine(settings.startDate(), settings.startHour)
    end = datetime.combine(settings.last_day_of_week(settings.startDate()), settings.endHour)
    time_slots = []
    time_slots_dict = {}
    slot_number = 1
    while start < end:
        time_slots.append(start)
        time_slots_dict[slot_number] = start
        slot_number += 1
        start += timedelta(minutes=settings.minTimeFrame)
        if start.hour == settings.endHour.hour:
            start = settings.next_day(start)


    # Create task variables
    # Create the CP-SAT model.
    model = cp_model.CpModel()

    # Create variables for each task and time slot
    #need to take care of the possibility that optional hours can be float
    for task in tasks:
        variables[task.id] = {}
        for slot in time_slots:
            hour_int_start = int(task.optionalHours[0])  # get the integer part of the hour
            minute_int_start = int((task.optionalHours[0] - hour_int_start) * 60)  # get the minute part of the hour
            hour_int_end = int(task.optionalHours[1])  # get the integer part of the hour
            minute_int_end = int((task.optionalHours[1] - hour_int_end) * 60)  # get the minute part of the hour

            # create a datetime object with today's date and the corresponding time
            start_hour_minute = datetime(1, 1, 1, hour_int_start, minute_int_start).time()
            time_task_may_start = datetime.combine(slot, start_hour_minute)
            end_hour_minute = datetime(1, 1, 1, hour_int_end, minute_int_end).time()
            time_task_may_end = datetime.combine(slot, end_hour_minute)
            if weekdays[slot.weekday()] in task.optionalDays and slot >= time_task_may_start and slot + timedelta(minutes=settings.minGap) <= time_task_may_end:
                # task_variables[task.id].append(f"{task.id}-{slot}")
                var_name = f"{task.id}-{slot}"
                variables[task.id][slot] = model.NewBoolVar(var_name)







        # Add constraints for each task
    for task in tasks:
        for i, slot1 in enumerate(time_slots):
            for j, slot2 in enumerate(time_slots):
                if i == j:
                    continue
                duration = task.length
                if slot2 < slot1:
                    duration += int((slot1 - slot2).total_seconds() / 60)
                else:
                    duration += int((slot2 - slot1).total_seconds() / 60)
                if duration <= settings.minTimeFrame:
                    model.Add(variables[task.id][slot1] + variables[task.id][slot2] <= 1)
                else:
                    for k in range(i + 1, j):
                        model.Add(variables[task.id][time_slots[k]] == 0)
                    model.Add(variables[task.id][slot1] + variables[task.id][slot2] <= 1)


    # Add constraints for each pair of overlapping tasks
    for i, task1 in enumerate(tasks):
        for j, task2 in enumerate(tasks):
            if i == j:
                continue
            if task1.start.date() != task2.start.date():
                continue
            if overlaps(task1.start, task1.start + timedelta(minutes=task1.length),
                        task2.start, task2.start + timedelta(minutes=task2.length)):
                for slot in time_slots:
                    model.Add(variables[task1.id][slot] + variables[task2.id][slot] <= 1)





     # Solve the model and return the schedule
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            for task in tasks:
                schedule[task.id] = []
                for slot in time_slots:
                    if solver.Value(variables[task.id][slot]) == 1:
                        schedule[task.id].append(
                            datetime.combine(datetime.today().date(), slot.time()).strftime("%Y-%m-%d %H:%M:%S"))
    return schedule



if __name__ == "__main__":

    scheduleSettings = {
        "startHour": "9:00:00",
        "endHour": "18:00:00",
        "minGap": 15,
        "maxHoursPerDay": 5,
        "maxHoursPerTypePerDay": {"A": 3, "B": 2},
        "minTimeFrame": 15
    }

    schedule = ScheduleSettings(**scheduleSettings)

    print(schedule.startHour)  # Output: 09:00:00
    print(schedule.endHour)  # Output: 18:00:00
    print(schedule.minGap)  # Output: 0:15:00
    print(schedule.maxHoursPerDay)  # Output: 5
    print(schedule.maxHoursPerTypePerDay)  # Output: {'A': 3, 'B': 2}
    print(schedule.minTimeFrame)  # Output: 0:15:00

    tasks_data = [
        {
            "id": 1,
            "name": "Task 1",
            "priority": "high",
            "length": 60,
            "deadline": datetime(2023, 4, 5, 18, 0, 0),
            "isRepeat": False,
            "optionalDays": ["Monday", "Wednesday", "Friday"],
            "optionalHours": [11, 14],
            "rankListHistory": [1, 2, 3],
            "type": "A",
            "description": "This is task 1"
        },
        {
            "id": 2,
            "name": "Task 2",
            "priority": "medium",
            "length": 45,
            "deadline": datetime(2023, 4, 2, 18, 0, 0),
            "isRepeat": True,
            "optionalDays": ["Tuesday", "Thursday"],
            "optionalHours": [9, 13],
            "rankListHistory": [2, 1, 3],
            "type": "B",
            "description": "This is task 2"
        },
        {
            "id": 3,
            "name": "Task 3",
            "priority": "low",
            "length": 30,
            "deadline": datetime(2023, 4, 1, 18, 0, 0),
            "isRepeat": False,
            "optionalDays": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
            "optionalHours": [10, 12],
            "rankListHistory": [3, 2, 1],
            "type": "B",
            "description": "This is task 3"
        }
    ]

    tasks = []
    for data in tasks_data:
        task = Task(**data)
        tasks.append(task)


    print(generate_schedule(tasks,schedule))
    # convert into JSON:
    # a1 = json.dumps(scheduleSettingsJSON)
    # a2 = json.dumps(TasksJSON)
    # the result is a JSON string:
    # generate_schedule(tasksObject, a2)

def SimpleSatProgram():
    """Minimal CP-SAT example to showcase calling the solver."""
    # Creates the model.
    model = cp_model.CpModel()

    # Creates the variables.
    num_vals = 3
    x = model.NewIntVar(0, num_vals - 1, 'x')
    y = model.NewIntVar(0, num_vals - 1, 'y')
    z = model.NewIntVar(0, num_vals - 1, 'z')

    # Creates the constraints.
    model.Add(x != y)

    # Creates a solver and solves the model.
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print('x = %i' % solver.Value(x))
        print('y = %i' % solver.Value(y))
        print('z = %i' % solver.Value(z))
    else:
        print('No solution found.')


# SimpleSatProgram()