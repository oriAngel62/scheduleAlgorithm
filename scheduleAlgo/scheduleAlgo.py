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
    def __init__(self, id: int, priority: str, length: int,
                 deadline: datetime, isRepeat: bool, optionalDays: List[str],
                 optionalHours: List[float], rankListHistory: List[int], type: str, description: str,
                 start: datetime = None):
        self.id = id
        self.priority = priority
        self.start = start if start else datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)
        self.length = length
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


# function to return key for any value
def get_key(val,dict):
    for key, value in dict.items():
        if val == value:
            return key
    return "key doesn't exist"

# def check_chosen_seq(task_id,seq,dict,variables,settings):
#     index = []
#     for i, slot in enumerate(seq):
#         index[i] = get_key(slot,dict)
#     if sum(variables[(j, task_id)] for j in len(index) ==  int(tasks[i].length / settings.minTimeFrame) + 1):




def generate_schedule(tasks: List[Task], settings: ScheduleSettings, variables=None) -> dict:
    schedule = {}
    date_format = "%Y-%m-%d %H:%M:%S"
    weekdays = {
        0: "Sunday",
        1: "Monday",
        2: "Tuesday",
        3: "Wednesday",
        4: "Thursday",
        5: "Friday",
        6: "Saturday"
    }
    # Create time slots for the week
    start = datetime.combine(settings.startDate(), settings.startHour)
    end = datetime.combine(settings.last_day_of_week(settings.startDate()), settings.endHour)
    time_slots = []
    time_slots_dict = {}
    slot_number = 0
    while start < end:
        time_slots.append(start)
        time_slots_dict[slot_number] = start
        slot_number += 1
        start += timedelta(minutes=settings.minTimeFrame)
        if start.hour == settings.endHour.hour:
            start = settings.next_day(start)

    num_slots = len(time_slots)
    num_tasks = len(tasks)

    # Create task variables
    # Create the CP-SAT model.
    model = cp_model.CpModel()

    variables = {}
    for i in range(1,8):
        for j in range(num_slots):
            # A variable is True if time slot i is assigned to task j.
            variables[(i, j)] = model.NewBoolVar(f"day{i}_slot{j}")

    # Add the constraints.

    total_length = 0
    gap = 1
    num_of_slots_in_hour = 60 / settings.minTimeFrame
    max_slots_per_day = settings.maxHoursPerDay * num_of_slots_in_hour
    for task in tasks:
        num_of_slots_in_task = task.length / settings.minTimeFrame
        total_length = total_length + num_of_slots_in_task + gap


    #maxHoursPerDay constrain
    for i in range(1,8):
        model.Add(sum(variables[(i, j)] for j in range(num_slots)) == int(max_slots_per_day))

    # Each time slot can only be linked to one task.
    model.Add(sum(variables[(i, j)] for i in range(1,8) for j in range(num_slots)) == int(total_length) )


    task_to_index_its_options = {}
    consecutive_slots = {}
    #Each task can be linked with only consecutive time slots according to its durability.
    for j, task in enumerate(tasks):
        consecutive_slots[task.id] = []
        # task_to_index_its_options[j] = len(consecutive_slots)
        hour_int_start = int(task.optionalHours[0])  # get the integer part of the hour
        minute_int_start = int(
            (task.optionalHours[0] - hour_int_start) * 60)  # get the minute part of the hour
        hour_int_end = int(task.optionalHours[1])  # get the integer part of the hour
        minute_int_end = int(
            (task.optionalHours[1] - hour_int_end) * 60)  # get the minute part of the hour


        for i in range(num_slots - int(task.length / settings.minTimeFrame)):
            count_seq = 0
            start_hour_minute = datetime(1, 1, 1, hour_int_start, minute_int_start).time()
            time_task_may_start = datetime.combine(time_slots[i], start_hour_minute)
            end_hour_minute = datetime(1, 1, 1, hour_int_end, minute_int_end).time()
            time_task_may_end = datetime.combine(time_slots[i], end_hour_minute)
            time_task_may_end = time_task_may_end + timedelta(minutes=15)
            if not weekdays[time_slots[i].weekday()] in task.optionalDays or time_slots[i] < time_task_may_start or time_slots[i] > time_task_may_end:
                continue
            for k in range(i, i + int(task.length / settings.minTimeFrame) + 1):
                # create a datetime object with today's date and the corresponding time
                if k == i or time_slots[k] != timedelta(minutes=settings.minTimeFrame) + time_slots[k - 1]:
                    # Start a new consecutive sequence.
                    consecutive_sequence = [time_slots[k]]
                else:
                    # Add to the current consecutive sequence and add a new sequence.
                    if k == i + int(task.length / settings.minTimeFrame):
                        consecutive_sequence.append(time_slots[k])
                        if len(consecutive_sequence) == int(task.length / settings.minTimeFrame) + 1:
                            every_slot_counter = 0
                            for slot in consecutive_sequence:
                                if weekdays[slot.weekday()] in task.optionalDays and slot >= time_task_may_start and slot <= time_task_may_end:
                                    every_slot_counter += 1
                            if every_slot_counter == int(task.length / settings.minTimeFrame) + 1:
                                # tuple of (sequence,day,startslot,endslot)
                                consecutive_slots[task.id].append((consecutive_sequence,slot.weekday(),i,k))

                                # valid_sequences[j].append(consecutive_sequence)
                                count_seq += 1
                        consecutive_sequence = []
                    else:
                        consecutive_sequence.append(time_slots[k])


    i=1
    # seq_var = {}
    # for task in tasks:
    #     task_seq = consecutive_slots[task.id]
    #     for i, sequence in enumerate(task_seq):
    #         seq_var[(task.id, i)] = model.NewBoolVar(f"task{task.id}_seq{i}")
    #     # sequence_vars = []
    #     # for sequence in task_seq:
    #     #     sequence_vars.append(sequence)
    #     total_seq_to_task = len(task_seq)
    #     model.Add(sum(seq_var[(task.id, sequence_index)] for sequence_index in range(total_seq_to_task)) == 1)


    # Solve the model.
    solver = cp_model.CpSolver()
    status = solver.Solve(model)



    # Create the schedule.
    schedule = {}
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        # for i in range(num_slots):
        #     for j in range(num_tasks):
        #         if solver.BooleanValue(variables[(i, j)]):
        #                 schedule[time_slots[i].strftime(date_format)] = tasks[j].id
        num_of_tasks = len(tasks)
        for i in range(1, num_of_tasks + 1):
            for j, seq in enumerate(consecutive_slots[i]):
                if solver.BooleanValue(seq_var[(i, j)]):
                    for slot in seq: #seq_var[(i, j)]:
                        if solver.Value(variables[(get_key(slot,time_slots_dict),i)]) == 0:
                            model.Add(variables[(get_key(slot,time_slots_dict),i)] == 1)
                    # schedule[time_slots[get_key(consecutive_slots[i][j],i)].strftime(date_format)] = i
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            for i in range(num_slots):
                for j in range(num_tasks):
                    if solver.BooleanValue(variables[(i, j + 1)]):
                            schedule[time_slots[i].strftime(date_format)] = tasks[j].id


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
            "priority": "high",
            "length": 60,
            "deadline": datetime(2023, 4, 15, 18, 0, 0),
            "isRepeat": False,
            "optionalDays": ["Sunday"],
            "optionalHours": [9, 12.75],
            "rankListHistory": [1, 2, 3],
            "type": "A",
            "description": "This is task 1"
        },
        {
            "id": 2,
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
            "priority": "low",
            "length": 30,
            "deadline": datetime(2023, 4, 1, 18, 0, 0),
            "isRepeat": False,
            "optionalDays": ["Monday"],
            "optionalHours": [13, 14],
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