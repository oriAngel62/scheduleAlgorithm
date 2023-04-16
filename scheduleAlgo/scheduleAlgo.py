from typing import List
from datetime import datetime, timedelta
from enum import Enum
from ortools.sat.python import cp_model
from typing import List
import json
from math import prod
# from signalrcore import Connection
import random

class TaskType(Enum):
    WORK = "work"
    STUDY = "study"
    EXERCISE = "exercise"
    CHORES = "chores"
    HOBBY = "hobby"
    OTHER = "other"


SAUNDAY = 0
SATURDAY = 6
weekdays = {
    0: "Sunday",
    1: "Monday",
    2: "Tuesday",
    3: "Wednesday",
    4: "Thursday",
    5: "Friday",
    6: "Saturday"
}
RANK_POLICY = 1
NUMOFSOLUTIONS = 1
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
                 minTimeFrame: int = 15):
        self.startHour = datetime.strptime(startHour, "%H:%M:%S").time()
        self.endHour = datetime.strptime(endHour, "%H:%M:%S").time()
        self.minGap = minGap
        self.maxHoursPerDay = maxHoursPerDay
        self.minTimeFrame = minTimeFrame
        self.slotLength = int((datetime.combine(datetime.min, datetime.strptime('00:15:00', '%H:%M:%S').time()) - datetime.min).total_seconds() / 60)


    def numSlots(self) -> int:
        return int((datetime.combine(datetime.now(), self.endHour) - datetime.combine(datetime.now(), self.startHour)).total_seconds() / (60 * self.minTimeFrame))

    def numDays(self) -> int:
        return 7  # assuming weekly schedule

    def startDate(self) -> datetime:
        # assuming weekly schedule starting on Sunday
        today = datetime.today()
        return today - timedelta(days=today.weekday() + 1)

    def last_day_of_week(self) -> datetime:
        today = datetime.today()
        days_until_end_of_week = 5 - today.weekday()
        last_day = today + timedelta(days=days_until_end_of_week)
        return last_day
    def next_day(self, date) -> datetime:
        tomorrow = date + timedelta(days=1)
        tomorrow = datetime(tomorrow.year, tomorrow.month, tomorrow.day, hour=self.startHour.hour)
        return tomorrow

    def slotLength(self) -> int:
        return self.minTimeFrame

    def number_slots_aday(self, allslots) -> int:
        return len(allslots) / 7


    def get_end_hour(self):
        return self.endHour
def overlaps(start1: int, end1: int, start2: int, end2: int) -> bool:
    return end1 > start2 and end2 > start1


# function to return key for any value
def get_key(val,dict):
    for key, value in dict.items():
        if val == value:
            return key
    return "key doesn't exist"

def datetime_to_slot(datetime_obj, time_slots_dict):
    day_of_week = (datetime_obj.weekday() + 1) % 7  # Adjust for custom weekday start
    for slot_number, time_slot in time_slots_dict.items():
        if time_slot.hour == datetime_obj.hour and time_slot.minute == datetime_obj.minute and day_of_week == time_slot.weekday():
            return (day_of_week, slot_number)
    return datetime_obj  # Return None if no slot found for given datetime

def init_variables(time_slots_dict):
    variables = {}
    for slot_number, time_slot in time_slots_dict.items():
        day_of_week = (time_slot.weekday() + 1) % 7
        variables[(day_of_week, slot_number)] = None
    return variables
def generate_schedule(tasks: List[Task], settings: ScheduleSettings, variables=None) -> dict:

    date_format = "%Y-%m-%d %H:%M:%S"
    # Create time slots for the week
    start = datetime.combine(settings.startDate(), settings.startHour)
    end = datetime.combine(settings.last_day_of_week(), settings.endHour)
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

    variables = {}
    for slot_number, time_slot in time_slots_dict.items():
        day_of_week = (time_slot.weekday() + 1) % 7
        variables[(day_of_week, slot_number)] = None


    total_length = 0
    gap = 1
    num_of_slots_in_hour = 60 / settings.minTimeFrame
    max_slots_per_day = settings.maxHoursPerDay * num_of_slots_in_hour
    for task in tasks:
        num_of_slots_in_task = task.length / settings.minTimeFrame
        total_length = total_length + num_of_slots_in_task + gap




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
        deadline = task.deadline


        for i in range(num_slots - int(task.length / settings.minTimeFrame)):
            count_seq = 0
            start_hour_minute = datetime(1, 1, 1, hour_int_start, minute_int_start).time()
            time_task_may_start = datetime.combine(time_slots[i], start_hour_minute)
            end_hour_minute = datetime(1, 1, 1, hour_int_end, minute_int_end).time()
            time_task_may_end = datetime.combine(time_slots[i], end_hour_minute)
            time_task_may_end = time_task_may_end + timedelta(minutes=15)
            if not weekdays[(time_slots[i].weekday() + 1) % 7] in task.optionalDays or time_slots[i] < time_task_may_start or time_slots[i] > time_task_may_end :
                continue
            if time_slots[i] > task.deadline:
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
                                if weekdays[
                                    (slot.weekday() + 1) % 7] in task.optionalDays and slot >= time_task_may_start and slot <= time_task_may_end:
                                    every_slot_counter += 1
                            if every_slot_counter == int(task.length / settings.minTimeFrame) + 1:
                                # Check if the sequence ends before the deadline.
                                # if datetime.combine(time_slots[k], end_hour_minute) <= task.deadline:
                                    # tuple of (sequence,day,startslot,endslot)
                                    consecutive_slots[task.id].append((consecutive_sequence, (slot.weekday() + 1) % 7, i, k))
                                    count_seq += 1
                        consecutive_sequence = []
                    else:
                        consecutive_sequence.append(time_slots[k])
    num_of_tasks = len(tasks)
    task_sum = 1

    switch_consecutive_slots_sequence_according_to_rank(tasks, consecutive_slots, time_slots_dict)
    high_priority_task_list = []
    medium_priority_task_list = []
    low_priority_task_list = []
    for task in tasks:
        if task.priority == "high":
            high_priority_task_list.append(task)
        elif task.priority == "medium":
            medium_priority_task_list.append(task)
        else:
            low_priority_task_list.append(task)


    solutions = {}
    for solution_index in range(0,NUMOFSOLUTIONS):
        sort_by_least_options(high_priority_task_list, consecutive_slots)
        sort_by_least_options(medium_priority_task_list, consecutive_slots)
        sort_by_least_options(low_priority_task_list, consecutive_slots)
        all_tasks_sorted_together = high_priority_task_list + medium_priority_task_list + low_priority_task_list

        # Try to schedule tasks one at a time
        schedule = {}
        variables = init_variables(time_slots_dict)
        result, unscheduled_tasks = backtrack(schedule, all_tasks_sorted_together, consecutive_slots, settings, variables,time_slots_dict, 1)
        if len(unscheduled_tasks) != 0:
            print(f"cannot schedule tasks - {unscheduled_tasks}")

        flipped = {}
        task_day_start_end = {}

        for key, value in result.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                flipped[value].append(key)

        for key, value in flipped.items():
            start = end_hour_minute.strftime("%H:%M:%S")
            datetime_start = None
            end = start_hour_minute.strftime("%H:%M:%S")
            datetime_end = None
            for i, slot in enumerate(value):
                slot_time = slot.strftime("%H:%M:%S")
                if slot_time < start :
                    datetime_start = slot
                if slot_time > end and i < len(value):
                    datetime_end = slot
            task_day_start_end[key] = (datetime_start,datetime_end)

        solutions[solution_index] = task_day_start_end, unscheduled_tasks
    return solutions


def sort_by_least_options(task_list, all_blocks):
    task_list.sort(key=lambda task: len(all_blocks[task.id]))
    sublists = {}
    prev = 0
    i = 0
    same_num_of_options_list = []
    for task in task_list:
        if len(all_blocks[task.id]) != prev:
            if prev != 0:
                sublists[i] = same_num_of_options_list
                i += 1
                same_num_of_options_list = []
            prev = len(all_blocks[task.id])
        same_num_of_options_list.append(task)

    if same_num_of_options_list != None:
        sublists[i] = same_num_of_options_list

    # Shuffle each sublist.
    for sublist in sublists.values():
        random.shuffle(sublist)

    # Merge the shuffled sublists into a single list.
    all_tasks_sorted_together = []
    for sublist in sublists.values():
        all_tasks_sorted_together.extend(sublist)

    return all_tasks_sorted_together

def checkLimitHourADay(variables, settings, day):
    summ = 0
    slotsADay = int(settings.number_slots_aday(variables))
    for i in range(slotsADay * day, slotsADay * (day + 1)):
        if variables[(day, i)] is not None:
            summ += 1
    return summ <= settings.maxHoursPerDay


def backtrack(schedule, tasks, consecutive_slots, settings, variables,time_slots_dict, current_task_index):
    # If all tasks have been scheduled, return the solution.
    if current_task_index > len(tasks):
        return schedule, []

    unscheduled_tasks = []

    for block_index, block in enumerate(consecutive_slots[tasks[current_task_index - 1].id]):
        slots, day, start_slot_index, end_slot_index = block
        for slot_index in range(start_slot_index, end_slot_index - len(slots) + 2):
            # Check if the current slot is already scheduled.
            conflict = False
            for i, slot in enumerate(slots):
                if variables[(day, slot_index + i)] is not None:
                    conflict = True
                    break
            if conflict:
                continue

            # max hour per day check
            if checkLimitHourADay(variables, settings, day):
                # If the current slot is not scheduled, schedule the task on these slots.
                for i, slot in enumerate(slots):
                    variables[(day, start_slot_index + i)] = tasks[current_task_index - 1].id
                    schedule[slot] = tasks[current_task_index - 1].id
            else:
                continue

            # Recursively try to schedule the next task.
            next_task_schedule, next_unscheduled_tasks = backtrack(schedule.copy(), tasks, consecutive_slots, settings,
                                                                   variables,time_slots_dict, current_task_index + 1)

            # If the next task has been scheduled, return the solution.
            if len(next_unscheduled_tasks) == 0:
                return next_task_schedule, next_unscheduled_tasks

            # If the next task could not be scheduled, add the current task to the unscheduled tasks list and undo the current task's schedule.
            for i, slot in enumerate(slots):
                variables[(day, start_slot_index + i)] = None
                del schedule[slot]

    # If the current task could not be scheduled in any of its consecutive blocks, add it to the unscheduled tasks list.
    unscheduled_tasks.append(tasks[current_task_index - 1].id)

    task_start_end_time = {}

    return schedule, unscheduled_tasks

def sort_by_rank_and_start_time(rank_list_history):
    return sorted(rank_list_history, key=lambda x: (x['rank'], x['startTime'] ), reverse=True)

def switch_consecutive_slots_sequence_according_to_rank(tasks_data, consecutive_slots, time_slots_dict):
    for task in tasks_data:
        sorted_task = sort_by_rank_and_start_time(task.rankListHistory)
        for i, consecutive_sequence in enumerate(consecutive_slots[task.id]):
            for rank_data in sorted_task:
                rank = rank_data['rank']
                start_time = rank_data['startTime']
                end_time = rank_data['endTime']
                assigned_slots = []

                #convert time to tuple
                tuple_start = datetime_to_slot(rank_data['startTime'], time_slots_dict)
                tuple_end = datetime_to_slot(rank_data['endTime'], time_slots_dict)

                consecutive_sequence_start = datetime_to_slot(consecutive_sequence[0][0], time_slots_dict)
                consecutive_sequence_end = datetime_to_slot(consecutive_sequence[0][-1], time_slots_dict)

                if consecutive_sequence_start == tuple_start and consecutive_sequence_end == tuple_end and rank > RANK_POLICY:
                    consecutive_slots[task.id].pop(i)
                    consecutive_slots[task.id].insert(0, consecutive_sequence)
                    break

def on_solution_received(solution):
    print("Solution received by server:", solution)
def siganlRConnection(solution):
    # Connect to the SignalR hub on the server
    with Connection("http://localhost:3000/signalr") as connection:
        # Set up the function to be called when the server receives the solution
        connection.on("ReceiveSolution", on_solution_received)

        # Start the connection
        connection.start()

        # Send the solution dictionary to the server
        solution = {"task1": ["Monday 10:00", "Monday 12:00"], "task2": ["Tuesday 14:00", "Tuesday 16:00"]}
        connection.send("SendSolution", solution)

        # Wait for the server to acknowledge receipt of the solution
        input("Press any key to exit...")


if __name__ == "__main__":

    scheduleSettings = {
        "startHour": "9:00:00",
        "endHour": "18:00:00",
        "minGap": 15,
        "maxHoursPerDay": 5,
        "minTimeFrame": 15
    }

    schedule = ScheduleSettings(**scheduleSettings)

    # print(schedule.startHour)  # Output: 09:00:00
    # print(schedule.endHour)  # Output: 18:00:00
    # print(schedule.minGap)  # Output: 0:15:00
    # print(schedule.maxHoursPerDay)  # Output: 5
    # print(schedule.maxHoursPerTypePerDay)  # Output: {'A': 3, 'B': 2}
    # print(schedule.minTimeFrame)  # Output: 0:15:00

    tasks_data = [
        {"id": 1, "priority": "high", "length": 60, "deadline": datetime(2023, 4, 13, 9, 0, 0), "isRepeat": False,
         "optionalDays": ["Sunday"],
         "optionalHours": [10.50, 12.50],
         "rankListHistory": [
             {"rank": 6, "startTime": datetime(2023, 4, 3, 11, 30), "endTime": datetime(2023, 4, 3, 12, 30)},
             {"rank": 2, "startTime": datetime(2023, 4, 9, 9, 0), "endTime": datetime(2023, 4, 9, 10, 0)},
             {"rank": 6, "startTime": datetime(2023, 4, 8, 12, 0), "endTime": datetime(2023, 4, 8, 13, 0)}
         ],
         "type": "A",
         "description": "This is task 1"
         },
        {
            "id": 2,
            "priority": "medium",
            "length": 45,
            "deadline": datetime(2023, 4, 11, 11, 0, 0),
            "isRepea"
            "t": True,
            "optionalDays": ["Monday"],
            "optionalHours": [10.50, 13.00],
            "rankListHistory": [
                {"rank": 2, "startTime": datetime(2023, 4, 3, 11, 30), "endTime": datetime(2023, 4, 3, 12, 15)},
                {"rank": 1, "startTime": datetime(2023, 4, 2, 9, 0), "endTime": datetime(2023, 4, 2, 9, 45)},
                {"rank": 3, "startTime": datetime(2023, 4, 1, 9, 0), "endTime": datetime(2023, 4, 1, 9, 45)}
            ],
            "type": "B",
            "description": "This is task 2"
        },
        {
            "id": 3,
            "priority": "low",
            "length": 60,
            "deadline": datetime(2023, 4, 11, 18, 0, 0),
            "isRepeat": False,
            "optionalDays": ["Sunday","Monday"],
            "optionalHours": [10.75, 12],
            "rankListHistory": [
                {"rank": 3, "startTime": datetime(2023, 4, 1, 11, 0), "endTime": datetime(2023, 4, 1, 12, 0)},
                {"rank": 2, "startTime": datetime(2023, 4, 1, 10, 0), "endTime": datetime(2023, 4, 1, 11, 0)},
                {"rank": 1, "startTime": datetime(2023, 4, 1, 9, 0), "endTime": datetime(2023, 4, 1, 10, 0)}
            ],
            "type": "B",
            "description": "This is task 3"
        }
        # {
        #     "id": 4,
        #     "priority": "low",
        #     "length": 60,
        #     "deadline": datetime(2023, 4, 11, 18, 0, 0),
        #     "isRepeat": False,
        #     "optionalDays": ["Sunday"],
        #     "optionalHours": [12, 15],
        #     "rankListHistory": [
        #         {"rank": 3, "startTime": datetime(2023, 4, 1, 11, 0), "endTime": datetime(2023, 4, 1, 12, 0)},
        #         {"rank": 2, "startTime": datetime(2023, 4, 1, 10, 0), "endTime": datetime(2023, 4, 1, 11, 0)},
        #         {"rank": 1, "startTime": datetime(2023, 4, 1, 9, 0), "endTime": datetime(2023, 4, 1, 10, 0)}
        #     ],
        #     "type": "B",
        #     "description": "This is task 3"
        # }
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



# Define the function that will be called by the server when it receives the solution



# SimpleSatProgram()