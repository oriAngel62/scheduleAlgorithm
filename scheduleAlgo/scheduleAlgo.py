from datetime import datetime, timedelta
from enum import Enum
from typing import List
import json
import random
from flask import Flask , request, jsonify
from dateutil import parser

app = Flask(__name__)

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
RANK_POLICY = 3
NUMOFSOLUTIONS = 1
START_AT_ZERO = 0
START_AT_ONE = 1
MIN_IN_HOUR = 60
RANDOM_LENGTH = 10
DYAS_A_WEEK = 7
DEAFAULT_RANK = 4
STARTING_HOUR = 9
DEAFAULT_MIN_TIME_FRAME = 15

COMMON_TIME_FORMAT = "%H:%M:%S"
EXTENDED_TIME_FORMAT = '%Y-%m-%d %H:%M:%S'

class Task:
    def __init__(self, id: int, priority: str, length: int,
                 deadline: datetime, isRepeat: bool, optionalDays: List[str],
                 optionalHours: List[float], rankListHistory: List[int], type: str, description: str,
                 start: datetime = None,rank = DEAFAULT_RANK):
        self.id = id
        self.priority = priority
        self.start = start if start else datetime.now().replace(hour=STARTING_HOUR, minute=START_AT_ZERO, second=START_AT_ZERO, microsecond=START_AT_ZERO)
        self.length = length
        self.deadline = deadline
        self.isRepeat = isRepeat
        self.optionalDays = optionalDays
        self.optionalHours = optionalHours                                   # two integers - start and end hours
        self.rankListHistory = rankListHistory
        self.type = type
        self.description = description
        self.rank = rank


class ScheduleSettings:
    def __init__(self, startHour: str, endHour: str, minGap: int, maxHoursPerDay: int,
                 minTimeFrame: int = DEAFAULT_MIN_TIME_FRAME):
        self.startHour = datetime.strptime(startHour, COMMON_TIME_FORMAT).time()
        self.endHour = datetime.strptime(endHour, COMMON_TIME_FORMAT).time()
        self.minGap = minGap
        self.maxHoursPerDay = maxHoursPerDay
        self.minTimeFrame = minTimeFrame
        self.slotLength = int((datetime.combine(datetime.min, datetime.strptime('00:15:00', COMMON_TIME_FORMAT).time()) - datetime.min).total_seconds() / MIN_IN_HOUR)


    def numSlots(self) -> int:
        return int((datetime.combine(datetime.now(), self.endHour) - datetime.combine(datetime.now(), self.startHour)).total_seconds() / (MIN_IN_HOUR * self.minTimeFrame))

    def numDays(self) -> int:
        return 7  # assuming weekly schedule

    def startDate(self) -> datetime:
        # assuming weekly schedule starting on Sunday
        today = datetime.today()
        return today - timedelta(days=(today.weekday() + START_AT_ONE)%7)

    def last_day_of_week(self) -> datetime:
        today = datetime.today()
        days_until_end_of_week = 6 - ((today.weekday() +1) % 7)
        last_day = today + timedelta(days=days_until_end_of_week)
        return last_day

    def next_day(self, date) -> datetime:
        tomorrow = date + timedelta(days=START_AT_ONE)
        tomorrow = datetime(tomorrow.year, tomorrow.month, tomorrow.day, hour=self.startHour.hour)
        return tomorrow

    def slotLength(self) -> int:
        return self.minTimeFrame

    def number_slots_aday(self, allslots) -> int:
        return len(allslots) / DYAS_A_WEEK


    def get_end_hour(self):
        return self.endHour
def overlaps(start1: int, end1: int, start2: int, end2: int) -> bool:
    return end1 > start2 and end2 > start1


# function to return key for any value
def get_key(val,dictionary):
    for key, value in dictionary.items():
        if val == value:
            return key
    return "key doesn't exist"

def datetime_to_slot(datetime_obj, time_slots_dict):
    day_of_week = (datetime_obj.weekday() + START_AT_ONE) % DYAS_A_WEEK  # Adjust for custom weekday start
    for slot_number, time_slot in time_slots_dict.items():
        if time_slot.hour == datetime_obj.hour and time_slot.minute == datetime_obj.minute and day_of_week == (time_slot.weekday() +START_AT_ONE) % DYAS_A_WEEK:
            return (day_of_week, slot_number)
    return datetime_obj  # Return None if no slot found for given datetime

def init_variables(time_slots_dict):
    variables = {}
    for slot_number, time_slot in time_slots_dict.items():
        day_of_week = (time_slot.weekday() + START_AT_ONE) % DYAS_A_WEEK
        variables[(day_of_week, slot_number)] = None
    return variables

def formatFromVsToAlg(oldFormat,typeformat = None):
    dt = parser.isoparse(oldFormat)
    if typeformat is None:
        output_time = dt.strftime(COMMON_TIME_FORMAT)
    else:
        output_time = dt.strftime(typeformat)
    return output_time

def toFloatList(stringList):
    newList = []
    for timeString in stringList:
        hours, minutes, seconds = map(int, timeString.split(':'))
        hour_float = hours + minutes / MIN_IN_HOUR
        newList.append(hour_float)
    return newList


def toCorrectJsonTasks(jsonFromRequest):
    tasks = []
    for mission in jsonFromRequest:
        if mission is not None:
            id = int(mission['Id'])
            priority = mission['Priority']
            length = float(mission['Length'])
            optionalDays =mission['OptionalDays']
            optionalHours = toFloatList(mission['OptionalHours'])
            deadline = formatFromVsToAlg(mission['DeadLine'], EXTENDED_TIME_FORMAT)
            type = mission['Type']
            description = mission['Description']
            rankedList = mission['RankListHistory']
            tasks.append(Task(id,priority,length,deadline,False,optionalDays,optionalHours,rankedList,type,description))
    return tasks


def setTheSettings(outsideData):
    starthour = formatFromVsToAlg(outsideData['StartHour'])
    endhour = formatFromVsToAlg(outsideData['EndHour'])
    mingap = int(outsideData['MinGap'])
    maxhours = int(outsideData['MaxHoursPerDay'])
    minframe = int(outsideData['MinTimeFrame'])
    return ScheduleSettings(starthour, endhour, mingap, maxhours, minframe)

def tasksSortedToPriorityLists(tasks):
    high_priority_task_list = []
    medium_priority_task_list = []
    low_priority_task_list = []
    for task in tasks:
        if task.priority == "High":
            high_priority_task_list.append(task)
        elif task.priority == "Medium":
            medium_priority_task_list.append(task)
        else:
            low_priority_task_list.append(task)
    return high_priority_task_list,medium_priority_task_list,low_priority_task_list

def create_slots(settings):
    start = datetime.combine(settings.startDate(), settings.startHour)
    end = datetime.combine(settings.last_day_of_week(), settings.endHour)
    time_slots = []
    time_slots_dict = {}
    slot_number = START_AT_ZERO
    while start < end:
        time_slots.append(start)
        time_slots_dict[slot_number] = start
        slot_number += START_AT_ONE
        start += timedelta(minutes=settings.minTimeFrame)
        if start.hour == settings.endHour.hour:
            start = settings.next_day(start)
    return time_slots_dict, time_slots


def calc_all_options_for_all_tasks(settings, tasks, time_slots):
    consecutive_slots = {}
    num_slots = len(time_slots)
    # Each task can be linked with only consecutive time slots according to its durability.
    for j, task in enumerate(tasks):
        consecutive_slots[task.id] = []
        # task_to_index_its_options[j] = len(consecutive_slots)
        hour_int_start = int(task.optionalHours[START_AT_ONE])  # get the integer part of the hour
        minute_int_start = int((task.optionalHours[START_AT_ONE] - hour_int_start) * MIN_IN_HOUR)  # get the minute part of the hour
        hour_int_end = int(task.optionalHours[START_AT_ZERO])  # get the integer part of the hour
        minute_int_end = int((task.optionalHours[START_AT_ZERO] - hour_int_end) * MIN_IN_HOUR)  # get the minute part of the hour
        if hour_int_start > hour_int_end or hour_int_end == hour_int_start and minute_int_start > minute_int_end:
            hour = hour_int_start
            minute = minute_int_start
            hour_int_start = hour_int_end
            minute_int_start = minute_int_end
            hour_int_end = hour
            minute_int_end = minute
        for i in range(num_slots - int(task.length / settings.minTimeFrame)):
            count_seq = START_AT_ZERO
            start_hour_minute = datetime(START_AT_ONE, START_AT_ONE, START_AT_ONE, hour_int_start, minute_int_start).time()
            time_task_may_start = datetime.combine(time_slots[i], start_hour_minute)
            end_hour_minute = datetime(START_AT_ONE, START_AT_ONE, START_AT_ONE, hour_int_end, minute_int_end).time()
            time_task_may_end = datetime.combine(time_slots[i], end_hour_minute)
            time_task_may_end = time_task_may_end + timedelta(minutes=settings.minTimeFrame)
            if not weekdays[(time_slots[i].weekday() + START_AT_ONE) % DYAS_A_WEEK] in task.optionalDays or time_slots[i] < time_task_may_start or time_slots[i] > time_task_may_end :
                continue
            deadlineDatatime = datetime.strptime(task.deadline, EXTENDED_TIME_FORMAT)
            if time_slots[i] > deadlineDatatime:
                continue
            for k in range(i, i + int(task.length / settings.minTimeFrame) + START_AT_ONE):
                # create a datetime object with today's date and the corresponding time
                if k == i or time_slots[k] != timedelta(minutes=settings.minTimeFrame) + time_slots[k - START_AT_ONE]:
                    # Start a new consecutive sequence.
                    consecutive_sequence = [time_slots[k]]
                else:
                    # Add to the current consecutive sequence and add a new sequence.
                    if k == i + int(task.length / settings.minTimeFrame):
                        consecutive_sequence.append(time_slots[k])
                        if len(consecutive_sequence) == int(task.length / settings.minTimeFrame) + START_AT_ONE:
                            every_slot_counter = START_AT_ZERO
                            for slot in consecutive_sequence:
                                if weekdays[(slot.weekday() + START_AT_ONE) % DYAS_A_WEEK] in task.optionalDays and time_task_may_start <= slot <= time_task_may_end:
                                    every_slot_counter += START_AT_ONE
                            if every_slot_counter == int(task.length / settings.minTimeFrame) + START_AT_ONE:
                                # Check if the sequence ends before the deadline.
                                consecutive_slots[task.id].append((consecutive_sequence, (slot.weekday() + START_AT_ONE) % DYAS_A_WEEK, i, k))
                                count_seq += START_AT_ONE
                        consecutive_sequence = []
                    else:
                        consecutive_sequence.append(time_slots[k])
    return consecutive_slots


def find_all_problematic_tasks(consecutive_slots, tasks):
    unschedualed_tasks = []
    no_options_tasks_ids = [idTask for idTask, value in consecutive_slots.items() if value == []]
    for idTask in no_options_tasks_ids:
        del consecutive_slots[idTask]
        for taskProblem in tasks:
            if taskProblem.id in no_options_tasks_ids:
                tasks.remove(taskProblem)
        unschedualed_tasks.append(idTask)
    return unschedualed_tasks, tasks


def get_salt_int():
    salt_chars = "1234567890"
    salt = ""
    while len(salt) <= RANDOM_LENGTH:  # length of the random string.
        index = random.randint(START_AT_ZERO, len(salt_chars) - START_AT_ONE)
        salt += salt_chars[index]
    return int(salt)


def sort_and_shuffle(tasks,consecutive_slots,time_slots_dict):
    high_priority_task_list, medium_priority_task_list, low_priority_task_list = tasksSortedToPriorityLists(tasks)
    sort_by_least_options(high_priority_task_list, consecutive_slots)
    sort_by_least_options(medium_priority_task_list, consecutive_slots)
    sort_by_least_options(low_priority_task_list, consecutive_slots)
    all_tasks_sorted_together = high_priority_task_list + medium_priority_task_list + low_priority_task_list
    random.seed(get_salt_int())  # solution index comes from the backend
    for task in all_tasks_sorted_together:
        random.shuffle(consecutive_slots[task.id])
    switch_consecutive_slots_sequence_according_to_rank(all_tasks_sorted_together, consecutive_slots, time_slots_dict)
    return all_tasks_sorted_together , consecutive_slots


def to_solution_format(settings, result, unschedualed_tasks):
    flipped = {}
    task_day_start_end = {}
    if result:
        for key, value in result.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                flipped[value].append(key)
        for key, value in flipped.items():
            start = settings.endHour.strftime(COMMON_TIME_FORMAT)
            datetime_start = None
            end = settings.startHour.strftime(COMMON_TIME_FORMAT)
            datetime_end = None
            for i, slot in enumerate(value):
                slot_time = slot.strftime(COMMON_TIME_FORMAT)
                if slot_time < start:
                    datetime_start = slot
                    start = slot_time
                if slot_time > end and i < len(value):
                    datetime_end = slot
            task_day_start_end[key] = (datetime_start, datetime_end)
        return task_day_start_end, unschedualed_tasks
    return [], []


def solution_to_json(solutions):
    json_array = []
    flag = START_AT_ZERO
    for solution_id, solution_data in solutions.items():
        time_slots_dict, unscheduled_tasks = solution_data
        if flag == START_AT_ZERO:
            json_array.append(str(len(unscheduled_tasks)))
            flag = START_AT_ONE
        json_array.append(str(solution_id))
        if not time_slots_dict:
            json_array.append(str(solution_id))  # if no solution available
        else:
            for mission_id, time_slots in time_slots_dict.items():
                start_slot, last_slot = time_slots
                json_array.append(str(mission_id))
                json_array.append(start_slot.strftime(EXTENDED_TIME_FORMAT))
                json_array.append(last_slot.strftime(EXTENDED_TIME_FORMAT))
    return json_array


@app.route('/algoComplete', methods=['POST'])
def generate_schedule():
    data = request.get_json()
    settingsFromJson = data['ScheduleSetting']
    tasksFromJson = data['AlgoMission']
    settings = setTheSettings(settingsFromJson)
    tasks = toCorrectJsonTasks(tasksFromJson)
    # Create time slots for the week
    time_slots_dict, time_slots = create_slots(settings)
    consecutive_slots = calc_all_options_for_all_tasks(settings, tasks, time_slots)
    unschedualed_tasks, tasks = find_all_problematic_tasks(consecutive_slots, tasks)
    solutions = {}
    for solution_index in range(START_AT_ZERO,NUMOFSOLUTIONS):
        tasks, consecutive_slots = sort_and_shuffle(tasks, consecutive_slots,time_slots_dict)
        # Try to schedule tasks one at a time
        schedule = {}
        variables = init_variables(time_slots_dict)
        result, unscheduled_tasks1 = backtrack(schedule, tasks, consecutive_slots, settings, variables,time_slots_dict, START_AT_ONE,len(tasks),time_slots_dict)
        unschedualed_tasks.extend(unscheduled_tasks1)
        if len(unschedualed_tasks) != START_AT_ZERO:
            print(f"cannot schedule tasks - {unschedualed_tasks}")
        solutions[solution_index] = to_solution_format(settings, result, unschedualed_tasks)
        # Return the result as JSON
    json_array = solution_to_json(solutions)
    return json.dumps(json_array)


def sort_by_least_options(task_list, all_blocks):
    task_list.sort(key=lambda task: len(all_blocks[task.id]))
    sublists = {}
    prev = START_AT_ZERO
    i = START_AT_ZERO
    same_num_of_options_list = []
    for task in task_list:
        if len(all_blocks[task.id]) != prev:
            if prev != START_AT_ZERO:
                sublists[i] = same_num_of_options_list
                i += 1
                same_num_of_options_list = []
            prev = len(all_blocks[task.id])
        same_num_of_options_list.append(task)
    if same_num_of_options_list:
        sublists[i] = same_num_of_options_list
    # Shuffle each sublist.
    for sublist in sublists.values():
        random.shuffle(sublist)
    # Merge the shuffled sublists into a single list.
    all_tasks_sorted_together = []
    for sublist in sublists.values():
        all_tasks_sorted_together.extend(sublist)
    return all_tasks_sorted_together


def checkLimitHourADay(variables, settings, day, task_length):
    summ = 0
    slotsADay = int(settings.number_slots_aday(variables))
    for i in range(slotsADay * day, slotsADay * (day + START_AT_ONE)):
        if variables[(day, i)] is not None:
            summ += START_AT_ONE
    if summ <= settings.maxHoursPerDay * (MIN_IN_HOUR / settings.minTimeFrame) - START_AT_ONE - task_length / settings.minTimeFrame:
        return True
    return False


def check_for_another_option_for_task(previousTaskId, consecutive_slots, variables, desired_slots_indexes):
    for block in consecutive_slots[previousTaskId]:
        block_slots, day, start_slot_index, end_slot_index = block
        red_flag = START_AT_ZERO
        for i in range(start_slot_index,end_slot_index + START_AT_ONE):
            if (variables[(day, i)] is not None and variables[(day, i)] != previousTaskId) or (i in desired_slots_indexes):
                red_flag = START_AT_ONE
        if red_flag == START_AT_ZERO:
            return True
    return False



def check_for_alternative(tasks, unscheduledId, consecutive_slots, variables,schedule,time_slots_dict):
    allPreviuosTaskId = set(variables.values())
    allPreviuosTaskId.discard(None)
    conflicted_tasks = []
    optionalTasksWithForbidenSlots = {}
    for previousTaskId in allPreviuosTaskId:
        slots = [slotString for slotString, value in schedule.items() if value == previousTaskId]
        for slot in slots:
            for block in consecutive_slots[unscheduledId]:
                bad_block = False
                block_slots, day, start_slot_index, end_slot_index = block
                for i in range(start_slot_index,end_slot_index + START_AT_ONE):
                    # if there is other task on this section then continue looping to next block
                    if variables[(day, i)] != previousTaskId and variables[(day, i)] is not None:
                        bad_block = True
                        continue
                if bad_block:
                    continue
                if time_slots_dict[start_slot_index] <= slot <= time_slots_dict[end_slot_index]:
                    conflicted_tasks.append(previousTaskId)
                    for task in tasks:
                        if task.id == unscheduledId:
                            unscheduledTask = task
                        if task.id == previousTaskId:
                            prevTask = task
                    desired_slots_indexes = list(range(start_slot_index, end_slot_index + START_AT_ONE))
                    if check_for_another_option_for_task(previousTaskId,consecutive_slots,variables,desired_slots_indexes) is False or (unscheduledTask.priority == "Medium" and prevTask.priority == "High"):
                        conflicted_tasks.remove(previousTaskId)
                    else:
                        optionalTasksWithForbidenSlots[previousTaskId] = desired_slots_indexes
    conflicted_tasks = list(set(conflicted_tasks))
    problamatic_tasks = []
    for task in tasks:
        if task.id in conflicted_tasks:
            problamatic_tasks.append(task)
    return problamatic_tasks, optionalTasksWithForbidenSlots

def findBestOption(tasks_list_with_options ,variables, consecutive_slots,previous_tasks_id_with_options_with_consecutive):
    improved = []
    notImproved = []
    for task_with_option in tasks_list_with_options:
        for block in consecutive_slots[task_with_option.id]:
            block_slots, day, start_slot_index, end_slot_index = block
            bad_block = False
            for i in range(start_slot_index,end_slot_index+START_AT_ONE):
                forbiden_list = previous_tasks_id_with_options_with_consecutive[task_with_option.id]
                if variables[(day,i)] is not None and variables[(day,i)] != task_with_option.id or i in forbiden_list:
                    bad_block = True
                    continue
            if bad_block:
                continue
            went_in = False
            if task_with_option.rankListHistory:
                for ranking in task_with_option.rankListHistory:
                    if ranking.startTime == start_slot_index:
                        if ranking.rank > task_with_option.rank:
                            improved.append((task_with_option,start_slot_index))
                            went_in = True
                        else:
                            if ranking.rank < task_with_option.rank and ranking.rank <= RANK_POLICY:
                                went_in = True
                                pass
                            else:
                                notImproved.append((task_with_option,start_slot_index))
                                went_in = True
            if not went_in:
                sofarTasks = []
                for duo in notImproved:
                    sofarTasks.append(duo[START_AT_ZERO])
                if task_with_option not in sofarTasks:
                    notImproved.append((task_with_option,start_slot_index))
    #improved.sort(key=lambda item: item.rank)
    #notImproved.sort(key=lambda task: task.rank)
    if improved:
        return improved[START_AT_ZERO][START_AT_ZERO],improved[START_AT_ZERO][START_AT_ONE]
    elif notImproved:
        return notImproved[START_AT_ZERO][START_AT_ZERO], notImproved[START_AT_ZERO][START_AT_ONE]
    else:
        return [],[]

def switchSlots(settings, taskToReplace, startingSlotToMove,schedule,variables,time_slots_dict):
    old_slots = []
    for slot, missionId in schedule.items():
        if taskToReplace.id == missionId:
            old_slots.append(slot)
    slots_to_delete = []
    for datetimeSlot, id in schedule.items():
        if taskToReplace.id == id:
            slots_to_delete.append(datetimeSlot)
    for slot in slots_to_delete:
        del schedule[slot]
    for i,variable in enumerate(variables):
        day = int(i / settings.number_slots_aday(time_slots_dict))
        if variables[(day,i)] == taskToReplace.id:
            variables[(day, i)] = None
    for i in range(int(taskToReplace.length / settings.minTimeFrame) + START_AT_ONE):
        slot = time_slots_dict[startingSlotToMove]
        schedule[slot] = taskToReplace.id
        indexOfSlot = datetime_to_slot(slot,time_slots_dict)
        day = int(indexOfSlot[START_AT_ONE] / settings.number_slots_aday(time_slots_dict))
        variables[(day, indexOfSlot[START_AT_ONE] )] = taskToReplace.id
        startingSlotToMove += START_AT_ONE
    return schedule , variables

def skipAndDeleteTask(tasks,unscheduledId,consecutive_slots, current_task_index):
    del consecutive_slots[unscheduledId]
    tasks.remove(tasks[current_task_index - START_AT_ONE])

def backtrack(schedule, tasks, consecutive_slots, settings, variables,time_slots_dict, current_task_index,originalNumTasks,time_to_slots_dict):
    # If all tasks have been scheduled, return the solution.
    if current_task_index > len(tasks):
        return schedule, []
    current_task = tasks[current_task_index - START_AT_ONE]
    unscheduled_tasks = []
    scheduled = False
    for block in consecutive_slots[current_task.id]:
        slots, day, start_slot_index, end_slot_index = block
        for slot_index in range(start_slot_index, end_slot_index - len(slots) + int(settings.minTimeFrame / settings.minGap) + START_AT_ONE):
            # Check if the current slot is already scheduled.
            conflict = False
            count_slots = START_AT_ZERO
            # max hour per day check
            if checkLimitHourADay(variables, settings, day,current_task.length):
                for i, slot in enumerate(slots):
                    if variables[(day, slot_index + i)] is not None:
                        conflict = True
                        break
                    count_slots += START_AT_ONE
            if conflict:
                continue
            if count_slots == len(slots):
                # If the current slot is not scheduled, schedule the task on these slots.
                for i, slot in enumerate(slots):
                    variables[(day, start_slot_index + i)] = current_task.id
                    schedule[slot] = current_task.id
                    for rank in current_task.rankListHistory:
                        if rank.startTime <= slot <= rank.endTime:
                            current_task.rank = rank.rank
                scheduled = True
            else:
                continue
        if count_slots != len(slots):
                continue
        # Recursively try to schedule the next task.
        next_task_schedule, next_unscheduled_tasks = backtrack(schedule.copy(), tasks, consecutive_slots, settings,
                                                                   variables,time_slots_dict, current_task_index + START_AT_ONE,originalNumTasks,time_to_slots_dict)

        if next_task_schedule == "back":
            for i, slot in enumerate(slots):
                variables[(day, start_slot_index + i)] = None
                del schedule[slot]
            continue
        # If the next task has been scheduled, return the solution. if there was a problem return empty lists.
        if next_task_schedule == [] and next_unscheduled_tasks == []:
            return [], []
        if len(next_unscheduled_tasks) + len(set(next_task_schedule.values())) == originalNumTasks:
            return next_task_schedule, next_unscheduled_tasks
    # If the current task could not be scheduled in any of its consecutive blocks, add it to the unscheduled tasks list.
    if not scheduled:
        unscheduledId = current_task.id
        unscheduled_tasks.append(unscheduledId)
        unscheduled_task_priority = current_task.priority
        successfullSwitch = False
        if unscheduled_task_priority == "High":
            return "back", []
        elif unscheduled_task_priority == "Medium":
            previous_tasks_id_with_options, previous_tasks_with_options = check_for_alternative(tasks, unscheduledId, consecutive_slots, variables,schedule,time_to_slots_dict)
            high_priority_task_list, medium_priority_task_list, low_priority_task_list = tasksSortedToPriorityLists(previous_tasks_id_with_options)
            all_tasks_list = high_priority_task_list+medium_priority_task_list+low_priority_task_list
            taskToReplace,startingSlotToMove = findBestOption(all_tasks_list,variables,consecutive_slots,previous_tasks_with_options)
            if taskToReplace == []:
                skipAndDeleteTask(tasks,unscheduledId,consecutive_slots, current_task_index)
            else:
                schedule , variables = switchSlots(settings,taskToReplace,startingSlotToMove,schedule,variables,time_slots_dict)
                successfullSwitch = True
        else:
            skipAndDeleteTask(tasks, unscheduledId, consecutive_slots,current_task_index)
        next_task_schedule, next_unscheduled_tasks = backtrack(schedule.copy(), tasks, consecutive_slots,
                                                               settings,
                                                               variables, time_slots_dict, current_task_index,
                                                               originalNumTasks, time_to_slots_dict)
        if successfullSwitch is False:
            next_unscheduled_tasks.append(unscheduledId)
        return next_task_schedule, next_unscheduled_tasks
    return [], []

def sort_by_rank_and_start_time(rank_list_history):
    return sorted(rank_list_history, key=lambda x: (x['Rank'], x['StartTime'] ), reverse=True)

def switch_consecutive_slots_sequence_according_to_rank(tasks_data, consecutive_slots, time_slots_dict):
    for task in tasks_data:
        sorted_task = sort_by_rank_and_start_time(task.rankListHistory)
        for i, consecutive_sequence in enumerate(consecutive_slots[task.id]):
            for rank_data in sorted_task:
                rank = rank_data['Rank']
                start_time_string = rank_data['StartTime']
                start_time_string = start_time_string.replace("T", " ")
                end_time_string = rank_data['EndTime']
                end_time_string = end_time_string.replace("T", " ")
                datetimeStart= datetime.strptime(start_time_string, EXTENDED_TIME_FORMAT)
                datetimeEnd = datetime.strptime(end_time_string, EXTENDED_TIME_FORMAT)
                #convert time to tuple
                tuple_start = datetime_to_slot(datetimeStart, time_slots_dict)
                tuple_end = datetime_to_slot(datetimeEnd, time_slots_dict)
                consecutive_sequence_start = datetime_to_slot(consecutive_sequence[0][0], time_slots_dict)
                consecutive_sequence_end = datetime_to_slot(consecutive_sequence[0][-1], time_slots_dict)
                if consecutive_sequence_start == tuple_start and consecutive_sequence_end == tuple_end and rank > RANK_POLICY:
                    consecutive_slots[task.id].pop(i)
                    consecutive_slots[task.id].insert(0, consecutive_sequence)
                    break



if __name__ == "__main__":
    app.run(host='localhost', port=5000)


