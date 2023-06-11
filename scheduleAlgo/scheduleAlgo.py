from typing import List
from datetime import datetime, timedelta
from enum import Enum
from ortools.sat.python import cp_model
from typing import List
import json
from math import prod
# from signalrcore import Connection
import random
import requests
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
RANK_POLICY = 1
NUMOFSOLUTIONS = 1

COMMON_TIME_FORMAT = "%H:%M:%S"
EXTENDED_TIME_FORMAT = '%Y-%m-%d %H:%M:%S'
class Task:
    def __init__(self, id: int, priority: str, length: int,
                 deadline: datetime, isRepeat: bool, optionalDays: List[str],
                 optionalHours: List[float], rankListHistory: List[int], type: str, description: str,
                 start: datetime = None,rank =4):
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
        self.rank = rank


class ScheduleSettings:
    def __init__(self, startHour: str, endHour: str, minGap: int, maxHoursPerDay: int,
                 minTimeFrame: int = 15):
        self.startHour = datetime.strptime(startHour, COMMON_TIME_FORMAT).time()
        self.endHour = datetime.strptime(endHour, COMMON_TIME_FORMAT).time()
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
        hour_float = hours + minutes / 60
        newList.append(hour_float)
    return newList


def toCorrectJsonTasks(jsonFromRequest):
    tasks = []
    # data = json.loads(jsonFromRequest)
    for mission in jsonFromRequest:
        if mission is not None:
            id = int(mission['Id'])
            priority = mission['Priority']
            length = float(mission['Length'])
            optionalDays =mission['OptionalDays']
            optionalHours = toFloatList(mission['OptionalHours'])
            deadline = formatFromVsToAlg(mission['DeadLine'],"%Y-%m-%d %H:%M:%S")   #need the whole format
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
    return ScheduleSettings(starthour,endhour,mingap,maxhours,minframe)

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


@app.route('/algoComplete', methods=['POST'])
def generate_schedule():
    data = request.get_json()
    settingsFromJson = data['ScheduleSetting']
    tasksFromJson = data['AlgoMission']
    settings = setTheSettings(settingsFromJson)

    tasks = toCorrectJsonTasks(tasksFromJson)
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
        hour_int_start = int(task.optionalHours[1])  # get the integer part of the hour
        minute_int_start = int(
            (task.optionalHours[1] - hour_int_start) * 60)  # get the minute part of the hour
        hour_int_end = int(task.optionalHours[0])  # get the integer part of the hour
        minute_int_end = int(
            (task.optionalHours[0] - hour_int_end) * 60)  # get the minute part of the hour
        if hour_int_start > hour_int_end or hour_int_end == hour_int_start and minute_int_start > minute_int_end:
            hour = hour_int_start
            minute = minute_int_start
            hour_int_start = hour_int_end
            minute_int_start = minute_int_end
            hour_int_end = hour
            minute_int_end = minute

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
            deadlineDatatime = datetime.strptime(task.deadline, date_format)
            if time_slots[i] > deadlineDatatime:
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
    unschedualed_tasks = []
    no_options_tasks_ids = [idTask for idTask, value in consecutive_slots.items() if value == []]
    for idTask in no_options_tasks_ids:
        del consecutive_slots[idTask]
        for taskProblem in tasks:
            if taskProblem.id in no_options_tasks_ids:
                tasks.remove(taskProblem)
        unschedualed_tasks.append(idTask)

    high_priority_task_list, medium_priority_task_list, low_priority_task_list = tasksSortedToPriorityLists(tasks)


    solutions = {}
    for solution_index in range(0,NUMOFSOLUTIONS):
        sort_by_least_options(high_priority_task_list, consecutive_slots)
        sort_by_least_options(medium_priority_task_list, consecutive_slots)
        sort_by_least_options(low_priority_task_list, consecutive_slots)
        all_tasks_sorted_together = high_priority_task_list + medium_priority_task_list + low_priority_task_list

        random.seed(solution_index)              #solution index comes from the backend
        for task in all_tasks_sorted_together:
            random.shuffle(consecutive_slots[task.id])
        switch_consecutive_slots_sequence_according_to_rank(tasks, consecutive_slots, time_slots_dict)
        # Try to schedule tasks one at a time
        schedule = {}
        variables = init_variables(time_slots_dict)
        result, unscheduled_tasks1 = backtrack(schedule, all_tasks_sorted_together, consecutive_slots, settings, variables,time_slots_dict, 1,len(tasks),time_slots_dict)
        unschedualed_tasks.extend(unscheduled_tasks1)
        if len(unschedualed_tasks) != 0:
            print(f"cannot schedule tasks - {unschedualed_tasks}")

        flipped = {}
        task_day_start_end = {}
        if result != []:
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

            solutions[solution_index] = task_day_start_end, unschedualed_tasks
        # Return the result as JSON
    json_array = []
    flag = 0
    for solution_id, solution_data in solutions.items():
        time_slots_dict, unscheduled_tasks = solution_data
        if flag == 0:
            json_array.append(str(len(unscheduled_tasks)))
            flag = 1
        json_array.append(str(solution_id))

        # if unscheduled_tasks == []:
        #     json_array.append(unscheduled_tasks)
        # else:
        #     for taskId in unscheduled_tasks:
        #         json_array.append(str(taskId))
        for mission_id, time_slots in time_slots_dict.items():
            start_slot, last_slot = time_slots
            json_array.append(str(mission_id))
            json_array.append(start_slot.strftime(EXTENDED_TIME_FORMAT))
            json_array.append(last_slot.strftime(EXTENDED_TIME_FORMAT))

    return json.dumps(json_array)


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

def checkLimitHourADay(variables, settings, day,task_length):
    summ = 0
    slotsADay = int(settings.number_slots_aday(variables))
    for i in range(slotsADay * day, slotsADay * (day + 1)):
        if variables[(day, i)] is not None:
            summ += 1
    if summ <= settings.maxHoursPerDay * 4 - 1 - task_length / 15:
        return True
    return False

def check_for_another_option(previousTaskId,consecutive_slots,variables,desired_slots_indexes,index_to_slot_dict):
    for block in consecutive_slots[previousTaskId]:
        block_slots, day, start_slot_index, end_slot_index = block
        red_flag= 0
        for i in range(start_slot_index,end_slot_index+1):
            if (variables[(day,i)] is not None and variables[(day,i)] != previousTaskId) or (i in desired_slots_indexes):
                red_flag =1
        if red_flag == 0:
            return True
    return False



def check_for_alternative(tasks, unscheduledId, consecutive_slots, variables,schedule,time_slots_dict):
    allPreviuosTaskId = set(variables.values())
    allPreviuosTaskId.discard(None)
    conflicted_tasks = []
    for previousTaskId in allPreviuosTaskId:
        slots = [slotString for slotString, value in schedule.items() if value == previousTaskId]
        for slot in slots:
            for block in consecutive_slots[unscheduledId]:
                block_slots, day, start_slot_index, end_slot_index = block
                for i in range(start_slot_index,end_slot_index+1):
                    if variables[(day,i)] != previousTaskId and variables[(day,i)] is not None:   # if there is other task on this section then i dont care and continue looping
                        break
                if time_slots_dict[start_slot_index] <= slot and time_slots_dict[end_slot_index] >= slot:
                    conflicted_tasks.append(previousTaskId)
                    for task in tasks:
                        if task.id == unscheduledId:
                            unscheduledTask = task
                        if task.id == previousTaskId:
                            prevTask = task
                    desired_slots_indexes = list(range(start_slot_index, end_slot_index+1))
                    if check_for_another_option(previousTaskId,consecutive_slots,variables,desired_slots_indexes,time_slots_dict) is False or (unscheduledTask.priority == "Medium" and prevTask.priority == "High"):
                        conflicted_tasks.remove(previousTaskId)


    optionalTasks = [tasks[taskId] for taskId in conflicted_tasks]
    return optionalTasks

def findBestOption(task,tasks_list_with_options ,variables, settings,consecutive_slots):
    improved = []
    notImproved = []
    for task_with_option in tasks_list_with_options:
        for block in consecutive_slots[task_with_option.id]:
            block_slots, day, start_slot_index, end_slot_index = block
            for i in range(start_slot_index,end_slot_index+1):
                if variables[(day,i)] is not None:
                    break

            if task_with_option.rankListHistory is not []:
                for ranking in task_with_option.rankListHistory:
                    if ranking.startTime == start_slot_index:
                        if ranking.rank > task_with_option.rank:
                            improved.append((task_with_option,start_slot_index))
                        else:
                            if ranking.rank < task_with_option.rank and ranking.rank <= 3:
                                pass
                            else:
                                notImproved.append((task_with_option,start_slot_index))
    improved = sorted(improved, key=lambda task: task.rank)
    notImproved = sorted(notImproved, key=lambda task: task.rank)
    if len(improved) != 0:
        return improved[0][0],improved[0][1]
    elif len(notImproved) != 0:
        return notImproved[0][0], notImproved[0][1]
    else:
        return [],[]

def switchSlots(settings, taskToReplace, startingSlotToMove,schedule,variables,time_slots_dict):
    old_slots = []
    for slot, missionId in schedule.items():
        if taskToReplace.id == missionId:
            old_slots.append(slot)
    totalNumOfNewSlots = len(old_slots)
    slots_to_delete = []

    for datetimeSlot, id in schedule.items():
        if taskToReplace.id == id:
            slots_to_delete.append(datetimeSlot)

    for slot in slots_to_delete:
        del schedule[slot]

    for i,variable in enumerate(variables):
        if variable[(i / settings.number_slots_aday(time_slots_dict),i)] == taskToReplace.id:
            variable[(i / settings.number_slots_aday(time_slots_dict), i)] = None

    for i in range((taskToReplace.length / 15) + 1):
        slot = time_slots_dict[startingSlotToMove]
        schedule[slot] = taskToReplace.id
        indexOfSlot = datetime_to_slot(slot,time_slots_dict)
        variables[(indexOfSlot / settings.number_slots_aday(time_slots_dict), indexOfSlot)] = taskToReplace.id

def skipAndDeleteTask(tasks,unscheduledId,consecutive_slots, current_task_index):
    del consecutive_slots[unscheduledId]
    tasks.remove(tasks[current_task_index - 1])











def backtrack(schedule, tasks, consecutive_slots, settings, variables,time_slots_dict, current_task_index,originalNumTasks,time_to_slots_dict):
    # If all tasks have been scheduled, return the solution.
    if current_task_index > len(tasks):
        return schedule, []

    unscheduled_tasks = []
    scheduled = False
    for block in consecutive_slots[tasks[current_task_index - 1].id]:
        slots, day, start_slot_index, end_slot_index = block
        for slot_index in range(start_slot_index, end_slot_index - len(slots) + 2):
            # Check if the current slot is already scheduled.
            conflict = False
            count_slots =0
            if checkLimitHourADay(variables, settings, day,tasks[current_task_index - 1].length):

                for i, slot in enumerate(slots):
                    if variables[(day, slot_index + i)] is not None:
                        conflict = True
                        break
                    count_slots += 1

            if conflict:
                continue

            # max hour per day check
            if count_slots == len(slots) :
                # If the current slot is not scheduled, schedule the task on these slots.
                for i, slot in enumerate(slots):
                    variables[(day, start_slot_index + i)] = tasks[current_task_index - 1].id
                    schedule[slot] = tasks[current_task_index - 1].id
                    for rank in tasks[current_task_index - 1].rankListHistory:
                        if slot >= rank.startTime and slot <= rank.endTime:
                            tasks[current_task_index - 1].rank = rank.rank
                scheduled = True
            else:
                continue

        if count_slots != len(slots):
                continue
        # Recursively try to schedule the next task.
        next_task_schedule, next_unscheduled_tasks = backtrack(schedule.copy(), tasks, consecutive_slots, settings,
                                                                   variables,time_slots_dict, current_task_index + 1,originalNumTasks,time_to_slots_dict)

        # If the next task has been scheduled, return the solution.
        if next_task_schedule == [] and next_unscheduled_tasks == []:
            return [] , []
        if len(next_unscheduled_tasks) + len(set(next_task_schedule.values())) == originalNumTasks:
            return next_task_schedule, next_unscheduled_tasks

        # If the next task could not be scheduled, add the current task to the unscheduled tasks list and undo the current task's schedule.
        # for i, slot in enumerate(slots):
        #     variables[(day, start_slot_index + i)] = None
        #     del schedule[slot]

    # If the current task could not be scheduled in any of its consecutive blocks, add it to the unscheduled tasks list.
    if not scheduled:
        unscheduled_tasks.append(tasks[current_task_index - 1])
        unscheduledId = tasks[current_task_index - 1].id
        unscheduled_task_priority = unscheduled_tasks[-1].priority
        successfullSwitch = False
        if unscheduled_task_priority == "High":
            previous_tasks_with_options = check_for_alternative(tasks, unscheduledId, consecutive_slots, variables,schedule,time_to_slots_dict)
            high_priority_task_list, medium_priority_task_list, low_priority_task_list = tasksSortedToPriorityLists(previous_tasks_with_options)
            taskToReplace, startingSlotToMove = findBestOption(tasks[current_task_index - 1],
                                                               high_priority_task_list ,
                                                               variables, settings,consecutive_slots)
            if taskToReplace == []:
                return [] , []
            else:
                switchSlots(settings, taskToReplace, startingSlotToMove,schedule,variables,time_slots_dict)
                successfullSwitch = True

        elif unscheduled_task_priority == "Medium":
            previous_tasks_with_options = check_for_alternative(tasks,unscheduledId,consecutive_slots,variables,schedule,time_to_slots_dict)
            high_priority_task_list, medium_priority_task_list, low_priority_task_list = tasksSortedToPriorityLists(previous_tasks_with_options)
            all_tasks_list = high_priority_task_list+medium_priority_task_list+low_priority_task_list
            taskToReplace,startingSlotToMove = findBestOption(unscheduled_tasks[-1],all_tasks_list,variables,settings,consecutive_slots)
            if taskToReplace == []:
                skipAndDeleteTask(tasks,unscheduledId,consecutive_slots, current_task_index)
            else:
                switchSlots(settings,taskToReplace,startingSlotToMove,schedule,variables,time_slots_dict)
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
                assigned_slots = []
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


