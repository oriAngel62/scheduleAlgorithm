# import json
#
# missions = [
#     {
#         "2023-04-03 09:00:00": [
#             {
#                 "id": 1,
#                 "name": "Complete work project",
#                 "priority": "high",
#                 "start": "2023-04-03 09:00:00",
#                 "length": 120,
#                 "deadline": "2023-04-06 17:00:00",
#                 "isRepeat": false,
#                 "isSeparable": true,
#                 "optionalDays": [],
#                 "optionalHours": [],
#                 "rankListHistory": [1, 2, 3, 4],
#                 "type": "work",
#                 "description": "Finish the final report and submit it to the manager"
#             },
#             {
#                 "id": 2,
#                 "name": "Study for exam",
#                 "priority": "medium",
#                 "start": "2023-04-03 10:00:00",
#                 "length": 120,
#                 "deadline": "2023-04-05 12:00:00",
#                 "isRepeat": false,
#                 "isSeparable": true,
#                 "optionalDays": [],
#                 "optionalHours": [],
#                 "rankListHistory": [1, 3, 2, 4],
# ]
#
# # convert to JSON format
# json_data = json.dumps(missions, indent=4)
#
# # print the JSON data
# print(json_data)