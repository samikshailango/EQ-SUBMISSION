

EASY_CASES = [
    (
        "If a train leaves at 14:30 and arrives at 18:05, how long is the journey?",
        "3 hour",
        True,
    ),
    (
        "Alice has 3 red apples and twice as many green apples as red. How many apples does she have in total?",
        "9",
        True,
    ),
    ("What is 25 plus 47?", "72", True),
    ("A store had 120 items and sold 45. How many are left?", "75", True),
    (
        "If a movie starts at 19:15 and lasts 2 hours 10 minutes, what time does it end?",
        "21:25",
        True,
    ),
    (
        "Tom has 5 dogs and 3 times as many cats as dogs. How many pets does Tom have in total?",
        "20",
        True,
    ),
    ("What is 12 multiplied by 8?", "96", True),
    (
        "A meeting needs 60 minutes. There are free slots: 09:00-09:30, 09:45-10:30, 11:00-12:00. "
        "Which slots can fit the meeting?",
        "11:00-12:00",
        True,
    ),
]

TRICKY_CASES = [
    (
        "A train leaves at 23:40 and arrives at 00:15 the next day. How long is the journey?",
        "35 minute",
        True,
    ),
    (
        "Sarah has twice as many apples as Ben, and Ben has 3 fewer apples than Carol who has 10. "
        "How many apples does Sarah have?",
        "14",
        True,
    ),
    (
        "A meeting needs 45 minutes. Free slots: 09:00-09:40, 10:00-10:50, 13:15-14:00. Which slots fit?",
        "10:00-10:50",
        True,
    ),
    (
        "A store had 50 items, sold some, and now has -5 items. What went wrong?",
        None,
        False,
    ),
    (
        "What is the difference between 100 and 37, then add 15?",
        "78",
        True,
    ),
]

ALL_CASES = EASY_CASES + TRICKY_CASES
