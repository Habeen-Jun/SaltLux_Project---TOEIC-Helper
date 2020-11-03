from datetime import datetime


def GetCurrentDatetime():
    now = datetime.now()
    return ('%s_%s%s_%s%s%s' % (now.year, now.month, now.day, now.hour, now.minute, now.second))
