def time_to_minutes(time):
    """Convert HH:MM into minutes."""

    hour, minute = map(int, time.split(":"))
    return hour * 60 + minute


def check_conflict(block_start, block_end, train_times):
    """
    Check whether any train passes during the maintenance block.
    """
    start = time_to_minutes(block_start)
    end = time_to_minutes(block_end)
    conflicts = []

    for train_time in train_times:
        train = time_to_minutes(train_time)
        if start <= train <= end:
            conflicts.append(train_time)
    return conflicts