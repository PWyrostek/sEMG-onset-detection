def onset_komi(data, h = 0.03):
    data = abs(data)
    for i in range(0, len(data)):
        if data[i] > h:
            return i
    return -1
