"""Select an explicit training-only window without changing validation records."""


def training_window(entries, count, offset=0):
    if type(offset) is not int or offset < 0 or type(count) is not int or count < 1:
        raise ValueError(
            "Calibration window requires positive count and nonnegative integer offset"
        )
    train = [entry for entry in entries if entry["split"] == "train"]
    if offset + count > len(train):
        raise ValueError("Calibration window exceeds available training records")
    return train[offset : offset + count]
