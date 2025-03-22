


class BaseDefender:
    def __init__(self):
        self.name = "BaseDefender"

    def defend(self):
        raise NotImplementedError("Defender must implement defend method")

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name