class VersionSpace:
    def __init__(self, tt, tf, ft, programs, mappings):
        self.tt, self.tf, self.ft = tt, tf, ft
        self.mappings = mappings
        self.programs = programs
        self.parent = None
        self.children = []
