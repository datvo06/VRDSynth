

class VersionSpace:
    def __init__(self, tt, tf, ft, programs, mappings):
        # TT: predicted true, groundtruth true
        self.tt, self.tf, self.ft = tt, tf, ft
        self.mappings = mappings
        self.programs = programs
        self.parent = None
        self.children = []



def is_counter(vs_main, vs_dependent):
    if not vs_dependent.tt - vs_main.tt:    # The true positive of two set must be disjoint
        return False
    if len(vs_dependent.tf - vs_main.tt) == len(vs_dependent.ft): # The false positive of dependent must be a subset of main's true positive
        return False
    return True


def join_version_space_counter(vs_main, vs_dependent):
    # Towards the goal of improving precision and recall
    tt_ = vs_dependent.tt
    tf_ = vs_dependent.tf - vs_main.tt
    ft_ = vs_dependent.ft
    mappings_ = vs_dependent.mappings



def is_joinable(vs1, vs2):
    # Towards the goal of improving precision and recall
    # The two version spaces are joinable when
    # 1. one positive completely cover the other's negative, and after joining, F1 Score++
    # Towards this case one of the vs's precision will increase
    # 2. when the set of covered positive 
    pass


def join_version_space(vs1, vs2):
    # Towards the goal of improving precision and recall
    # The two version spaces are joinable when
    # 1. one positive completely cover the other's negative, and after joining, F1 Score++
    v1tf = vs1.tf
    v2tt = vs2.tt
    v2ft = vs2.ft
    excluded = v1tf - v2tt - v2ft
    # 2. when the set of covered positive 
    pass
