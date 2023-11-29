def get_p_r_f1(tt, tf, ft):
    return len(tt) / (len(tt) + len(tf)), len(tt) / (len(tt) + len(ft)), 2 * len(tt) / (2 * len(tt) + len(tf) + len(ft))
