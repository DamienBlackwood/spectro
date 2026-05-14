def lazy_pyplot():
    """Defer matplotlib import until a command actually plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt
