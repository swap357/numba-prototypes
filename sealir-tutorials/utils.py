def in_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        return shell == "ZMQInteractiveShell"
    except NameError:
        return False


IN_NOTEBOOK = in_notebook()
