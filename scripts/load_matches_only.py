try:
    import _bootstrap
except ModuleNotFoundError:
    from scripts import _bootstrap

from load_atp import load_matches

if __name__ == "__main__":
    load_matches()
