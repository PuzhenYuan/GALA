import sys
from gala.checkpoint import resolve_checkpoint

if __name__ == '__main__':
    print(resolve_checkpoint(sys.argv[1]))
