"""Launch sigma-combat from the repo root."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from games.sigma_combat.main import main
if __name__ == '__main__':
    main()
