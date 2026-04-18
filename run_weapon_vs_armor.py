"""Launch Weapon vs. Armor from the repo root."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from games.weapon_vs_armor.main import main
if __name__ == '__main__':
    main()
