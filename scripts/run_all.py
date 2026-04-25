#### A script to run all participant-scenario combinations for AL 
import subprocess
import sys

ALLOWED_SCENARIOS = {
    'ID5':  [('Melon', 'Crave')],
    'ID9':  [('Melon', 'Crave')],
    'ID10': [('Carrot','Crave'), ('Nectarine','Crave'),
             ('Nectarine','Use'), ('Carrot','Use')],
    'ID11': [('Carrot','Crave'), ('Nectarine','Crave'), ('Almond','Crave'),
             ('Carrot','Use'), ('Nectarine','Use'), ('Almond','Use')],
    'ID12': [('Melon','Crave'), ('Nectarine','Crave'),
             ('Melon','Use'), ('Nectarine','Use')],
    'ID13': [('Nectarine','Use'), ('Carrot','Use'), ('Almond','Use')],
    'ID14': [('Carrot','Crave'), ('Carrot','Use')],
    'ID15': [('Carrot','Crave'), ('Carrot','Use')],
    'ID18': [('Carrot','Use'), ('Carrot','Crave')],
    'ID19': [('Melon','Crave'), ('Almond','Crave'),
             ('Melon','Use'), ('Almond','Use')],
    'ID20': [('Melon','Use'), ('Nectarine','Use'),
             ('Melon','Crave'), ('Nectarine','Crave')],
    'ID21': [('Nectarine','Use'),
             ('Melon','Crave'), ('Nectarine','Crave')],
    'ID25': [('Almond','Crave'), ('Carrot', 'Crave'), 
             ('Almond','Use')],
    'ID26': [('Carrot','Use')],
    'ID27': [('Melon','Use'), ('Nectarine','Use'),
             ('Melon','Crave'), ('Nectarine','Crave')],
    'ID28': [('Coffee','Use'), ('Almond','Use')]      
}

### Note: that        #('GHB','Use')], is taken out for ID12 since there is no other participant with GHB

POOL = "mixed"
UNLABELED_FRAC = "0.7"
DROPOUT_RATE = "0.3"
RESULTS_SUBDIR = "ALL_Participants_Results"

def main():
    for user, pairs in ALLOWED_SCENARIOS.items():
        for fruit, scenario in pairs:
            cmd = [
                sys.executable, "submit_batch.py",
                "--user", user,
                "--pool", POOL,
                "--fruit", fruit,
                "--scenario", scenario,
                "--unlabeled_frac", UNLABELED_FRAC,
                "--dropout_rate", DROPOUT_RATE,
                "--results_subdir", RESULTS_SUBDIR,
            ]
            print("Running:", " ".join(cmd))
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Skipping due to RuntimeError: {e}")

if __name__ == "__main__":
    main()
