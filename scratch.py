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

Almond_Use_users = []
Almond_Crave_users = []
Almond_Crave_users = []
Melon_Crave_users = []
Melon_Use_users = []
Carrot_Crave_users = []
Carrot_Use_users = []
Nectarine_Crave_users = []
Nectarine_Use_users = []

for u, pairs in ALLOWED_SCENARIOS.items():
    for fruit, scenario in pairs:
        if fruit == "Almond" and scenario == "Use":
            Almond_Use_users.append(u)
        if fruit == "Almond" and scenario == "Crave":
            Almond_Crave_users.append(u)
        if fruit == 'Melon' and scenario == 'Crave':
            Melon_Crave_users.append(u)
        if fruit == 'Melon' and scenario == 'Use':
            Melon_Use_users.append(u)
        if fruit == 'Carrot' and scenario == 'Crave':
            Carrot_Crave_users.append(u)
        if fruit == 'Carrot' and scenario == 'Use':
            Carrot_Use_users.append(u)
        if fruit == 'Nectarine' and scenario == 'Crave':
            Nectarine_Crave_users.append(u)
        if fruit == 'Nectarine' and scenario == 'Use':
            Nectarine_Use_users.append(u)       
            

print("ALMOND_Use_Users:", ", ".join(u.strip(' ') for u in Almond_Use_users))
print("Almond_Crave_users:", ", ".join(u.strip(' ') for u in Almond_Crave_users))
print("Melon_Crave_users:", ", ".join(u.strip(' ') for u in Melon_Crave_users))
print("Melon_Use_users:", ", ".join(u.strip(' ') for u in Melon_Use_users))
print("Carrot_Crave_users:", ", ".join(u.strip(' ') for u in Carrot_Crave_users))
print("Carrot_Use_users:",  ", ".join(u.strip(' ') for u in Carrot_Use_users))
print("Nectarine_Crave_users:", ", ".join(u.strip(' ') for u in Nectarine_Crave_users) )
print(
    "Nectarine_Use_users:",
    ", ".join(u.strip(' ') for u in Nectarine_Use_users)
)
