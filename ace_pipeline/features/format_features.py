def format_features(in_file, out_file, num_atoms=4000):
    with open(in_file, 'r') as f:
        lines = f.readlines()
    
    descriptors = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or line.startswith('c_pace') or line[0].isdigit() is False:
            continue
        parts = line.split()
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        atom_id = parts[0]
        ace_values = list(map(float, parts[1:]))
        descriptors.append(ace_values)
        if len(descriptors) >= num_atoms:
            break
    
    with open(out_file, 'w') as f:
        for row in descriptors:
            f.write(' '.join(map(str, row)) + '\n')

    
format_features("features/ace_fcc.txt", "data/X_fcc.dat")
format_features("features/ace_bcc.txt", "data/X_bcc.dat")
format_features("features/ace_hcp.txt", "data/X_hcp.dat")
format_features("features/ace_liquid.txt", "data/X_liquid.dat")

