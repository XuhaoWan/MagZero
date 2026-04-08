def process_params_file(input_file, output_file, element_list):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Step 1: Extract header
    header = []
    iparams_blocks = []
    current_block = []
    in_iparams = False

    for line in lines:
        if line.startswith("# Impurity problem number"):
            if current_block:
                iparams_blocks.append(current_block)
            current_block = [line]
            in_iparams = True
        elif in_iparams:
            current_block.append(line)
        else:
            header.append(line)

    # Append the last block
    if current_block:
        iparams_blocks.append(current_block)
    
    # Step 2: Parse iparams blocks
    iparams_dicts = {}
    for block in iparams_blocks:
        element = None
        params = {}
        for line in block:
            if "Coulomb repulsion (F0) for" in line:
                # Extract the element name
                element = line.split("for")[-1].strip().split()[0].strip("',\"]")
            elif ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().strip('"')
                value, comment = value.split(",", 1)
                
                # Remove extra brackets from values
                value = value.strip().strip("[]").strip()
                comment = comment.strip().strip()
                print(value)
                print(comment)
                params[key] = (value, comment)
        
        if element:
            iparams_dicts[element] = iparams_dicts.get(element, []) + [params]
    
    # Step 3: Generate new iparams based on element list
    new_iparams = []
    for idx, element in enumerate(element_list):
        if element in iparams_dicts:
            # Cycle through available iparams for the element
            params = iparams_dicts[element][idx % len(iparams_dicts[element])]
            new_dict_name = f"iparams{idx}"
            new_iparams.append((new_dict_name, params))
    
    # Step 4: Write to new file
    with open(output_file, 'w') as f:
        # Write header
        f.writelines(header)
        
        # Write iparams
        for dict_name, params in new_iparams:
            f.write(f"\n# {dict_name}\n")
            f.write(f"{dict_name}={{\n")
            for key, (value, comment) in params.items():
                # Add quotes only around strings, avoid double brackets
                if not value.replace('.', '', 1).isdigit():
                    value = f'"{value}"'
                f.write(f'     "{key}"               : [{value:<15}, {comment}]\n')
            f.write("}\n")

    print(f"Processed file written to {output_file} successfully.")

element_list = ["Fe", "Fe", "Fe"]  # 示例元素列表
process_params_file("params.dat", "new_params.dat", element_list)

