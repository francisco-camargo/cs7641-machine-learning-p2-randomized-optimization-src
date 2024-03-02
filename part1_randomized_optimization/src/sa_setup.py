import itertools

def sa_setup(temp_type, init_temp, decay, min_temp):
    # Compile the temp info into usable form
    temp_prep = list(itertools.product(*[temp_type, init_temp, decay, min_temp]))
    temperature_list = []
    for item in temp_prep:
        temp_type, init_temp, decay, min_temp = item
        temperature_list += [temp_type(init_temp, decay, min_temp)]
    return temperature_list