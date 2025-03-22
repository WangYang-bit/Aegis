def compute_index_in_scenario(sence_id,len_record):
    for i, ele in enumerate(len_record):
        if sence_id < ele:
            scenario_index = i
            break
    if scenario_index < 1:
        return sence_id, 0
    else:
        return sence_id - len_record[scenario_index - 1], scenario_index
    