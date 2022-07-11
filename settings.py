import os
import json


def create_rslt_directory(path):
    rlt_path = path + "/Results"
    if os.path.exists(rlt_path):
        print("folder exists")
    else:
        os.mkdir(rlt_path)
    print("Result directory created")
    return rlt_path


def create_rslt_json_file(json_path, dictionary):
    new_path = json_path.replace("/Results", "")
    split_path = new_path.split("/")
    json_file = split_path[-1]
    path = json_path + "/" + json_file + ".json"
    if os.path.isfile(path):
        print(path)
        with open(path, "r+") as file:
            data = json.load(file)
            data.update(dictionary)
            file.seek(0)
            json.dump(data, file, indent=4)
    else:
        with open(path, "w") as outfile:
            json.dump(dictionary, outfile, indent=4)
    return


def logfile_to_json(path):
    folder = path.split("/")
    f_name = folder[2].split("_")
    filename = (
        "Ultrasat_BSI_L_"
        + f_name[1]
        + "_"
        + f_name[2]
        + "_"
        + f_name[3]
        + "_"
        + "Result.txt"
    )
    if os.path.isfile(path + "/" + filename):
        print("Log file exist")
        dict1 = {}
        # creating dictionary
        with open(path + "/" + filename) as fh:
            for line in fh:
                # reads each line and trims of extra the spaces
                # and gives only the valid words
                command, description = line.replace(" ", "_").split(None, 1)
                dict1[command] = description.strip()
        # creating json file
        file = path + "/" + filename.replace(".txt", ".json")
        out_file = open(file, "w")
        json.dump(dict1, out_file, indent=4, sort_keys=False)
        out_file.close()
        return 1
    else:
        print("Log file doesnt exist")
    return 0


def file_struct_name(path, campaign):
    path_parts = path.split("/")
    struct_name = (
        path_parts[-1].replace("LOT", "/Ultrasat_BSI_L") + "_" + campaign + "_#"
    )
    return struct_name


def generate_int_ptc_paths(path):
    result_path = create_rslt_directory(path)
    ptc_campaign = "PTC_int_hr"
    ptc_struct = file_struct_name(path, ptc_campaign)
    light_ptc_path = path + "/" + ptc_campaign
    return path, result_path, ptc_campaign, ptc_struct, light_ptc_path


def generate_dc_paths(path):
    # for dark current measurement
    result_path = create_rslt_directory(path)
    dc_campaign = "Dark_exp_hr"
    dc_struct = file_struct_name(path, dc_campaign)
    dc_path = path + "/" + dc_campaign
    return path, result_path, dc_campaign, dc_struct, dc_path


def generate_exp_ptc_paths(path):
    # for PTC hr measurement
    result_path = create_rslt_directory(path)
    ptc_hr_campaign = "PTC_exp_hr"
    ptc_hr_struct = file_struct_name(path, ptc_hr_campaign)
    ptc_hr_path = path + "/" + ptc_hr_campaign
    return path, result_path, ptc_hr_campaign, ptc_hr_struct, ptc_hr_path


