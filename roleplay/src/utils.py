# coding=utf-8
import csv
import json
import os


def read_csv_to_json(file_path, role_name, role_info):
    json_list = []
    
    with open(file_path, mode="r", newline="", encoding="utf-8") as csvfile:
        csv_reader = csv.reader(csvfile)
        _ = next(csv_reader)
        
        for row in csv_reader:
            json_object = {
                "role_name": role_name,
                "role_info": role_info,
                "dialog": row[1].split("\n"),
            }
            json_list.append(json_object)
    
    return json_list


def save_json(json_list, output_path):
    with open(output_path, "w", encoding="utf-8") as jsonfile:
        json.dump(json_list, jsonfile, ensure_ascii=False, indent=4)


def decode_csv_to_json(role_data_path, role_name, role_info, json_output_path):
    json_data = read_csv_to_json(role_data_path, role_name, role_info)
    save_json(json_data, json_output_path)
    

def load_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as file:
        text = file.read()
    return text


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_to_json(data, filepath, flag="w"):
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    with open(filepath, flag, encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=3))


def is_float(my_str):
    try:
        num = float(my_str)
        return True
    except ValueError:
        return False
