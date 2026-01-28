import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
from . import usgs_library as usgs
import tarfile
import re
import fileutils
import json
import zipfile

def authenticateAPI():
    username = ""
    token = ""
    return usgs.authenticate(username, token)

def format_items(items):
    return "".join([f'["{x}"]' if isinstance(x, str) else f'[{x}]' for x in items])

def extract_and_save(tar, file_member, output_dir):
    """Extracts a file from a tar archive and returns the path to the extracted file."""
    tar.extract(file_member, path=output_dir)
    return os.path.join(output_dir, file_member.name)

def find_all_keys_with_paths(data, target_key, path="", results=None):
    """Finds all occurrences of a key in a nested structure and returns values with their paths."""
    if results is None:
        results = []
    if isinstance(data, dict):
        if target_key in data:
            results.append(f'{path}["{target_key}"]')
        for key, value in data.items():
            find_all_keys_with_paths(value, target_key, f'{path}["{key}"]', results)
    elif isinstance(data, list):
        for index, item in enumerate(data):
            find_all_keys_with_paths(item, target_key, f'{path}["{index}"]', results)
    return results

def find_key_by_value(data, value_pattern, path=None, depth=0):
    if path is None:
        path = []  # Initialize the path as an empty list

    result_paths = []

    # If the data is a dictionary, search through the keys
    if isinstance(data, dict):
        for key, value in data.items():
            new_path = path + [key]
            #if value == target_value:
            if isinstance(value, str) and re.search(value_pattern, value):
                result_paths.append(new_path)
                #print(depth, result_paths,"dict", "leaf")
            elif isinstance(value, (dict, list)):
                result_paths.extend(find_key_by_value(value, value_pattern, new_path, depth+1))
                #if len(result_paths)>0: print(depth, result_paths,"dict")
    # If the data is a list, iterate over elements
    elif isinstance(data, list):
        for index, item in enumerate(data):
            new_path = path + [index]
            #if item == target_value:
            if isinstance(item, str) and re.search(value_pattern, item):
                result_paths.append(new_path)
                #print(depth, result_paths,"list", "leaf")
            elif isinstance(item, (dict, list)):
                result_paths.extend(find_key_by_value(item, value_pattern, new_path, depth+1))
                #if len(result_paths)>0: print(depth, result_paths,"list")
    return result_paths

def read_filenames_and_extract_dates(folder_path, pattern=".zip"):
    files_data = []

    for filename in os.listdir(folder_path):
        if filename.endswith(pattern):
            base_name = os.path.splitext(filename)[0]
            parts = base_name.split("_")
            if len(parts) >= 4:
                date_str = parts[3]
                files_data.append({'displayId': base_name, 'date_string': date_str})

    df = pd.DataFrame(files_data)
    return df

def getmetadata_json(id, folder, metadict): #from folder
    fname=id+'.tar'
    tarfname=None
    search_pattern='LC*_'+fname.split("_")[2]+'_'+fname.split("_")[3]+'_*.tar'
    for f in fileutils.find_files(folder, search_pattern, listtype="list"):
        tarfname=f
        break
    if tarfname is None:
        return None, None
    with tarfile.open(os.path.join(folder,tarfname), 'r') as tar:
        metafile=os.path.splitext(os.path.basename(tarfname))[0]+'_MTL.json'
        member = tar.getmember(metafile)        
        extracted_file = tar.extractfile(member) # Extract it into memory
        if extracted_file:
            content = extracted_file.read()  # This is now in memory as bytes
            mdata=json.loads(content)
            #text = content.decode('utf-8')  # or appropriate encoding
        else: 
            return metadict
    for k in metadict:
        d=metadict[k]
        if "jsonpath" not in d:
            paths=find_all_keys_with_paths(mdata, k)
            if paths:
                d["jsonpath"]=paths[0]
            else:
                print("Field not found in metadata. Exiting...")
        d["value"]=eval('mdata'+d["jsonpath"])
    return metadict, tarfname

def parse_txt_metadata(txt_content, fields):
    metadata = {}
    for line in txt_content.splitlines():
        line = line.strip()
        if "=" in line:
            key, val = map(str.strip, line.split("=", 1))
            if key in fields:
                metadata[key]={}
                # Clean up quotes or infer types
                if val.startswith('"') and val.endswith('"'):
                    val = val[1:-1]
                #elif val.replace(".", "", 1).isdigit():
                #    val = float(val) if "." in val else int(val)
                metadata[key]['value'] = val
    return metadata

def parse_json_metadata(content, fields_list):
    mdata = json.loads(content)
    fields_dict={field : {} for field in fields_list}
    for k in fields_dict:
        d = fields_dict[k]
        if "jsonpath" not in d:
            paths = find_all_keys_with_paths(mdata, k)
            if paths:
                d["jsonpath"] = paths[0]
            else:
                print("Field not found in metadata. Exiting...")
                continue
        d["value"] = eval('mdata' + d["jsonpath"])
    return fields_dict

def getmetadata(id, folder, metadict_or_fields, archive_type='tar', meta_format='json'):
    fname = id + ('.tar' if archive_type == 'tar' else '.zip')
    archive_path = None
    search_pattern = 'LC*_' + fname.split("_")[2] + '_' + fname.split("_")[3] + '_*.' + archive_type

    for f in fileutils.find_files(folder, search_pattern, listtype="list"):
        archive_path = f
        break

    if archive_path is None:
        return None, None

    # Construct expected file name
    base_name = os.path.splitext(os.path.basename(archive_path))[0]
    if meta_format == 'json':
        metafile = base_name + '_MTL.json'
    else:
        metafile = base_name + '_MTL.txt'

    if archive_type == 'tar':
        open_func = tarfile.open
        mode = 'r'
    elif archive_type == 'zip':
        open_func = zipfile.ZipFile
        mode = 'r'
    else:
        raise ValueError("Unsupported archive type. Use 'tar' or 'zip'.")

    with open_func(os.path.join(folder, archive_path), mode) as archive:
        try:
            if archive_type == 'tar':
                member = archive.getmember(metafile)
                extracted_file = archive.extractfile(member)
            else:  # zip
                metafile = os.path.join(id,metafile)
                extracted_file = archive.open(metafile)
        except KeyError:
            print(f"Metadata file {metafile} not found in archive.")
            return None, archive_path

        if extracted_file:
            content = extracted_file.read().decode('utf-8')
        else:
            return None, archive_path

    # Process metadata
    if meta_format == 'json':
        metadict_or_fields = parse_json_metadata(content, metadict_or_fields)
        '''
        mdata = json.loads(content)
        for k in metadict_or_fields:
            d = metadict_or_fields[k]
            if "jsonpath" not in d:
                paths = find_all_keys_with_paths(mdata, k)
                if paths:
                    d["jsonpath"] = paths[0]
                else:
                    print("Field not found in metadata. Exiting...")
                    continue
            d["value"] = eval('mdata' + d["jsonpath"])
        '''
        return metadict_or_fields, archive_path
    else:
        fields = metadict_or_fields  # In this case, it's a list of field names
        metadata = parse_txt_metadata(content, fields)
        return metadata, archive_path


def getmetadata_m2m(api_key, id, metalist): #from internet
    '''
    UL_LAT	Upper Left Latitude
    UL_LON	Upper Left Longitude
    UR_LAT	Upper Right Latitude
    UR_LON	Upper Right Longitude
    LL_LAT	Lower Left Latitude
    LL_LON	Lower Left Longitude
    LR_LAT	Lower Right Latitude
    LR_LON	Lower Right Longitude
    CENTER_LAT
    CENTER_LON
    Nadir/Off Nadir
    Day/Night Indicator
    Scene Cloud Cover L1
    Sun Elevation L0RA
    Sun Azimuth L0RA
    Roll Angle
    Landsat Product Identifier ??
    '''
    datasetname="landsat_ot_c2_"+id[5:7].lower()
    mdata=usgs.retrieve_metadata(api_key, datasetname, id, verbose=True)
    
    for d in metalist:
        if "key" not in d:
            paths=find_key_by_value(mdata, "Start Time")
            if paths:
                path=paths[0]
                print(f"Found access path(s) for value 'Start Time': {path}")
                d["key"]=path
            else:
                print("'Start Time' variable not found in metadata. Exiting...")

        else:
            path=d["key"]
            d["value"]=eval('mdata'+format_items(d["key"][:-1])+'["value"]')
    return metalist