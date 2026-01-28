import datetime
import requests
import json
import os
import re

def sendRequest(url, data, apiKey=None):
    headers = {'X-Auth-Token': apiKey} if apiKey else {}
    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()  # Raises HTTPError for bad responses
        response_data = response.json()
        if response_data.get('errorCode'):
            print(f"Error accessing {url}: {response_data['errorCode']} - {response_data['errorMessage']}")
            return None
        return response_data['data']
    except requests.exceptions.HTTPError as err:
        print(f"HTTP Error accessing {url}: {err}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed for {url}: {e}")
        return None

def authenticate(username, token):
    #Authenticate with the USGS API and get an API key.
    LOGIN_URL = "https://m2m.cr.usgs.gov/api/api/json/stable/login-token"
    payload = {'username': username, 'token': token}
    api_key = sendRequest(LOGIN_URL, payload)
    if api_key:
        print(f"Authentication successful! API Key: {api_key}")
    else:
        print("Authentication failed.")
    return api_key


def search_data(api_key, dataset_name, spatial_filters, start_date, end_date, cloudfilter={"min": 0, "max": 100}, max_scenes=5, verbose=True):
    #Search for Landsat 8-9 Level 1 data with cloud masks and limit results for testing.
    SEARCH_URL = "https://m2m.cr.usgs.gov/api/api/json/stable/scene-search"
    payload = {
        #"datasetName": "landsat_ot_c2_l1",                
        "datasetName": dataset_name,
        "maxResults": max_scenes,  
        "sceneFilter": {
            #"spatialFilter": { "filterType": "mbr", "lowerLeft": {"latitude": -100, "longitude": -110}, "upperRight": {"latitude": 19, "longitude": 109}},
            "spatialFilter": spatial_filters,
            "acquisitionFilter": {"start":start_date,"end": end_date },
            "cloudCoverFilter": cloudfilter  # Include all cloud coverage
        },
    }

    if verbose:
        print(payload)
        
    results = sendRequest(SEARCH_URL, payload, api_key)

    if results and "results" in results:
        if verbose:
            print(f"Retrieved {len(results['results'])} scenes.")
        return results["results"]
    else:
        if verbose:
            print("No scenes found.")
        return None

def search_data_ext(api_key, dataset_name, filterdict, sortfield=None, sortdirection=None, max_scenes=5, verbose=True):
    #Search for Landsat 8-9 Level 1 data with cloud masks and limit results for testing.
    SEARCH_URL = "https://m2m.cr.usgs.gov/api/api/json/stable/scene-search"

    payload = {
        #"datasetName": "landsat_ot_c2_l1",                
        "datasetName": dataset_name,
        "maxResults": max_scenes,  
        "sceneFilter": filterdict
    }
 
    if sortfield:
        payload['sortField']=sortfield

    if sortdirection:
        payload['sortDirection']=sortdirection
    
    if verbose:
        print(payload)
        
    results = sendRequest(SEARCH_URL, payload, api_key)

    if results and "results" in results:
        if verbose:
            print(f"Retrieved {len(results['results'])} scenes.")
        return results["results"]
    else:
        if verbose:
            print("No scenes found.")
        return None


def retrieve_metadata(api_key, dataset_name, id, idtype = 'displayId', verbose=True):
    #Search for Landsat 8-9 Level 1 data with cloud masks and limit results for testing.
    SEARCH_URL = "https://m2m.cr.usgs.gov/api/api/json/stable/scene-search"
    '''
    example:
    {
        "datasetName": "landsat_ot_c2_l2",
        "entityId": "LC08_L2SP_012025_20201231_20210308_02_T1",
        "idType": "displayId",
        "metadataType": "full",
        "useCustomization": false
    }
    '''
    payload = {     
        "datasetName": dataset_name,
        "entityId": id,
        "idType": idtype,
        "metadataType": "full",
        "useCustomization": False
    }
 
    results = sendRequest(SEARCH_URL, payload, api_key)

    if results and "results" in results:
        if verbose:
            print(f"Retrieved {len(results['results'])}.")
        return results["results"]
    else:
        if verbose:
            print("No Results")
        return None



def add_scenes_to_list(api_key, list_id, dataset_name, entity_ids):
    ADD_SCENES_URL = "https://m2m.cr.usgs.gov/api/api/json/stable/scene-list-add"
    payload = {
        "listId": list_id,
        "datasetName": dataset_name,
        "entityIds": entity_ids  # List of entityIds
    }

    result = sendRequest(ADD_SCENES_URL, payload, api_key)
    if result:
        print(f"Scenes added successfully: {result}")
    else:
        print("Failed to add scenes to the list.")

def download_options(api_key, dataset_name, list_id, verbose=True):
    DOWNLOAD_OPTIONS_URL = "https://m2m.cr.usgs.gov/api/api/json/stable/download-options"      
    # Request download options for the entire list
    payload = {
        "datasetName": dataset_name,
        "entityIds": list_id,
        "includeSecondaryFileGroups": True 
        
    }
    
    #print(json.dumps(payload))
    results = sendRequest(DOWNLOAD_OPTIONS_URL, payload, api_key)
   
   # Initialize an empty list for downloads
    downloads = []

    # Filter results to find available downloads
    for option in results:
        if option.get('available', False):
            downloads.append({
                'entityId': option['entityId'],
                'productId': option['id']
            })

    # Output the number of available downloads found
    if downloads:
        if verbose:
            print(f"Available downloads found: {len(downloads)}")
    else:
        if verbose:
            print("No available downloads found.")
    
    return downloads



def download_request(api_key, downloads):
    DOWNLOAD_REQUEST_URL = "https://m2m.cr.usgs.gov/api/api/json/stable/download-request"
    label = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # Unique label using current date-time
    payload = {
        'downloads': downloads,
        'label': label
    }

    # Send the request to get direct download URLs
    request_results = sendRequest(DOWNLOAD_REQUEST_URL, payload, api_key)
    return request_results, label

def download_retrieve(api_key, label):
    DOWNLOAD_RETRIEVE_URL = "https://m2m.cr.usgs.gov/api/api/json/stable/download-retrieve"
    payload = {'label': label}
    download_urls = []

    response = sendRequest(DOWNLOAD_RETRIEVE_URL, payload, api_key)
    if response and 'available' in response:
        for download in response['available']:
            if 'url' in download:
                download_urls.append(download['url'])
                print(f"Download ready: {download['url']}")
        if not download_urls:
            print("No available downloads at the moment.")
    else:
        print("No downloads are currently available or the request failed.")

    return download_urls

#Maybe add multithreading         
def download_file(url, path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Ensure we notice bad responses
        
        content_disposition = response.headers.get('content-disposition')
        if content_disposition and 'filename=' in content_disposition:
            filename = content_disposition.split('filename=')[1].strip('"')
        else:
            filename = f"download_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"

        full_path = os.path.join(path, filename)

        with open(full_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Downloaded {filename} to {full_path}")
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error occurred: {e}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the file: {e}")
    except OSError as e:
        print(f"OS Error occurred: {e}")

