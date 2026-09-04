import requests
from utils.func import extract_before_parenthesis
import os
from requests.exceptions import SSLError
import time
import sys
import pandas as pd
from utils.paths import DATABASE
import numpy as np
import re


def distance_km(value):
    """Parse a numeric distance without evaluating text from the database/API."""
    match = re.fullmatch(r'\s*([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)\s*(km|m)\s*', str(value))
    if not match:
        raise ValueError(f'Invalid distance: {value!r}')
    number = float(match.group(1).replace(',', ''))
    return number if match.group(2) == 'km' else number / 1000

# This tool refers to the "DistanceMatrix" in the paper. Considering this data obtained from Google API, we consistently use this name in the code. 
# Please be assured that this will not influence the experiment results shown in the paper. 

class GoogleDistanceMatrix:
    def __init__(self, subscription_key: str="") -> None:
        self.gplaces_api_key: str = subscription_key
        self.data =  pd.read_csv(DATABASE / 'googleDistanceMatrix/distance.csv')
        print("GoogleDistanceMatrix loaded.")

    def run(self, origin, destination, mode='driving'):
        origin = extract_before_parenthesis(origin)
        destination = extract_before_parenthesis(destination)
        info = {"origin": origin, "destination": destination,"cost": None, "duration": None, "distance": None}
        response = self.data[(self.data['origin'] == origin) & (self.data['destination'] == destination)]
        if len(response) > 0:
                if pd.isna(response['duration'].values[0]) or pd.isna(response['distance'].values[0]):
                    return "No valid information."
                info["duration"] = response['duration'].values[0]
                info["distance"] = response['distance'].values[0]
                if 'driving' in mode:
                    info["cost"] = int(distance_km(info["distance"]) * 0.05)
                elif mode == "taxi":
                    info["cost"] = int(distance_km(info["distance"]))
                if 'day' in info["duration"]:
                    return "No valid information."
                return f"{mode}, from {origin} to {destination}, duration: {info['duration']}, distance: {info['distance']}, cost: {info['cost']}"

        return f"{mode}, from {origin} to {destination}, no valid information."   
    
    def run_for_evaluation(self, origin, destination, mode='driving'):
        origin = extract_before_parenthesis(origin)
        destination = extract_before_parenthesis(destination)
        info = {"origin": origin, "destination": destination,"cost": None, "duration": None, "distance": None}
        response = self.data[(self.data['origin'] == origin) & (self.data['destination'] == destination)]
        if len(response) > 0:
                if pd.isna(response['duration'].values[0]) or pd.isna(response['distance'].values[0]):
                    return info
                info["duration"] = response['duration'].values[0]
                info["distance"] = response['distance'].values[0]

                if 'day' not in info["duration"]:
                    if 'driving' in mode:
                        info["cost"] = int(distance_km(info["distance"]) * 0.05)
                    elif mode == "taxi":
                        info["cost"] = int(distance_km(info["distance"]))

                return info

        return info 


    def run_online(self, origin, destination, mode="driving"):
        # mode in ['driving','taxi','walking', 'distance','transit']
        endpoint = "https://maps.googleapis.com/maps/api/distancematrix/json"

        params = {
            "origins": origin,
            "destinations": destination,
            "mode": "driving" if mode in {"taxi", "self-driving"} else mode,
            "key": self.gplaces_api_key
        }

        for attempt in range(3):
            try:
                response = requests.get(endpoint, params=params, timeout=30)
                response.raise_for_status()
                break
            except SSLError:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)

        data = response.json()
        info = {"origin": origin, "destination": destination,"cost": None, "duration": None, "distance": None}
        if data['status'] == "OK":
            element = data['rows'][0]['elements'][0]
            if element['status'] == "OK":
                info["duration"] = element['duration']['text']
                info["distance"] = element['distance']['text']
                if 'driving' in mode:
                    info["cost"] = int(distance_km(info["distance"]) * 0.05)
                elif mode == "taxi":
                    info["cost"] = int(distance_km(info["distance"]))
                # if 'day' in info["duration"]:
                #     return "No valid information."
                return f"{mode}, from {origin} to {destination}, duration: {info['duration']}, distance: {info['distance']}, cost: {info['cost']}"

        return "No valid information."   
    
    def run_for_annotation(self, origin, destination, mode="driving"):
        # mode in ['driving','taxi','walking', 'distance','transit']
        endpoint = "https://maps.googleapis.com/maps/api/distancematrix/json"

        params = {
            "origins": extract_before_parenthesis(origin),
            "destinations": extract_before_parenthesis(destination),
            "mode": "driving" if mode in {"taxi", "self-driving"} else mode,
            "key": self.gplaces_api_key
        }
        
        response = requests.get(endpoint, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        info = {}
        if data['status'] == "OK":
            element = data['rows'][0]['elements'][0]
            if element['status'] == "OK":
                info["duration"] = element['duration']['text']
                info["distance"] = element['distance']['text']
                info["cost"] = None
                if 'driving' in mode:
                    info["cost"] = int(distance_km(info["distance"]) * 0.05)
                elif mode == "taxi":
                    info["cost"] = int(distance_km(info["distance"]))
        else:
            info = {"duration": "N/A", "distance": "N/A", "cost": "N/A", "Hint":"Please check the input."}
        return info
    
