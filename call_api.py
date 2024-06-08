import requests
import os
import base64
from PIL import Image
from io import BytesIO
import json


def base64_to_image(base64_string):
    # Remove the data URI prefix if present
    if "data:image" in base64_string:
        base64_string = base64_string.split(",")[1]

    # Decode the Base64 string into bytes
    image_bytes = base64.b64decode(base64_string)
    return image_bytes

def create_image_from_bytes(image_bytes):
    # Create a BytesIO object to handle the image data
    image_stream = BytesIO(image_bytes)

    # Open the image using Pillow (PIL)
    image = Image.open(image_stream)
    return image

def call_api_sync_data(url, headers):
    response = requests.get(url=url, headers=headers)
    employees = list(response.json()['employees'])
    if len(employees):
        folder = 'examples/emp_images'
        if os.path.exists(folder):
            # Delete all files in the folder
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                os.remove(file_path)
        else:
            # Create the folder
            os.makedirs(folder)

        for emp in employees:
            if emp['image'] and emp['image_id']:
                image_name = emp['image_id'] + ".jpg"
                file_path = os.path.join(folder, image_name)
                image_bytes = base64_to_image(emp['image'])
                img = create_image_from_bytes(image_bytes)
                img.save(file_path)

def post_attendance(image_id, type, timestamp, timezone):
    url = "http://127.0.0.1:8069/api/employee/attendance"
    headers = {
        "api-key": "@HKo#@eud&oDl^I9Drmp",
        "Content-Type": "application/json",
    }
    payload = {
        "image_id": image_id,
        "type": type,
        "timestamp": timestamp,
        "timezone": timezone
    }
    print("payload: ", payload)
    response = requests.post(url=url, data=json.dumps(payload), headers=headers)
    print("response: ", response.content)
    if response.status_code == 200:
        print("Successfully")
    else:
        print("Failed")