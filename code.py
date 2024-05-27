

!pip install -qU python-dotenv openai

import os
import dotenv
#import openai
import json

from google.colab import drive
drive.mount('/content/drive')

"""client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
)
"""

#dotenv.load_dotenv("/content/drive/MyDrive/GenAI/key.txt")
from openai import OpenAI

text1="The appointment was for trouble breathing and it was scheduled on December 21, 2022. Doctor Valentina was fantastic. She was very positive about communication and took the time to explain everything clearly. Nurse Danny was also helpful during the appointment. The receptionist, Jose, was friendly and helpful. However, the lobby and waiting rooms were dated and dirty. It was difficult to find a parking spot. Overall, I wouldn't recommend this clinic."

def format_json(reviews):
  json_formatted_str=json.dumps(reviews,indent=2)
  return(json_formatted_str)

functions=[
        {
            "name": "extract_data",
            "description": "Extraction of reviews of patients at hospital.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reviews": {
                        "title": "reviews",
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "Appointment Date": {
                                    "type": "string",
                                    "description": "Appointment date of the patient"
                                },
                                "Reason": {
                                    "type": "string",
                                    "description": "Reason for the appointment"
                                },
                                "Doctor": {
                                    "type": "string",
                                    "description": "Name of the doctor seen during the appointment"
                                },
                                "Doctor Rating": {
                                    "type": "string",
                                    "description": "Rating given for the doctor's performance"
                                },
                                "Doctor Comment": {
                                    "type": "string",
                                    "description": "Comments or feedback about the doctor's performance"
                                },
                                "Nurse": {
                                    "type": "string",
                                    "description": "Name of the nurse seen during the appointment"
                                },
                                "Nurse Rating": {
                                    "type": "string",
                                    "description": "Rating given for the nurse's performance"
                                },
                                "Nurse Comment": {
                                    "type": "string",
                                    "description": "Comments or feedback about the nurse's performance"
                                },
                                "Receptionist": {
                                    "type": "string",
                                    "description": "Name of the receptionist encountered during the appointment"
                                },
                                "Receptionist Rating": {
                                    "type": "string",
                                    "description": "Rating given for the receptionist's performance"
                                },
                                "Receptionist Comment": {
                                    "type": "string",
                                    "description": "Comments or feedback about the receptionist's performance"
                                },
                                "Waiting Room Rating": {
                                    "type": "string",
                                    "description": "Rating given for the waiting room's condition"
                                },
                                "Waiting Room Comment": {
                                    "type": "string",
                                    "description": "Comments or feedback about the waiting room's condition"
                                },
                                "Parking Rating": {
                                    "type": "string",
                                    "description": "Rating given for the parking availability or condition"
                                },
                                "Parking Comment": {
                                    "type": "string",
                                    "description": "Comments or feedback about the parking availability or condition"
                                },
                                "Overall Rating": {
                                    "type": "string",
                                    "description": "Overall rating or recommendation based on the entire appointment experience"
                                },
                                "Overall Comment": {
                                    "type": "string",
                                    "description": "General comments or feedback about the overall appointment experience"
                                }
                            }
                        }
                    },
                },
                "required": ["Appointment Date"],
            },
        }
    ]

available_functions={
    'extract_data':format_json,
}

messages=[{'role':'user','context':text1}]

client = OpenAI(api_key="Update your OpenAI key here")

response = client.chat.completions.create(
model="gpt-3.5-turbo", messages=messages, functions=functions, function_call="auto")



import pandas as pd
import re

# Read the CSV file
data = pd.read_csv('/content/competition_dataset_3_13.csv', header=None, names=['Text'])

# Function to extract relevant information
def extract_info(text):
    # Extract appointment date
    date_pattern = r'(\d{1,2}/\d{1,2}/\d{4})|(\d{1,2} \w+ \d{4})'
    appointment_date = re.search(date_pattern, text)
    appointment_date = appointment_date.group() if appointment_date else None

    # Extract reason
    reason_pattern = r'appointment\s+(?:was\s+)?for\s+(\w+)'
    reason = re.search(reason_pattern, text, re.IGNORECASE)
    reason = reason.group(1) if reason else None

    # Extract doctor's name
    doctor_pattern = r'Dr\.\s(\w+)'
    doctor = re.search(doctor_pattern, text)
    doctor = doctor.group(1) if doctor else None

    # Extract doctor's rating
    doctor_rating_pattern = r'Dr\.\s\w+\s(was\s\w+\snegative|was\s\w+\spositive)'
    doctor_rating = re.search(doctor_rating_pattern, text, re.IGNORECASE)
    doctor_rating = doctor_rating.group(1).split()[-1] if doctor_rating else None

    # Extract doctor's comment
    doctor_comment_pattern = r'Dr\.\s\w+,?\s(.*?)\.'
    doctor_comment = re.search(doctor_comment_pattern, text)
    doctor_comment = doctor_comment.group(1) if doctor_comment else None

    # Extract nurse's name
    nurse_pattern = r'Nurse\s(\w+)'
    nurse = re.search(nurse_pattern, text)
    nurse = nurse.group(1) if nurse else None

    # Extract nurse's rating
    nurse_rating_pattern = r'Nurse\s\w+\swas\s(\w+)'
    nurse_rating = re.search(nurse_rating_pattern, text, re.IGNORECASE)
    nurse_rating = nurse_rating.group(1) if nurse_rating else None

    # Extract nurse's comment
    nurse_comment_pattern = r'Nurse\s\w+\s(.*?)\.'
    nurse_comment = re.search(nurse_comment_pattern, text)
    nurse_comment = nurse_comment.group(1) if nurse_comment else None

    # Extract receptionist's name
    receptionist_pattern = r'receptionist,\s(\w+)'
    receptionist = re.search(receptionist_pattern, text)
    receptionist = receptionist.group(1) if receptionist else None

    # Extract receptionist's rating
    receptionist_rating_pattern = r'receptionist,\s\w+,?\s(was\s\w+\s\w+)'
    receptionist_rating = re.search(receptionist_rating_pattern, text, re.IGNORECASE)
    receptionist_rating = ' '.join(receptionist_rating.group(1).split()[-2:]) if receptionist_rating else None

    # Extract receptionist's comment
    receptionist_comment_pattern = r'receptionist,\s\w+,?\s(.*?)\.'
    receptionist_comment = re.search(receptionist_comment_pattern, text)
    receptionist_comment = receptionist_comment.group(1) if receptionist_comment else None

    # Extract waiting room rating
    waiting_room_rating_pattern = r'waiting\sroom\ss(?:tyle\s)?(?:of\sthe\sclinic\s)?was\s(\w+\sand\s\w+)'
    waiting_room_rating = re.search(waiting_room_rating_pattern, text, re.IGNORECASE)
    waiting_room_rating = waiting_room_rating.group(1) if waiting_room_rating else None

    # Extract waiting room comment
    waiting_room_comment_pattern = r'waiting\sroom\ss(?:tyle\s)?(?:of\sthe\sclinic\s)?(.*?)\.'
    waiting_room_comment = re.search(waiting_room_comment_pattern, text, re.IGNORECASE)
    waiting_room_comment = waiting_room_comment.group(1) if waiting_room_comment else None

    # Extract parking rating
    parking_rating_pattern = r'parking\ss(?:ituation\s)?was\s(\w+)'
    parking_rating = re.search(parking_rating_pattern, text, re.IGNORECASE)
    parking_rating = parking_rating.group(1) if parking_rating else None

    # Extract parking comment
    parking_comment_pattern = r'parking\ss(?:ituation\s)?(.*?)\.'
    parking_comment = re.search(parking_comment_pattern, text, re.IGNORECASE)
    parking_comment = parking_comment.group(1) if parking_comment else None

    # Extract overall rating
    overall_rating_pattern = r'(?:i\s+)?(?:would|wouldn\'t)\s+recommend\s+this\s+clinic'
    overall_rating = re.search(overall_rating_pattern, text, re.IGNORECASE)
    overall_rating = 'recommend' if overall_rating and 'would' in overall_rating.group() else 'not recommend'

    # Extract overall comment
    overall_comment_pattern = r'(?:i\s+)?(?:would|wouldn\'t)\s+recommend\s+this\s+clinic\.(.*)'
    overall_comment = re.search(overall_comment_pattern, text, re.IGNORECASE)
    overall_comment = overall_comment.group(1).strip() if overall_comment else None

    return {
        'appointment_date': appointment_date,
        'reason': reason,
        'doctor': doctor,
        'doctor_rating': doctor_rating,
        'doctor_comment': doctor_comment,
        'nurse': nurse,
        'nurse_rating': nurse_rating,
        'nurse_comment': nurse_comment,
        'receptionist': receptionist,
        'receptionist_rating': receptionist_rating,
        'receptionist_comment': receptionist_comment,
        'waiting_room_rating': waiting_room_rating,
        'waiting_room_comment': waiting_room_comment,
        'parking_rating': parking_rating,
        'parking_comment': parking_comment,
        'overall_rating': overall_rating,
        'overall_comment': overall_comment
    }

# Apply the function to extract structured data
structured_data = data['Text'].apply(extract_info)

# Convert the structured data to a DataFrame
structured_df = pd.DataFrame(structured_data.tolist())

# Save the DataFrame as a CSV file
structured_df.to_csv('structured_data_2.csv', index=False)
