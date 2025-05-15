# smartfarm-classification-api

An api to determine the severity level of anthracnose disease in chilli. The chilli is placed in a closed chamber and deployed in "cPanel Cloud Premium" services provided by Niagahoster.
This project is a part of STEI ITB's capstone design. This repo is an implementation of a **Cloud-Based Backend System for Chili's Antrachnose Disease Screening Chamber**. This repository is still in development. Credit to TA242501022 team, Najmi Azzahra Feryputri.

## Repo Structure 
| Directory / Files  | Description |
| -------------------| ------------- |
| Model              | Directory contains the PyTorch-based weight and bias file of the pre-trained YOLO11-m image classification model. |
| Results            | Directory contains a text file. The content of the text file is the JSON structure resulting from `/classify` |
| Scheduler          | Directory contains a JSON config file to set the cron job to automatically execute endpoint `/classify` and `/clean_old_dirs`  |
| Storage            | Directory to store chilli images |
| Temp               | Directory to save chilli images from Raspberry Pi. Act as a buffer storage between the Raspberry Pi and the Storage |
| app.py             | Python source code. Contain all api for classifying. Postman documentation will be available soon |
| passenger_wsgi.py  | WSGI entry to run the Python app in cPanel |
| requirements.txt   | All the requirements needed |

## How to Use 
### Local Deployment
1. Clone this repository.
2. Ask me privately about the environment file to access the database.
3. Make a virtual env on your local computer, install all the requirements.
4. Run the Python Flask application, `app.py`.
5. Try to access all the api. You can use browser to test the GET api. You have to use CURL or POSTMAN to test the POST api.

