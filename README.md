# smartfarm-classification-api

An api to do severity level of antrachnose in chilli. The chilli placed at a closed chamber. Deployed in "cPanel Cloud Premium" services in Niagahoster.
This project is a part of STEI ITB's capstone design. This repo is implementation of **Cloud-Based Backend System for Chili's Antrachnose Disease Screening Chamber**
This repository still in development. 
Credit to TA242501022 team, Najmi Azzahra Feryputri.

## Repo Structure 
| Directory / Files  | Description |
| -------------------| ------------- |
| Model              | Directory contain the pytorch-based weight anda bias file of pre-trained yolo11-m image classification model. |
| Results            | Directory contain text file. Content of the text file is the json structure resulted from `/classify` |
| Scheduler          | Directory contain json config file to set the cron job to automatically execute endpoint `/classify` and `/clean_old_dirs`  |
| Storage            | Directory to stored chilli images |
| Temp               | Directory to saved chilli images from raspberry pi. Act as an buffer storage between raspberri pi and Storage |
| app.py             | Python source code. Contain all api for classify backend. Postman documentation will available soon |
| passenger_wsgi.py  | WSGI entry to run the python app in cPanel |
| requirements.txt   | All the requirements needed |

## How to Use 
### Local Deployment
1. Clone this repository.
2. Ask me privately about the environment file to access the database.
3. Make virtual env in your local computer, install all the requirements.
4. Run the python flask application, `app.py`.
5. Try to access all the api. You can use browser to testing the GET api. You have to use CURL or POSTMAN to test the POST api.

