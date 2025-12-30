# Company shareholding app

This flask application is able to find the shareholder structure of any UK limited company.

## Description

UK registered companies are required by law to disclose their shareholder structure. This is first done at the incorporation of the company, and then whenever there is a change in shareholding, this is disclosed via an annual confirmation statement.
Although this is public information, there is no direct way to easily obtain a company's shareholder structure, unlike with other company information such as address or directors. In addition, the files are in many different formats, sometimes even hand written, with inconsistencies in people and company names. 
The purpose of this project is to create a basic algorythm that, for a given company, can find out the complete company shareholder tree. The application will scan through all confirmation statements and incorporation documents, creating a complete company shareholding table, showing who the shareholders were during which time periods. Then, it looks for those shareholders in Companies House again, and recursively completes the whole shareholder tree.
The output is visualised in a network graph. The front end enables the user to select a date as well as the depth (levels) that the algorythm will go towards the ultimate beneficial owner.


### Dependencies

All package requirements are in requirements.txt.
This app will need access to:
1) The companies house API. This can be obtained for free from here: https://developer.company-information.service.gov.uk/get-started
2) LLM for image recognition. Currently, it is using gemini-2.0-flash-lite. 


### Installing

1) Create a .env file in the app folder with the following: GEMINI_API_KEY, COMPANIES_HOUSE_API_KEY, FLASK_SECRET_KEY
2) Run app.py

## Testing

Run test_utils.py

## Authors

Alessandro Brizio: brizio.alessandro@gmail.com

## Version History

* 0.1, 31/12/2025
    * Initial Release 

