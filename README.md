# Company shareholding app

This flask application is able to find all the shareholders of a UK company.

## Description

UK registered companies are required by law to disclose their shareholder structure. This is first done at the incorporation of the company, and then whenever there is a change in shareholding, this is disclosed via an annual confirmation statement.
Although this is public information, there is no direct way to easily obtain a company's shareholder structure directly, unlike with other company information such as address or directors. In addition, the files are in many different formats, sometimes even hand written, with inconsistencies in people and company names. 
The purpose of this project is to create a basic algorythm that, for a given company, can find out the complete company shareholder tree. The application will scan through all confirmation statements and incorporation documents, creating a complete company shareholding table, showing who the shareholders were during which time periods. Then, it looks for those shareholders in companies house again, and recursively completes the whole shareholder tree.
The output is visualised in a network graph. The front end enables the user to select a date as well as the depth (levels) that the algorythm will go towards the ultimate beneficial owner.


## Getting Started

### Dependencies

All package requirements are in requirements.txt.
This app will need access to the companies house API, as well as to a LLM that can read images. The current version uses Gemini.

### Installing

* How/where to download your program
* Any modifications needed to be made to files/folders

### Executing program

* How to run the program
* Step-by-step bullets
```
code blocks for commands
```

## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
```

## Authors

Alessandro Brizio: brizio.alessandro@gmail.com

## Version History


* 0.1
    * Initial Release

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments
