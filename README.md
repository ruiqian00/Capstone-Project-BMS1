# Streamlined Image Processing for Protein Crystallization

This is a project collaborated with Bristol Myers Squibb in a Capstone Project hosted by Carnegie Mellon University's Masters of Science in Data Analytics program.

## Overview

The end goal of the project is to create a phase diagram that shows the concentration which inhibits protein crystallization for every protein excipient type. The approach is to use a image classification model to classify a protein as the following classes: **crystalline, empty, clear** and **rest**. 

**Drop examples**

![image](https://github.com/RichardWang2/BMS_Project/assets/41966482/934357ce-af91-41cb-b091-1710ace811b3)

**Phase Diagram Example**

![image](https://github.com/RichardWang2/BMS_Project/assets/41966482/58b98532-f3f8-420b-83f1-c4118db41a75)

## How to use:

Upload a folder with 96 images, resembling the 96 well-plate. Label each image with **excipientname_numberinwellplate**. 
First, create and set up the conda environment 

The trained Swin Transformer model can be downloaded at https://drive.google.com/file/d/14GmQf5JnUH9pgb9su_0ke8xTycGJzdgj/view?usp=drive_link

`conda create -n <Your_environment_name> python=3.11`

`pip install environment.yml`

And then run 

`python test.py --test_dir <Your Test Data Directory>`

This will create a CSV file containing the file names and corresponding predictions.


