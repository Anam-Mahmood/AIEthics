# A Simple Way To Remove Bias From AI Models Using Low Code

## Workshop Resources

- Login/Sign Up for IBM Cloud: http://ibm.biz/wwcAIEthics
- Survey: https://ibm.biz/wwcSurvey

## Table of Contents

  * [Prerequisites](#prerequisites)
    + [Sign-up/Login to IBM Cloud](#sign-up-login-to-ibm-cloud)
  * [About the Workshop](#about-the-workshop)
  * [How does the fairness algorithm work?](#how-does-the-fairness-algorithm-work-)
  * [Architecture diagram](#architecture-diagram)
    + [Flow](#flow)
  * [Step 1: Create an IBM Cloud account](#step-1-create-an-ibm-cloud-account)
  * [Step 2: Create Watson Studio service](#step-2-create-watson-studio-service)
  * [Step 3: Create a new Watson Studio project](#step-3-create-a-new-watson-studio-project)
  * [Step 4: Add Data](#step-4-add-data)
  * [Step 5: Create the notebook](#step-5-create-the-notebook)
    + [Note: You will create three Notebooks here by repeating the below steps.](#note-you-will-create-three-notebooks-here-by-repeating-the-below-steps)
  * [Step 6: Insert the data as dataframe](#step-6-insert-the-data-as-dataframe)
  * [Step 7: Run the notebook & Analyze Result](#step-7-run-the-notebook---analyze-result)
  * [Feedback](#feedback)
  * [Reference](#reference)

## Prerequisites

### Sign-up/Login to IBM Cloud

If you are an existing user please login to IBM Cloud through http://ibm.biz/wwcAIEthics

And if you are not, don't worry! We have got you covered! There are 3 steps to create your account on IBM Cloud: 
1. Put your email and password. 
2. You get a verification link with the registered email to verify your account. 
3. Fill the personal information fields. 
** Please make sure you select the country you are in when asked at any step of the registration process.

![image](https://user-images.githubusercontent.com/54094367/136783264-38bc77b7-9bd4-42f6-867f-85a61feff3b7.png)

## About the Workshop 
How do we remove bias from the machine learning models and ensure that the predictions are fair? What are the three stages in which the bias mitigation solution can be applied? This code pattern answers these questions and more to help developers, data scientists, stakeholders take informed decision by consuming the results of predictive models.

Fairness in data, and machine learning algorithms is critical to building safe and responsible AI systems from the ground up by design. Both technical and business AI stakeholders are in constant pursuit of fairness to ensure they meaningfully address problems like AI bias. While accuracy is one metric for evaluating the accuracy of a machine learning model, fairness gives us a way to understand the practical implications of deploying the model in a real-world situation.

## How does the fairness algorithm work?

The bias mitigation algorithm can be applied in three different stages of model building. These stages are pre-processing, in-processing & post-processing. The below diagram demonstrates how it works.
![algorithm-working](https://github.com/Anam-Mahmood/AIEthics/blob/main/images/aif-360-flow.png?raw=true)

The AIF360 Python package contains nine different algorithms, developed by the broader algorithmic fairness research community, to mitigate that unwanted bias. They can all be called in a standard way, very similar to scikit-learn’s fit/predict paradigm. In this way, we hope that the package is not only a way to bring all of us researchers together, but also a way to translate our collective research results to data scientists, data engineers, and developers deploying solutions in a variety of industries. You can learn more about AIF 360 here.

## Architecture diagram
![architecture-diagram](https://github.com/Anam-Mahmood/AIEthics/blob/main/images/architecture-2.png?raw=true)


### Flow

1. Log in to Watson Studio powered by spark, initiate Cloud Object Storage, and create a project.
2. Upload the .csv data file to Object Storage.
3. Load the Data File in Watson Studio Notebook.
4. Install aif 360 Toolkit in the Watson Studio Notebook.
5. Analyze the results after applying the bias mitigation algorithm during pre-processing, in-processing & post-processing stages.


## Step 1: Create an IBM Cloud account

Login/Sign-up for IBM Cloud Account: http://ibm.biz/wwcAIEthics


## Step 2: Create Watson Studio service

Go to your IBM Cloud Dashboard, type in search box "Watson studio" select the service, on the service page click create

<img width="540" alt="1" src="https://user-images.githubusercontent.com/16270682/130227865-a245e3e9-e1b4-46d5-8048-f8809d846446.PNG">

<img width="910" alt="2" src="https://user-images.githubusercontent.com/16270682/130227908-5a72177b-2fda-4a8f-8d70-fd4f8761c36a.PNG">


## Step 3: Create a new Watson Studio project

Click on get started it will take you to a "Cloud Pak for Data" Dashboard. 

<img width="490" alt="3" src="https://user-images.githubusercontent.com/16270682/130227947-ead78d84-75e2-48ad-b489-18939abfdc46.PNG">

<img width="837" alt="4" src="https://user-images.githubusercontent.com/16270682/130227980-2aa2c5b4-65f7-4b39-b5a0-96fbd559536e.PNG">


Select create a project

<img width="580" alt="5" src="https://user-images.githubusercontent.com/16270682/130228261-90016159-e276-416c-af47-d3337baa4edb.PNG">

Click on create an empty project

<img width="916" alt="6" src="https://user-images.githubusercontent.com/16270682/130228274-2dcd1641-f504-43d6-aaee-ce62a942ab02.PNG">


Give your project a unique name and click on create 

<img width="930" alt="7" src="https://user-images.githubusercontent.com/16270682/130228284-6c2d3a6b-fe41-4d4a-abba-c2223443e233.PNG">

## Step 4: Add Data

Go to https://github.com/Anam-Mahmood/AI-Ethics and either clone or download code in zip file. Un-zip the folder.

![download-zip](https://github.com/Anam-Mahmood/AIEthics/blob/main/images/Screen%20Shot%202021-09-16%20at%205.25.19%20PM.png)

Now that you have the code files on your computer, go to your AutoAI project on IBM Cloud.
Click on Assets and select Browse and add fraud_data.csv file from your file system. Repeat the step and add the Pipeline_LabelEncoder-0.1.zip file as an asset.

<img width="918" alt="9" src="https://user-images.githubusercontent.com/16270682/130228315-b24224e9-a3c1-4710-9f35-6caf48bc9c3a.PNG">

## Step 5: Create the notebook


### Note: You will create three Notebooks here by repeating the below steps. 

From IBM Watson Dashboard click on "Add to project" and select "Notebook"

<img width="750" alt="10" src="https://user-images.githubusercontent.com/16270682/130228834-fb48f7fb-14f5-4cc5-9da6-15d770e9d16e.PNG">

<img width="548" alt="11" src="https://user-images.githubusercontent.com/16270682/130228859-7f3ad65d-1e87-4ee3-beaf-562ebf60bbda.PNG">

Create a  Pre-processing Notebook. Select the "From URL" tab and Enter a name for the notebook. Select the runtime (2 vCPU and 8 GB RAM.)
Enter this Notebook URL for Pre-processing : https://github.com/IBM/bias-mitigation-of-machine-learning-models-using-aif360/blob/main/notebooks/Pre-processing.ipynb

<img width="953" alt="12" src="https://user-images.githubusercontent.com/16270682/130231839-5dff0600-943b-4231-9a12-c7f061a37ffd.PNG">


After the notebook is imported, click on Not Trusted and select the option to trust the source of the notebook.

<img width="861" alt="13" src="https://user-images.githubusercontent.com/16270682/130231849-db8293d5-7543-4458-b62e-c85d428e23b2.PNG">


Repeat the above steps for in-processing and Post-processing 

Enter this Notebook URL for In-processing : https://github.com/IBM/bias-mitigation-of-machine-learning-models-using-aif360/blob/main/notebooks/In-processing.ipynb
Enter this Notebook URL for Post-processing : https://github.com/IBM/bias-mitigation-of-machine-learning-models-using-aif360/blob/main/notebooks/Post-processing.ipynb

## Step 6: Insert the data as dataframe

Open Pre-processing Notebook from Dashboard and click on edit

<img width="622" alt="14" src="https://user-images.githubusercontent.com/16270682/130236156-1172456b-d221-4141-b142-ba953a6adbdc.PNG">


<img width="671" alt="15" src="https://user-images.githubusercontent.com/16270682/130236095-1ef5f296-026b-4977-a788-26fcdd014de5.PNG">


Click on 0010 icon at the top right side which will bring up the data assets tab. Click on Insert to code dropdown for fraud_data.csv and select the option Insert Pandas Dataframe.

<img width="245" alt="Screenshot 2021-08-20 173819" src="https://user-images.githubusercontent.com/16270682/130236073-da6aa8f7-5d64-4ee5-a239-a8ed0d76a528.png">


## Step 7: Run the notebook & Analyze Result

Click on Run icon to run the code

<img width="576" alt="16" src="https://user-images.githubusercontent.com/16270682/130236006-504d58d2-ff57-4403-a2a5-b584b9aa5d4d.PNG">

When a notebook is executed, what is actually happening is that each code cell in the notebook is executed, in order, from top to bottom.

Each code cell is selectable and is preceded by a tag in the left margin. The tag format is In [x]:. Depending on the state of the notebook, the x can be:

    A blank, this indicates that the cell has never been executed.
    A number, this number represents the relative order this code step was executed.
    A *, this indicates that the cell is currently executing.

There are several ways to execute the code cells in your notebook:

    One cell at a time.
        Select the cell, and then press the Play button in the toolbar.
    Batch mode, in sequential order.
        From the Cell menu bar, there are several options available. For example, you can Run All cells in your notebook, or you can Run All Below, that will start executing from the first cell under the currently selected cell, and then continue executing all cells that follow.
        
        
After we run all cells in the notebook, the results are displayed at the end of each notebook per below.
Pre-processing results We can observe that, priviledged group had 37% more chance of getting a favorable outcome because of the bias in the dataset.

<img width="635" alt="17" src="https://user-images.githubusercontent.com/16270682/130235779-3a41c4cc-be3f-430e-9f01-62ff85523195.PNG">


## Feedback
Your feedback matters to us! Take this short survey and let us know how we are doing!
https://ibm.biz/wwcSurvey
![survey](https://github.com/Anam-Mahmood/AI-Ethics/blob/main/images/wwc%20survey%20qr%20code.png?raw=true)


## Reference

https://github.com/IBM/bias-mitigation-of-machine-learning-models-using-aif360 

Additional material:
- If you've never heard of AI Ethics before, watch this quick video: https://www.youtube.com/watch?v=aGwYtUzMQUk
- Read more about Trustworthy AI here: https://www.ibm.com/watson/trustworthy-ai





