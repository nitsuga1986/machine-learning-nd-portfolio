

# Tensorflow / Tensorboard example
TensorFlow is one of the most popular deep learning frameworks available. It's used for everything from cutting-edge machine learning research to building new features for the hottest start-ups in Silicon Valley. Here is an examplo of how to create and train a machine learning model, as well as how to leverage visualization tools to analyze and improve your model. Finally, an example of how to deploy models locally or in the cloud,

It is recommended to create the virtual environment located in the root of this repository:

    conda env create -f ai-universal-v1.yml
    conda activate ai

or

    source activate ai

## Custom visualizations to TensorBoard

Clone this repository and run:

    python3 tensorboard_visuals.py 
    tensorboard --logdir=logs

TensorBoard it will give you a URL to open in your browser, copy and paste that to your browser.


## Custom visualizations to TensorBoard

Clone this repository and run:

    python3 tensorboard_visuals.py 
    tensorboard --logdir=logs

TensorBoard it will give you a URL to open in your browser, copy and paste that to your browser.

## Export models for use in production

Clone this repository and run:

    python3 export_model_for_cloud.py 

After run, you'll find new exported model folder and inside is a file called `saved_model.pb`. This file contains the structure of our model in Google's special proto buff format. There's also a variables subfolder that contains a checkpoint of all the variables in our graph. This model is now ready to be uploaded to the Google cloud.

 - Create Google Cloud Project
 - Go to API manager > Library
 - Look for `Google Cloud Machine Learning Engine`
 - Enable billing
 - Install the [Google Cloud SDK](https://cloud.google.com/sdk/downloads) > Run the interactive installer

Now we can upload the model files to a Google Cloud storage bucket. We'll do that with this command. Here, we're calling gsutil. Gsutil is a utility that handles lots of basic Google Service operations, like creating new storage buckets, moving files around, changing permissions, and so on. `mb` stands for "Make bucket." Google has data centers all over the world, and you have to tell them which one you want to use. I used `us-central1`, which is located in Iowa in the United States. Then, we have to name the storage bucket (change number at the end).

    gsutil mb -l us-central1 gs://tensorflow-class-1000
Next, we need to upload our model files into the bucket. We'll do that with this command.

    gsutil cp -R exported_model/* gs://tensorflow-class-1000/earnings_v1
Model files are now stored on Google servers. Next, we have to tell the Google Machine Learning Engine that we want to create a new model.
    
    gcloud ml-engine models create earnings --regions us-central1
    gcloud ml-engine versions create v1 --model=earnings --origin gs://tensorflow-class-1000/earnings_v1
Model is now live in the Cloud, and ready to be used. We can try out our model by using the `gcloud predict` command.

    gcloud ml-engine predict --model=earnings --json-instances=sample_input_prescaled.json
There are several ways we can use our Cloud-based model. If you want to make a few predictions on a small dataset, you can just use the `gcloud` command, like we've done here. If you want to make predictions for thousands or millions of items, you can upload a data file to a Cloud storage bucket and then use the gcloud command to make predictions from that data file. You can also use the Google Cloud API Client Library for any programming language to make calls to your model from any other program.

## Call our machine learning model from a Python program
To use a cloud based machine learning model from another program, we need two things, first, we need access to make calls to the cloud. For this, we need a *credentials file*. This file keeps our cloud based service secure from unauthorized use. Second, we need to write the *code to call our cloud based model*, for that, we'll use the Google cloud API client library.
Open the following file, complete with the credentials and PROJECT_ID and run:

    python3 python3 call_cloud_service


