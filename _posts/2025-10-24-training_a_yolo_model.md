---
layout: post
title: "Training a YOLO model"
date: 2025-10-23
---
## tl;dr
- model training is more accessible than i expected (thanks in large part to [this template](https://github.com/mfranzon/yolo-training-template))
- matching the dataset to the usecase is key (this is obvious in hindsight)
- training loss and mAP figures are not a great proxy for real-world performance
- ml models have a lot of levers available for tuning performance

## Where do we begin?

Last week i came across a post by mfranzon’s about training a computer vision ML model on a laptop with just a few lines of Python. 

I’ve done a lot of experimenting with foundational models but hadn’t yet experimented with ML much. This felt like a low-friction opportunity to go ‘up the stack’ and get some hands-on experience training models to build better intuition on the subject.

_This post was written in an effort to be more intentional about documenting and sharing my experimentation. I haven’t done one of these in years so i’ll be shaking off some of the rust, bear with me._

## YOLO Training Template

mfranzon’s was generous enough to post his [training template here](https://github.com/mfranzon/yolo-training-template). The [readme](https://github.com/mfranzon/yolo-training-template/blob/main/README.md) says its a "template for training YOLO models on kaggle datasets". 

That might as well be a different language to me so naturally I had to start with some definitions.

- **You-Only-Look-Once (YOLO)** refers to an object detection algorithm that “looks” at an image only once to find both bounding boxes and probabilities simultaneously ([this blog](https://www.v7labs.com/blog/yolo-object-detection) from v7 was helpful).
- **Kaggle dataset** just refers to any one of the many datasets hosted on the Kaggle platform. Kaggle is great for learning about ML. 

Ok so now I have a good-enough understanding of **_what_** mfranzon is trying to  help me achieve. The next step is to understand **_how_**.

## Getting started
The readme describes a couple of python scripts  for running training and inference locally, but it also pulls it all into a Jupyter notebook which I know can run on  Colab. Back to some definitions.

- **Training** refers to the actual training of the yolo model. The act of giving it a bunch of labeled images to ‘learn’ from and validating its performance. 
- **Inferencing** refers to running the trained model across an unlabeled image (or video) and getting back a labeled images. Essentially asking the model to identify where on the image the objects of interest exists. 
- [**Jupyter Notebook**](https://jupyter.org/) is an interactive local dev environment that lets you build and run code in different ‘steps’ and see the results in-line. Its a great learning tool.
- [**Google Colab**](https://colab.research.google.com/) is a _web-based_ dev environment where you can run Jupyter notebooks and avoid any local setup. Plus you get to plug into google’s compute resources for free (!).

I decided to run the Jupyter notebook in Colab for a few reasons:
1. I wanted to avoid setup issues. Running locally  sometimes means troubleshooting dependencies. I wanted to hit the ground running, not spend an hour chasing down some mis-matched python version.
2. I’m  impatient. Training a model on your laptop sounds very cool, but the reality is that its pretty slow. I wanted to iterate and experiment quickly; waiting 1-4hrs to train locally would’ve killed my momentum. Colab give me access to high performance GPUs that will cut training down to minutes.

Enough semantics, at this point I understand what we’re trying to do and have a path forward for how I’ll do it. Now lets get started.

## What is this thing?
First things first, I cloned the repo and opened the notebook in Colab (turns out you can skip the cloning and just open the notebook directly in Colab). With the notebook open I read-through each cell to get a better understanding of what the scripts are doing and how they talk to eachother:

- Cells 1 & 2 install and import dependencies to the dev environment.
- Cell 3 defines the functions to train the model. Generally:
  - Download the dataset from Kaggle.
  - Structure the dataset images and labels in a way that works with the training library.
  - Run the actual training by calling the training library and passing in the training data (along with some other parameters).
- Cell 4 is all about inference. It defines functions that run inference on either an image, video or web-cam using the trained model.
- Cell 5 defines parameters that are sent to the training functions. We’ll need to update these before we  run the training.
- Cells 6 is where we actually call the functions in Cell 3, and pass the values from Cell 5 to run the training. 
- Cell 7 defines parameters that are sent to the  inference functions. We’ll have to update these before running inference.
- Cell 8 is where we run inference by calling the functions in Cell 4 & passing the values from Cell 7. 

Ok, that was a lot… But now we have a baseline understanding of how this notebook gets us from 0-to-trained model. Its time to pick the dataset and see what’s next.

## Picking a dataset
The training-template repo has a few example datasets that are well-suited for training. 

In theory this template could work with any dataset that conforms to one of the structures defined in Cell 3’s `detect_dataset_structure()`. But after a quick look at the options on Kaggle its clear that not all datasets are created equally. 

I’m here to run training and inference, not troubleshoot datasets, so we better stick with some ‘well-chosen’ datasets from `example_datasets.md` and keep it moving. I chose pothole detection dataset for a few reasons:
- Its been done. The author of the template repo himself trained on this dataset. There are also many example of models that do the same. This means I’ll have a lot of resources to pull from when I surely run into trouble.
- Theres a bunch of pothole datasets on kaggle, so we can add to our training set later if we want.
- There’s only 1 class, so it should train pretty quickly.

But there are a few downsides that I also considered:
- Its been done before. I’m not gonna get the rush of doing something new. But thats fine for me. I’m here to learn.
- There’s only 1 class which isn’t as rich as something like the Animal Detection data set which has 80. Again, we’ll live. 

At this point we have all our ducks in a row i’m ready to start training. I updated the `dataset_handle` parameter in cell 5 to reference the pothole detection handle and clicked 'run' on Cells 1-6. The training begins and now we wait...

## Run #1 Everythings a pothole
Training run #1 was pretty quick coming in at around ~5 min (thanks Google). I chose to only run training for 1 Epoch, my hope was that this would be fast and good-enough to start inferencing on videos from outside the training set.

I uploaded a couple of stock videos to my colab environment showing roads from an over-head persepctive and got ready to run inference on this. This process was also really easy thanks to the scripts in the template. All I had to do was:
- Upload the videos to colab & copy the path to their locations
- Paste the path in Cell 7 `input_source`
- Set your save-to location as the `output_path`
- Click 'run' on Cell 8 & wait. 

_Note on saving the output. Its really easy to save to Colab environment but bigger files to download is a hassle. I ended up mounting my gdrive and using that as my save-to location._

So how did my freshly trained model do? I'd call it... _enthusiastic_. Everythings a pothole! Grass? Pothole. Motorcycle? Pothole. Fun to watch, sure, but not quite the usefulness I had hoped for. 

![1epoch_annotated_vid.gif](/assets/2025-10-24-training_a_yolo_model/1epoch_annotated_vid.gif)

So what went wrong? My first guess was that the model hadn't seen enough training data, or gone through enough iterations of the training data, to get good at detecting potholes. 

This is called 'underfitting' and its a common problem that occurs when the model just hasn't learned enough. There's only one way to fix a model that's underfitting. More training, which means more epochs. 

## Prepping for another training run
Alright, so the 1 epoch run is underfitting. Not a big surprise since it only took a few moments to run. There's parameters in Cell 5 of the jupyter notebook that let me set how many epochs I want to run on my next training run. 

Before doing that, I want to make sure i'm _updating_ the training on my 1-epoch model, not starting from scratch. This isn't a major concern, but on principle I wanted to avoid starting from scratch.

After a quick review of `train_model()` I see that the solve is pretty simple. Just update the value of `model` to reference our newly created model. That way training begins from the 1-epoch model, not the empty model. 
 
When the training re-runs, it will over-write the 1-epoch model with the newly updated one, which is what I want. 

## Run #2 Nothings a pothole
I set `epochs = 20` and hit 'run' and wait. After just 20 min the training is complete and we can check the results.

As the training ran I watched the results of each epoch on the console. Each one includes some scores to help assess if the model is getting any better. Since I'm just experimenting I focused on `box_loss` and `mAP50`:
- **Box Loss (box_loss)** refers to how often the model drew a square around an object that wasn't a pothole. The lower this score, the better the model is performing. 
- **Mean Average Precision @ 50% overlap (mAP50)** looks at how often the model drew a square that overlapped at least 50% with the training data. The higher this score, the better the model is performing. 

With the training completed I thought it might be interesting to plot the loss and precision on a graph. 
![boxloss_blue_vs_mAP50_orange_by_Epoch.png](/assets/2025-10-24-training_a_yolo_model/boxloss_blue_vs_mAP50_orange_by_Epoch.png)
([data here](/assets/2025-10-24-training_a_yolo_model/boxloss_blue_vs_mAP50_orange_by_Epoch.json))


You'll notice that as as epochs complete, box_loss goes down while mAP goes up. This means my model is getting better at predicting potholes! (right?)

The only way to know for sure is to run a fresh inference using this new model.  Unfortunately the results were... _underwhelming_. Suddenly, nothing is a pothole, until the very end when it decides the motorcycle is a pothole. 

![overhead_annotated_20e_conf40pct.gif](/assets/2025-10-24-training_a_yolo_model/overhead_annotated_20e_conf40pct.gif)

This is also not useful, so where did I go wrong? 

## Troubleshooting overfitting
The first word that came to mind is "overfitting". Overfitting occurs when a model is trained for too long on a dataset that is not diverse enough. 

The behavior of an overfit model is that it will perform well on the trainig data, but poorly on validation data. This would translate to also performing porly on realworld data if it is not closely similar to the training data.  

Maybe overfitting could be the case here: 
- The kaggle dataset is small at only ~2700 images.
- Many of them are very similar images.
- I _did_ run for 20 epochs. Maybe that's enough to overfit?

I'm not sure, but the reality is that our training data shows that the model is performing well on both training data AND validation data. Its only falling short on realworld data. 

So IMO overfitting _is not_ the likely cause of the model's woes. Whats next?

## Troubleshooting confidence
So if its not the training, maybe its the inference. During inference, the model takes some parameters as inputs. One of those parameters is confidence threshold ()`conf_thresh`) which it uses to determine how liberally to detect objects.

**Confidence threshold** refers to the certainty the model must have that it has correctly detected an object before drawing a box around it during inference. The higher this value, the fewer boxes the model will draw. The lower the value, the more boxes it will draw.

I left this value as its default `conf_thresh = 0.5`. It could be the case that the model is detecting the potholes, but with too low confidence to draw the box. Maybe I should try tweaking confidence & rerunning the inference to see if the result improves. 

I progresively reduced confidence from `0.5` all the way down to `0.05` and checked reuslts at every interval. I will spare you each result but , but I thought the one below captured the problem with this approach well. 

Take a look at the inference results when we set confidence to `0.05`. My first observation was that its detecting a lot more actual potholes! This is a great start, but it doesn't last. 

As the scene changes to an overhead view, the model picks up less and less potholes. All throughout it thinks the motorcycle is a pothole. This is _a little_ better, but it feels like the models grip on what a pothole looks like is tenuous. 

![overhead_annotated_20e_conf05pct.gif](/assets/2025-10-24-training_a_yolo_model/overhead_annotated_20e_conf05pct.gif)

I think there are a few conclusions to draw from this:
1. First is that using ultra-low confidence is pretty risky. A lot of non-potholes are detected which reduces the utility of the inference severely. 
2. Second, while confidence is part of the solution, its not the only issue here.

Something else is afoot. 

## Domain Shift
So the model knows what _some_ potholes look like. And its able to pick up a lot of potholes but only at a very low confidence. 

I noticed when the shot moves to way overhead, the model is unable to pick up any potholes, even at a low confidence. 

I think this points to a problem related to domain shift.

**Domain shift** refers to a situation where training and validation data do not match well to real-world conditions. The net result is a model really good at detecting objects when conditions are similar to the training set, but which fails otherwise. 

I think thats what's at play here. 

To verify, I took a look at the dataset itself 

from our 'real-world'
- IDK i think its DOMAIN SHIFT. 
- Whats domain shift? its xyz
- Does it map to our situation? yee pretty much
- how do we confirm? 
  - inference on pictures more similar to training data
  - or expand training set to include pictures similar to my use-case (overhead drone shots of roads with potholes)

So now what?
- Gotta update the training set!

