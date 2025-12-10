---
layout: post
title: "WIP - Data strategy to correct domain shift"
date: 2025-12-06
image: /assets/data-strategy-post/test-img-w-masks.png
---

## tl;dr
- **Clear problem statements are half the battle.** Well defined usecases obviate data requirements. Without this, my experimentation was destined to fail.
- **Data makes a big difference (duh!).** By selectimg the right data alone i saw performance improvements by 2 OOM. Tweaking training parameters got me to the target benchmark.
- **Data curation is simple, _not easy_.** Sourcing, annotating, and processing data is a grind but its the _most_ important task and where HITL matters most. Afterall, model quality is _derived_ from data quality.
- **Clear outcomes flow from controlled variables.** Trying a bunch of things is the best way to know what does/doesn't work. But changing one variable at a time is key to drawing a line between input changes and performance improvements.

## Where do we begin?
Last month I decided to train a computer vision model. It was super easy and fun, but didn't net the best results. There's a [full write up here](/_posts/2025-10-24-training_a_yolo_model.md), but the gist of it is that I didn't really start with the end in mind, and thats how I discovered the importance of data strategy for ML applications. 

On the heels of that flop my intention was to take on a more thoughtful approach to the notion that _data strategy_ is the difference between a cvis model that works and mine.

So as to not leave any loose threads from my last post (and to have fun things to do over the long thanksgiving weekend) I set out to solve my domain shit problem by:
- Defining a [clear usecase](#defining-a-clear-use-case) to serve as a 'what' and 'why' anchor for my experimentation.
- Decide on a [success metric](#deciding-on-a-benchmark) for this usecase that will provide an objective measurement on performance.
- [Cultivate](#data-strategy) and [prepare](#preparing-the-data) relevant training data in support of my usecase.
- [Train some models](#running-experiments) to check their performance on my usecase.
- Draw some [conclusions](#so-what-conclusion) on what got me to (or kept me from) achieving the success criteria. 

Like all things, this seems simple on the surface, but required navigating a lot of nuance in domains I'm not that familiar with (see [Dunning-Kruger effect](https://resilienceshield.com/app/uploads/2021/05/Mt-Stupid-1-1.png)).

We'll start at the beginning.

## Defining a clear use case
Last time I was so gung ho to train a model that skipped over the most important part: **Defining the problem**

I needed to take a step back from the 'how' and lean into the 'what'. My previous model faceplanted when running inference on overhead footage of a dirt road. That's a pretty narrow failure condition so it felt like a good starting point. 

In the real world I would want to validate and 'harden' this problem definition with user reasearch and data, but under my current constraints (low budget, no commercial ambitions), this is sufficient. That said, I don't want to understate the importance of this part. I think its the _most important_ part of achieving success on this type of project (maybe all projects).

So I've got my target. **Usecase:** `Detect potholes from overhead view on dirt roads with little traffic.` 

Now that I know what problem I want to solve, I'll need a metric to tell me when I've solved it.

## Deciding on a benchmark
Determining whether or not the model is performant seems trivial: Either it detects the potholes under the stated conditions or it doesnt, right? 

Kinda... but there's a couple of questions that need answers:

1. **What measure signals success?**
    - [mAP](https://blog.roboflow.com/mean-average-precision) (mean average precision) is a standard metric in object detection that balances recall (finding all the objects) with precision (avoiding all the non-objects). 
    - Its a measure of how well the model detects objects when compared to the real object boundary ([this blog](https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173) gets into some of the math behind _how_ its calculated).
    - I'll be using [**mAP50**](https://docs.ultralytics.com/guides/yolo-performance-metrics/#class-wise-metrics) as my measure for performance. Its a pretty leinient metric, but still gives clear signal on whether or not the right object is being detected.
2. **What threshold is sufficient?** 
    - Stated differently: "If mAP50 is the test, what score must the model achieve to prove its ability?"
    - I'm trying to prove traction towards a solution, not achieve perfection on the first try. I think a `mAP50=50%` sufficiently demonstrates utility while leaving lots of room for improvement.

The metric (mAP50) and threshold (50%) I chose both depend on validating performance on images with real potholes already identified (test data). Running these metrics against the irrelevant test data would yield useless results. 

I'll need to manually tag some images with potholes which begs the question:

**What is a pothole?** 
- Definition: _a depression or hollow in a road surface caused by wear or sinking_.
- For the sake of this experiment I made some convenient assumptions:
    - All _road_ defects are potholes. _Ground_ defects off road **are not** potholes.
    - All puddles are potholes (but not the inverse). 
    - Not all potholes are circular. They can be uniquely shaped.

I used [roboflow](https://roboflow.com/) to pull frames from the video that caused my original model to fail and manually annotated ~1500 potholes to become my 'Test' dataset. I'll benchmark all the models I train against this test set to draw a clear conclusion on how well we're adressing the problem. 

Now that I've got a KPI to target we can talk data that'll get us there.

## Data strategy
With a clear problem statement the data required is obvious: I need overhead images of dirt roads with potholes. 

Again, this seems simple but there are nuances to manage:
- Dirt roads look differently at different times of day.
- The landscape around the roads can be dramatically different.
- Weather conditions play a role in how the road itself looks (e.g. snow covered)
- Potholes look differently at different overhead distances and angles. 
- View of road can be partially obstructed by natural elements, like trees.

I'll need to source training data that adequately represents this variance.

I expected to find public-license datasets that matched my usecase, but I was met with a-whole-lotta-nothing. I decided the best parth forward would be to source my own training data. 

For this experiment I followed the following constraints:
1. **Ethicaly sourced:** Ethics in data matters. I wanted to make sure the data I curated was ethically sourced. This is not a commercial pursuit, so I leaned on fair-use.
2. **Hybrid data:** I wanted to mix real and synthetic data. I used [Pexels](https://www.pexels.com/) for royalty-free stock footage and [Veo 3](https://deepmind.google/models/veo/) for some synthetic training examples.
3. **Highly curated:** I have limited bandwidth so I'm prioritizing quality>quantity. I cherry picked material that represented the nuance sufficiently, but didn't go beyond that.

I decided on 1 img per second of video. This approach got me to a final count of ~550 images covering the full spectrum of variance mentioned earlier. This is a good start, but these are just raw images. I still need to this collection into a labeled dataset. 

---

## Preparing the data
Turning these 550 images into a real dataset that can be used for fine tuning was a grind. 

Roboflow made this as painless as possible, but manually selecting _each individual pothole_ was tedious to say the least. I tried using a SOTA model (SAM3) to do the labeling for me, but found that the model struggled to identify 'pothole' from these images (maybe my model will fill a real gap in the pothole identifying marketplace).

Armed with my 'pothole' definition from earlier I got to tagging and bucketing images between train/valid. 

Once complete, I uploaded the [dataset to Kaggle](https://www.kaggle.com/datasets/nmata010/overhead-potholes-test-set-v1) so that I could pass it into my colab notebook with the scaffolding I already have in place (and so others can follow along). 

We're ready to start training and tweaking models!

## Running experiments
Alright! I have a problem to solve, a KPI to target, and a dataset that I'm hoping will get me there. Lets get into it!

This is the loop I generally followed. It didn't always go flow linearly, but for the most part I changed 1 variable at a time to get clear signal on what was affecting the mAP results:
1. Make an assumption.
2. Change a training variable to test that assumption.
3. Train a model with those variables.
4. Benchmark it on new dataset & record outcome.
5. Come to a conclusion & start over.

To get 1:1 comparisons across different models I made some tweaks to the [open-source colab notebook](https://github.com/mfranzon/yolo-training-template/blob/main/notebooks/yolo_template.ipynb) I'm using. These changes let me add validation without changing the existing training/inference scripts: 
- Pick which model to validate & where to save results
- Download new dataset and select `test` path
- Run validation on the test images & store results
- _**note:** thanks to @ultralytics for [this write up](https://docs.ultralytics.com/modes/val/#example-validation-with-arguments) which helped a lot_

(I'm about to get into the deets, but you can skip to the [summary table](#summary-table) if you're impatient)

### Control group 
First up I had to establish a baseline. I started with the models from the previous experiment (trained on the ground-level pothole dataset) and ran them agains the aerial-vew pothole test images.
- **Assumption:** I'm expecting these models to perform poorly. They're specialized for ground-level potholes and but don't generalize well to aerial view ("domain shift")
- **Result:** 

| Model | mAP50 |
| -- | -- |
| [Control_1e](https://huggingface.co/nmata010/street-level-pothole-detection-11192025_1epoch) | 0.45%
| [Control_20e](https://huggingface.co/nmata010/street-level-pothole-detection-11192025_20epoch) | 0.42%

- **Conclusion:** These two models were trained on the original ground-level dataset. They perform as poorly as I expected them to.

This effectively establishes a floor for how bad a model can perform. Any notable improvement to these scores is worth exploring. 

### Aerial_1e
My assumption is that the control models' performance (or lack of) is the result of domain shift. I swapped out the training data and ran a short training on the new dataset. 
- **Assumption:** Control models suffer from domain shift. Training on relevant data should net significant improvement. 
- **Results:**

| Model | mAP50 |
| -- | -- |
| [Aerial_1e](https://huggingface.co/nmata010/aerial-pothole-detection-11212025_1Epoch_newDS) | 10.2%

- **Conclusion:** This is a major improvement from baseline, though it comes far short of the 50% threshold. And we were able to achieve this with a relatively small dataset. 

This is clear signal that relevant data matters. In my opinion, this _confirms_ domain-shift as the key failure of the control models.

### Aerial_20e
- https://huggingface.co/nmata010/aerial-pothole-detection-11252025_20Epoch_newDS
- trained on aerial dataset for 20epoch
- hypothesis is that training for longer will result in significantly improved mAP50
- observation is that we do get a jump in performance with `mAP50=42.9%`
- conclusion is that training for longer on the same data set does improve performance significantly. We clearly have the right data and are heading in the right directioin, but again fall short of the 50% benchmark. The performance gains are non-linear, training for 20x longer only yielded 4x improvement
- here we see major improvements
### Aerial_350e
- https://huggingface.co/nmata010/aerial-pothole-detection-12022025_350Epoch_newDS
- trained on aerial dataset for 350epochs
- hypothesis is that since performance gains are non-linear, we should expect the improvements to trail off. Training for will achieve slight improvements
- observation is that we do get a milder but still very significant improvement and achieve `mAP50=50.4%`. This took around 2h to train and i struggled with colab limits. This maybe near the upper bound of the training i can do on colab for free. 
- conclusion is that the performance improvements indeed trail off but not before we reached our benchmark of mAP50=50%. This confirms that we can achieve a POC grade performance with a relatively small training set. How can we squeeze more out of the same data?
- here we start to see diminishing returns, but still notable improvements 
### Roboflow_Aerial_350e
- trained on aerial dataset for 350epochs; roboflow does hyperparameter tuning in the background. 
- hypothesis is that we'll see a very small improvement by leveraging hyper parameter tuning
- observation is that we see a small but meaningful improvement of 13% by using hyper parameter tuning.
- conclusion is that getting to production grade (mAP50=95%) likely requires more robust data _and_ hyper parameter tuning. 

### Summary Table


| # | Model | Assumption  | Dataset | Epochs | Result (mAP50) | Observation | Conclusion 
| -- | -- | -- | -- | -- | -- | -- | -- 
| 0 | Control_1e | -- | Street-level potholes | 1 | **0.45%** | Model fails on aerial images | **Baseline** 
| 1 | Control_20e | More training on same data will correct domain shift | Street-level potholes | 20 | **0.42%** | Model fails on aerial images. Underperforms baseline | **Rejected** 
| 2 | Aerial_1e | Training on images more releavnt to the test case will correct domain shift | Aerial view potholes | 1 | **10.2%** | Significantly outperforms basilne (2.2 OOM) but falls well short of benchmark (50%) |**Supported** 
| 3 | Aerial_20e | More training on same data will improve model performance | Aerial view potholes | 20 | **42.9%** | Big performance improvement (4.2x). Tracking towards benchmark but still falls short. Training time relates to performance non-linearly | **Supported** 
| 4 | Aerial_350e | A long training run will yield better performance but a reduced rate. | Aerial view potholes | 350 | **50.4%** | Achieves benchmark! Some improvement (17%) | **Supported** 
| 5 | RoboflowAerial_350e | hyper-parameter tuning = better perf | Aerial view potholes | 350 | **57%** | -- | **Supported** 
| 6 | meta/SAM3 | SOTA model will achieve 50% benchmark with no fine-tuning on domain data | -- | --| xx% | Falls far short on average (though does a very well on a few of the individual frames). | **Rejected** 



## So what? (conclusion)
- yea the data made the biggest differene
- long training runs also made a significant difference
- so did hyperparamater tweaking
- Confirms that data is king
- Got to POC level benchmark
- "and therefore...?" what do i conclude after all this? 

## Now what?
- cool I did it:
    - settled on a problem to solve
    - designed a data strategy that maps to the problem
    - trained a bunch of models to test different approaches
    - achieved a POC grade level of performance
- Whats next?
    - Getting to a production level of performance for fun?
    - Putthing this model on a drone and selling it to rural municipalities?
    - abandoning it altogether? 
Idk, but this was fun and i leraned a lot:
- data makes or breaks the product. Wrong data = no viable outcomes. cut and dry. 
- Figuring out what experiments to run is both an art and science. It requires thoughtfulness to avoid doing a bunch of things that aren't really helping get closer to a solution.
- idk what else i learned

## Data makes a difference
- Data makes a difference. And a big one at that. 
- I want to say something about diminishing returns (the diminishing returns are also visible in the 350e training images)


## The scientific method
I really wanted this to be scientific and not just 'trying things til something works'. This is the real process you have to follow in the real world. You have to control variables to create reliable observations, otherwise you don't know what worked and why. 

My plan is pretty simple:
- Observation: The models trained on street-level potholes performed poorly on aerial dirt-road usecase. (1e mAP50: 0.0045, 20e mAP50: 0.00421)
- Hypothesis: This is a domain shift resulting from training data that does not match the usecase. Using domain specific training data will allow me to achieve mAP50 of 0.5
- Experiment: Train various models and change one single idk idk idk this doesnt make sense for so many experiments unless i'm gonna write a whole paper on it.


- observation
- question
- hypothesis
- experiment
- analysis
- conclusion


## The Experiments

We ran a series of experiments, treating each as a variable to isolate what actually drives performance.

---
# Notes
I wanna improve my cvis model. Its trained to detect potholes but mostly at street-level. It fails when detecting on far overhead images. I wanna make it better at detecting overhead images. 

This is purely an experiment, but i want to take it seriously, so i'm going to figure out what the right data strategy is for this use-cse.

To start i'm gonna set a really narrow and specific usecase.


I think to solve this effectively i need to:
1. figure out what problem i wanna solve (see usecase)
    - Ok so first things first. Lets decide what problem i want to solve. This doesn't require any special know-how just pick a problem and lets solve it. Since i ran into this domain shift problem earlier by trying to run inference on an overhead video of a dirt road I'll decide on that as my problem to solve. 
    - **Usecase:** Detect potholes from overhead view on dirt roads with little traffic. 
2. hypothesize what data i need to solve it
    - diverse images of dirt roads taken from overhead
    - there's gotta be more to this
    - different weather? Different angles?
    - I'm going to have to source and annotate these. 
3. determine a measure of success (how do i know my hypothesis is proven true or false)
    - maybe one approach is to manually annotate one image and then use that as a benchmark? 
    - Or create a set of benchmark images?
        - i can get some frames (20?) from the video i used before and manualy annotate every single pothole to use as a benchmark set
        - I'll hve to learn how to benchmark images. but that's doable
    - success could be achieving mAP50 of 0.6
        - i have to run my 20e model on the benchmark dataset to get a baseline, and then i can set a success metric as a function of that. So like if my 20e model gets to mAP50 of 0.1, then i could target mAP50 of 0.5
        - Ok we used roboflow to chop up the target video into 23 frames (1 per second) and manually annotated 1444 individual potholes. 
        - This was tedious work that is a little mind-numbing for such a simple case.
        - Clear that at scale and for complex situations labeling is a VERY complex problem. 
        - I'm marking all the annotated images as 'test' images (roboflow), and exporting in yolov8 format since that's our yolo version from the template.
        - I uploaded my dataset to kaggle (https://www.kaggle.com/datasets/nmata010/overhead-potholes-test-set-v1)
    - benchmarking against annotated images will give me some sense of how its performing similar to how we could see mAP improving during training in the original experiment. But what about realworld performance? Since i don't have a 'real' usecase i'm not sure how to benchmark success. i'm gonna need to do some soulsearching here.
        - to benchmark without training i need to use model.val() which is not yet implemented in the repo template. model.val() is a function from ultralytics. We're already using model.train & model.export in the main training cell to train and save the model. 
        - The documentation for ultralytics model functions is here: https://docs.ultralytics.com/reference/engine/model/#ultralytics.engine.model.Model
        - I implemented a validation script using ultralytics. It takes the current model and performs validation against the validation dataset. 
        - To achieve that i had to modify the notebook. i want to make that contribution to the OSS repo. 
        - I used the script to run the validation and stored the validation results in my gdrive under /benchmarkes
        - I still need to interpret the output. I'm not exactly sure how to read the results. 
        - Ok i actually did update the OSS repo with the validation scripts. And i PRd that in https://github.com/mfranzon/yolo-training-template/pull/8. Ok it merged, cool!
    - I ran the benchmark on my 20epoch model. Its stored in gdrive under benchmarks. 
        - Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.1s/it 2.1s
            all         23       1444     0.0553     0.0284     0.0361    0.00975
        - Basically the model achieves a 50% overlaps only 3% of the time
        - the model achieves a 95% overlaps only <1% of the time
        - There are some annotated images that show how bad it was compared to the manually annotated images
    - Once i have other validation references I want to do a "proof of life" like a side by side showing inference results from this 20e model and the updated model. "proof of life"
    - While prepping the new dataset I realized i could turn the bounding boxes in my benchmark set into polygons which would give better precision. Here are the new baseline benhcmarks:
        - Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.1s/it 1.9s
            all         23       1430    0.00793    0.00629    0.00421   0.000662
        - The model achieves 50% overlap < 0.5% of the time
        - The model achieves 95% overlap < 0.0% of the time
        - Basically, its _really_ bad at detecting potholes in my test dataset
    - Here's a blog about why polygons are better than boxes for this type of application. (https://www.ayadata.ai/bounding-boxes-in-computer-vision-uses-best-practices-for-labeling-and-more/)
4. source or create the data
    - In a perfect world i'll find this dataset already annotated (fingers crossed)
    - in the absence of that i'll have to create the dataset and annotate it myself (sounds tedious but idk)
        - roboflow and CVAT have tools to do this. Gotta look into this
    - For unannotated images/videos i can go online to some stock sites.
    - I think i could also ask gemini to create some images for me? Possibly also videos? 
        - looks like this is a whole category of data called "synthetic data" which has its own rabbit holes. I'll try it out, but will abandon if needed. 
    - This whole section sounds like its gonna be a grind unless i find some neat dataset that fits. 
    - There may be an arugment for pivoting my use-case around the available datasets since this is not real-world, but that feels like cheating. 
    - OK here we go. I got 3 stock videos and a synthetic video. 
    - I'm using 3 videos from pexel.com (free) and 1 video that i created with nano banans
5. prepare the data
    - this is kinda the same as above.
    - annotated images
    - text for each image with the box information (yolo.txt)
    - idk anything about preparing training data so i need to cover a lot of ground here.
        - Roboflow has a tool to annotate and export. 
    - I uploded them to roboflow and used meta's SAM3 to quckly identify the potholes (the irony is not lost on me that i'm using a FM to detect potheledoles in my training data to train a model to detect potholes.)
    - SAM3 is SOTA so i don't feel bad that its better than my model at detecting potholes. But i did note that it really struggles with overhead images, dirt roads, and dealing with the non-road background. 
    - Anyway sam3 did a pretty good job on a couple of the videos and completely fails at 2 others. 
    - I manually lab the other remaining videos. That was very tedious. Data labeling is a grind.
    - Ok i also did some autmentation for brigness and exposure. I'll let the yolo training process deal with tilts. 
    - I think with that i have my dataset. I added all the new labeled images to train & val (90/10 split) and kept my 'benchmark' data in the test folder. 
    - Dataset is exported and uploaded to kaggle. We're ready to retrain.
6. train on the data
    - done this with kaggle dataset. But idk if the scripts will work with my own data.
    - can i upload my own dataset to kaggle and use the existing scripts to download/train on them?
        - ok yea kaggle lets me upload my own dataset and gives me a handle. 
        - i just need to make sure the dataset is structured in a way that the template supports (train/val/test directories)
    - Models i currently have:
        - yolo11n.pt (this is the pre-trained model that's pre configured in the notebook)
        - 11192025_1Epoch/best.pt (this is the first training run i did using the original data set)
        - 11192025_20Epoch/best.pt (this is the second training run i did using the original data set. I ran this _on top of_ the 1Epoch/best.pt model)
        - 11212025_1Epoch_newDS/best.pt (this is the first training run i did using the whole new dataset)
    - Benchmarks i want to get:
        - Model | mAP50 | Latency | Notes | Comments
        - 11192025_1Epoch/best.pt | (control)
        - 11192025_20Epoch/best.pt (more training epochs)
        - 11212025_1Epoch_newDS/best.pt (new data)
        - new20epoch_newDS/best.pt (new data & more training)
        - meta/[SAM3](https://github.com/roboflow/notebooks/blob/main/notebooks/how-to-segment-images-with-segment-anything-3.ipynb)
    - Models i need
        - new20epoch_newDS # 20e fresh model using new DS

| Model | mAP50 | Inference time | Hypothesis | Condition | Result | test
| --    | --    | --      | --    | --  | --  | -- |
| 1192025_1Epoch/best.pt | 0.0045 | 26.8ms | Baseline | Control Dataset; 1epoch | Complete fail |
| 11192025_20Epoch/best.pt | 0.00421 | 10.6ms | Extending training duration will make it better | 20E on control dataset | Much faster, but complete fail |
| 11212025_1Epoch_newDS/best.pt | 0.102 | 11.9ms | Domain specific data will improve performance | Aerial dataset; 1 epoch | 20x improvement vs baseline |
| new20epoch_newDS/best.pt | 0.429 | 12.4ms | More training is more better | domain specific data set; 20epochs | 4.2x improvement on 1e model; 95x improvement on baseline | 
| 12022025_350Epoch_newDS/best.pt | 0.504 | 33.4ms | Way more training is way more better | domain specific data set; 350epochs | 4.9x improvement on 1e model; 112x improvement on baseline |
| roboflow | 0.57 | -- | training optimization will yield better results | domain specific data set; 350epochs; trained on roboflow (instead of colab) | 5.6x improvement on 1e model; 126x improvement on baseline |
| meta/SAM3 | -- | -- | -- | SOTA | -- |

 - New model 12022025_350Epoch_newDS/best.pt took ~2h 
- Testing SAM3
    - SAM3 isn't yet supported in ultralytics so i can't just call model.val() to get a 1:1 comparison. 
    - what i'll do instead is i'll run sam3 for "pothole" on each of the images in the test set. 
    - On each image, i'll check how many potholes sam3 counts, and i'll compare it agains the count in the labeled images
    - I'll use that to get a success ratio which i think will serve as a good analogue for mAP50
    - updates: 
        - I modified the roboflow notebook to download the benchmark dataset and iterate through it
        - I'm saving the annotated images to a folder
        - I need to add some logging to save the object counts for each image to a csv or something. 
        - ok the csv is done. I'm going to now try to get polygon positions (to try and compare to mAP)
        - now i've included some code to get polygon positions from SAM3, but they're coming in as absolute pixel coordinates which is different from my test data labels (yolo format)
        - alright, i learned how to convert the absolute coordinates (from SAM3) to relative coordinates that look similar to my yolo test set. 
        - I wrote some code to do that and save a labels.txt file for each image that sam3 processes.
        - now we have a sam3 label file that i can use to compare against the test image label file.
        - calculating the mAP across SAM3 and my test dataset requires some polygon math that is out of my depth. This might be a good next-step
        - theres a thread here about transformer models (sam3) vs CNNs (yolo)
7. Test the hypothesis with the newly trained model
    - I should benchmark my existing model (existing dataset; 20epochs) to get a baseline
    - I should train with new data for 1 epoch and tweak confidence to see where that gets us vs baseline on the measure of success
    - then i can retrain for more epohchs and retest against the measure of success.
    - at this point i can start drawing conclusions
8. determine next steps (either i was right, in which case we're done. Or i was wrong, in which case i start over with a different hypothesis)

