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
    - I uploded them to roboflow and used meta's SAM3 to quckly identify the potholes (the irony is not lost on me that i'm using a FM to detect potholes in my training data to train a model to detect potholes.)
    - SAM3 is SOTA so i don't feel bad that its better than my model at detecting potholes. But i did note that it really struggles with overhead images, dirt roads, and dealing with the non-road background. 
    - Anyway sam3 did a pretty good job on a couple of the videos and completely fails at 2 others. 
    - I manually labeled the other remaining videos. That was very tedious. Data labeling is a grind.
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

| Model | mAP50 | Inference time | Hypothesis | Condition | Result
| --    | --    | --      | --    | --  | --  |
| 1192025_1Epoch/best.pt | 0.0045 | 26.8ms | Baseline | Control Dataset; 1epoch | Complete fail
| 11192025_20Epoch/best.pt | 0.00421 | 10.6ms | Extending training duration will make it better | 20E on control dataset | Much faster, but complete fail
| 11212025_1Epoch_newDS/best.pt | 0.102 | 11.9ms | Domain specific data will improve performance | Aerial dataset; 1 epoch | 20x improvement vs baseline
| new20epoch_newDS/best.pt | 0.429 | 12.4ms | More training is more better | domain specific data set; 20epochs | 4x improvement on 1e model; 95x improvement on baseline
| meta/SAM3 | -- | -- | -- | SOTA |


- Testing SAM3
    - SAM3 isn't yet supported in ultralytics so i can't just call model.val() to get a 1:1 comparison. 
    - what i'll do instead is i'll run sam3 for "pothole" on each of the images in the test set. 
    - On each image, i'll check how many potholes sam3 counts, and i'll compare it agains the count in the labeled images
    - I'll use that to get a success ratio which i think will serve as a good analogue for mAP50
    - updates: 
        - I modified the roboflow notebook to download the benchmark dataset and iterate through it
        - I'm saving the annotated images to a folder
        - I need to add some logging to save the object counts for each image to a csv or something. 
7. Test the hypothesis with the newly trained model
    - I should benchmark my existing model (existing dataset; 20epochs) to get a baseline
    - I should train with new data for 1 epoch and tweak confidence to see where that gets us vs baseline on the measure of success
    - then i can retrain for more epohchs and retest against the measure of success.
    - at this point i can start drawing conclusions
8. determine next steps (either i was right, in which case we're done. Or i was wrong, in which case i start over with a different hypothesis)