# Notes
I wanna improve my cvis model. Its trained to detect potholes but mostly at street-level. It fails when detecting on far overhead images. I wanna make it better at detecting overhead images. 

This is purely an experiment, but i want to take it seriously, so i'm going to figure out what the right data strategy is for this use-cse.

To start i'm gonna set a really narrow and specific usecase.
**Usecase:** Detect potholes from overhead view on dirt roads with little traffic. 

I think to solve this effectively i need to:
1. figure out what problem i wanna solve (see usecase)
2. hypothesize what data i need to solve it
    - diverse images of dirt roads taken from overhead
    - there's gotta be more to this
    - different weather? Different angles?
3. determine a measure of success (how do i know my hypothesis is proven true or false)
    - idk anything about this yet. i'm gonna do research
    - maybe one approach is to manually annotate one image and then use that as a benchmark? 
    - Or create a set of benchmark images?
        - i can get some frames (20?) from the video i used before and manualy annotate every single pothole to use as a benchmark set
        - I'll hve to learn how to benchmark images. but that's doable
    - success could be achieving mAP50 of 0.6
        - i have to run my 20e model on the benchmark dataset to get a baseline, and then i can set a success metric as a function of that. So like if my 20e model gets to mAP50 of 0.1, then i could target mAP50 of 0.5
    - benchmarking against annotated images will give me some sense of how its performing similar to how we could see mAP improving during training in the original experiment. But what about realworld performance? Since i don't have a 'real' usecase i'm not sure how to benchmark success. i'm gonna need to do some soulsearching here.
4. source or create the data
    - In a perfect world i'll find this dataset already annotated (fingers crossed)
    - in the absence of that i'll have to create the dataset and annotate it myself (sounds tedious but idk)
        - roboflow and CVAT have tools to do this. Gotta look into this
    - For unannotated images/videos i can go online to some stock sites.
    - I think i could also ask gemini to create some images for me? Possibly also videos? 
        - looks like this is a whole category of data called "synthetic data" which has its own rabbit holes. I'll try it out, but will abandon if needed. 
    - This whole section sounds like its gonna be a grind unless i find some neat dataset that fits. 
    - There may be an arugment for pivoting my use-case around the available datasets since this is not real-world, but that feels like cheating. 
5. prepare the data
    - this is kinda the same as above.
    - annotated images
    - text for each image with the box information (yolo.txt)
    - idk anything about preparing training data so i need to cover a lot of ground here.
        - Roboflow has a tool to annotate and export. 
6. train on the data
    - done this with kaggle dataset. But idk if the scripts will work with my own data.
    - can i upload my own dataset to kaggle and use the existing scripts to download/train on them?
        - ok yea kaggle lets me upload my own dataset and gives me a handle. 
        - i just need to make sure the dataset is structured in a way that the template supports (train/val/test directories)
7. Test the hypothesis with the newly trained model
    - I should benchmark my existing model (existing dataset; 20epochs) to get a baseline
    - I should train with new data for 1 epoch and tweak confidence to see where that gets us vs baseline on the measure of success
    - then i can retrain for more epohchs and retest against the measure of success.
    - at this point i can start drawing conclusions
8. determine next steps (either i was right, in which case we're done. Or i was wrong, in which case i start over with a different hypothesis)