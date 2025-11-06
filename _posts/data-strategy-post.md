# Notes
I wanna improve my cvis model. Its trained to detect potholes but mostly at street-level. It fails when detecting on far overhead images. I wanna make it better at detecting overhead images. 

This is purely an experiment, but i want to take it seriously, so i'm going to figure out what the right data strategy is for this use-cse.

To start i'm gonna set a really narrow and specific usecase.
**Usecase:** Detect potholes from overhead view on dirt roads with little traffic. 

I think to solve this effectively i need to:
- figure out what problem i wanna solve (see usecase)
- hypothesize what data i need to solve it
    - diverse images of dirt roads taken from overhead
    - there's gotta be more to this
- determine a measure of success (how do i know my hypothesis is proven true or false)
    - idk anything about this yet. i'm gonna do research
    - maybe one approach is to manually annotate one image and then use that as a benchmark? 
    - Or create a set of benchmark images?
    - benchmarking against annotated images will give me some sense of how its performing similar to how we could see mAP improving during training in the original experiment. But what about realworld performance? Since i don't have a 'real' usecase i'm not sure how to benchmark success. i'm gonna need to do some soulsearching here.
- source or create the data
    - In a perfect world i'll find this dataset already annotated (fingers crossed)
    - in the absence of that i'll have to create the dataset and annotate it myself (sounds tedious but idk)
    - For unannotated images/videos i can go online to some stock sites.
    - I think i could also ask gemini to create some images for me? Possibly also videos? 
    - This whole section sounds like its gonna be a grind unless i find some neat dataset that fits. 
    - There may be an arugment for pivoting my use-case around the available datasets since this is not real-world, but that feels like cheating. 
- prepare the data
    - this is kinda the same as above.
    - once i have the images i need to have them annotated
    - idk anything about preparing training data so i need to cover a lot of ground here.
- train on the data
    - done this with kaggle dataset. But idk if the scripts will work with my own data.
    - can i upload my own dataset to kaggle and use the existing scripts to download/train on them?
- Test the hypothesis with the newly trained model
    - I should benchmark my existing model (existing dataset; 20epochs) to get a baseline
    - I should train with new data for 1 epoch and tweak confidence to see where that gets us vs baseline on the measure of success
    - then i can retrain for more epohchs and retest against the measure of success.
    - at this point i can start drawing conclusions
- benchmark the results against the measures of success
    - this feels like its restating the previous point. 
    - maybe i'm missing something, but i can't think of it at the moment. 
- determine next steps (either i was right, in which case we're done. Or i was wrong, in which case i start over with a different hypothesis)