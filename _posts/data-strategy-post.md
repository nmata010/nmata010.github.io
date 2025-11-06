# Notes
I wanna improve my cvis model. Its trained to detect potholes but mostly at street-level. It fails when detecting on far overhead images. I wanna make it better at detecting overhead images. 

This is purely an experiment, but i want to take it seriously, so i'm going to figure out what the right data strategy is for this use-cse.

To start i'm gonna set a really narrow and specific usecase.
**Usecase:** Detect potholes from overhead view on dirt roads with little traffic. 

I think to solve this effectively i need to 
- figure out what problem i wanna solve (see usecase)
- hypothesize what data i need to solve it
- determine a measure of success (how do i know my hypothesis is proven true or false)
- source or create the data
- prepare the data
- train on the data
- Test the hypothesis with the newly trained model
- benchmark the results against the measures of success
- determine next steps (either i was right, in which case we're done. Or i was wrong, in which case i start over with a different hypothesis)