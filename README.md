# Manga Dialogue Extraction and Speaker Attribution



Approach;

The goal was to extract dialogues from three consecutive manga pages , maintain the reading order and speaker identities across each sequence. This pipeline combines a YOLO-based comic speech bubble detector that identifies the dialogue regions , PaddleOCR for text recognition, MAGI provides visual representations, and then a Random Forest classifier was used to estimate whether two dialogue regions belong to the same speaker , and clustering to assign the speaker IDs



## Experiments and Comparisons

- Fine tuning the OCR based on 434 training crops and it was evaluated on 73 held-out crops , and the pretrained OCR actually performed better with the accuracy 0.6881 , compared with 0.4556 after fine tuning. Thus I retained the pretrained model in the pipeline
- After fine tuning the MAGI model weights for crop embeddings there was only marginal improvement observed (ROC-AUC: 0.7938 → 0.7942; average precision: 0.6714 → 0.6723)
- For the comic speech-bubble detector , fine tuning was not attempted because suitable ground truth bounding boxes were unavailable.
- For speaker matching , I tested different pairwise matching thresholds and actually a  0.45 threshold performed better than 0.60 one in the validation experiments, so that was retained in the model



## What Worked and What Didn't



Basically what posed as the biggest challenge for me was combining each model's most suitable version for the project ,since each models performance directly affected the other's, the error propagates and compunds with each layer.  The pretrained models I used provided a good workable starting point , fine tuning them did not really work for this pipeline, making marginal and infact negative difference . Validation experiments were useful helped comparing the configurations before making the decisions. However , the results are nowhere I wanted them to be, with speaker clustering being one of the weakest point, along with maintaining their identities across a sequence was the most challenging part.



## What I Would Try Next

The model is definitely not in the best shape yet, improvement is required in the reading order reconstruction , some speech bubble boxes still go unidentified by the model , while the performance of OCR is not the best with stylized fonts , overlapping and confusing texts. and optimization is also required in speaker clustering , for which like instead of relying only on pairwise classification metrics , I would evaluate it across more sequences. basically instead of complicating individual models more complex , I'd focus on the overall final sequence level predictions for evaluation



## External Code and Model References

This project builds on the following external tools and pretrained models:

- **Comic Text Detector:** [https://github.com/dmMaze/comic-text-detector](https://github.com/dmMaze/comic-text-detector)
- **PaddleOCR:** [https://github.com/PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- **MAGI:** [https://huggingface.co/ragavsachdeva/magiv2](https://huggingface.co/ragavsachdeva/magiv2)
- **YOLOv8:** [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- **scikit-learn:** [https://scikit-learn.org/](https://scikit-learn.org/)



## Reproducibility

Experiment metrics and comparison results are included in the repository. Inference requires the appropriate dependencies and model artifacts to be available at the paths configured in the code. The required model artifacts are available here:  [https://github.com/batrakavya23-cmyk/Kavya_MangaTask/releases/tag/model-artifacts-v1](https://github.com/batrakavya23-cmyk/Kavya_MangaTask/releases/tag/model-artifacts-v1) Download the artifacts and place them at the paths expected by the inference code before running it
