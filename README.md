I finetuned a g5-small on billsum to give summaries of text.
Hosted it on huggingface.
And deployed it on streamlit


https://summarizerriver.streamlit.app/


https://www.kaggle.com/code/abhilashdas/summarizer


```markdown
# Text Summarization with T5

This repository contains a script for text summarization using the T5 model from Hugging Face's Transformers library.

## Dependencies

The script requires the following libraries:
- os
- warnings
- kaggle_secrets
- huggingface_hub
- transformers
- datasets
- evaluate
- rouge_score
- numpy

## Usage

The script first sets up the environment and installs necessary libraries. It then loads the 'billsum' dataset from Hugging Face's `datasets` library and splits it into training and testing sets.

The T5 model is used for the task of text summarization. The script includes a function `preprocess_function` for preprocessing the input data.

The script also includes a function `compute_metrics` for computing Rouge scores for the generated summaries.

The model is trained using Hugging Face's `Seq2SeqTrainer` with specified training arguments. The trained model is saved in the '/kaggle/working/basic_model' directory.

Finally, the script includes a sample usage of the trained model for summarizing a given text.

## Note

Please replace the placeholders for 'HUGGINGFACE_TOKEN' and 'wandb_api' with your actual tokens in the following lines:

## python 
```
os.environ['huggingface_read_token'] = user_secrets.get_secret("HUGGINGFACE_TOKEN")
os.environ['WANDB_API_KEY'] = user_secrets.get_secret("wandb_api")
```



![ai sumnmarizer](https://github.com/maximuu19/text_summarizer/assets/46569476/0834a0e3-0b34-4c45-9bf2-ac7d1b3b92a1)
