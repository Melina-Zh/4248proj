from transformers import pipeline
from .base_model import BaseModel

class HuggingFace(BaseModel):
    def __init__(self, *args, **kwargs):

        # Open and read the text
        if "text_path" in kwargs:
            self.text_path = kwargs['text_path']
        else:
            self.text_path = "./data/text.txt"
        
        if "output_path" in kwargs:
            self.output_path = kwargs['output_path']
        else:
            self.output_path = "./data/generated_summary.txt"

        f = open(self.text_path, "r", encoding="utf8")
        self.text = f.read()
        
        # Initialize the HuggingFace summarization pipeline
        self.summarizer = pipeline("summarization")
        

    def summarize(self):
        # generate, write and return summarized text
        summarized = self.summarizer(self.text, min_length=25, max_length=50)
        # write
        with open(self.output_path,"w") as f:
            f.write(str(summarized))
        # return
        return summarized