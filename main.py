import model_zoo
import argparse

def main(args):
    model_name = args.model
    text_path = args.text_path
    output_path = args.output_path

    model = getattr(model_zoo, model_name)
    model_ = model(text_path=text_path, output_path=output_path)
    model_.train()
    generated_summary = model_.summarize() # predict
    print(generated_summary)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-m', '--model', type=str, default='BaseModel', choices=["BaseModel", "HuggingFace"])
    args.add_argument('-t', '--text_path', type=str, default='data/text.txt')
    args.add_argument('-o', '--output_path', type=str, default='data/generated_summary.txt')

    args = args.parse_args()
    main(args)