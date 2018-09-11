from src.learning.model_event import evaluate_tagging

annotated_dir = '/home/ymeng/projects/sandbox/val_set_tagged/tagged/'
newsreader_dir = '/home/ymeng/projects/TEA/newsreader_annotations/1-20/'

results = evaluate_tagging(annotated_dir, newsreader_dir)
for key in sorted(results):
    print key, results[key] 
