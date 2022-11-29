import evaluate
rouge = evaluate.load('rouge')
pred = ['我是人']
ref = ['我是人']

predictions = ["hello there"]
references = ["hello here"]

result = rouge.compute(predictions=pred, references=ref)
print(result)

results = rouge.compute(predictions=predictions,references=references)
print(results)