python truthfulqa.py facebook/opt-125m
python truthfulqa.py facebook/opt-350m
python truthfulqa.py facebook/opt-1.3b
python truthfulqa.py facebook/opt-2.7b
python truthfulqa.py facebook/opt-6.7b

python truthfulqa.py facebook/opt-1.3b --no-demos
# python truthfulqa.py facebook/opt-1.3b 
python truthfulqa.py facebook/opt-1.3b --system-prompt 'Actually,' --no-demos
python truthfulqa.py facebook/opt-1.3b --system-prompt 'Actually,'
