# BERT_Synonym_Replacer
Demonstrates a variety of ways to replace a specified word with its closest synonym.

# Use
Input the filepath to the prompt data as a .xml file in the first argument. Some example data has been provided in lexsub_trial.xml. By default, the program will run a BERT-enabled lexicle substitution routine and print the output.

To use other lexicle substitution methods, input a 1, 2, or 3 as the second argument to specify a mode:
1. A simple lesk algorithm
2. Vector similarity between words (GoogleNews Word2Vec used, but has to be downloaded first. Can be found at this link: https://drive.google.com/u/1/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download WARNING: this file ~1.5GB)
3. A similar application of the BERT masked model, including sentiment of the overall context as additional input

There is also a perl script and reference data attached that allows for evaluation using "perl score.pl <piped lexsub_main output>.predict gold.trial"
