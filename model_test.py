from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# -----------------------------------
# 1. Choose the model
# -----------------------------------

MODEL_NAME = "sshleifer/distilbart-cnn-12-6"


# -----------------------------------
# 2. Load tokenizer
# -----------------------------------

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# -----------------------------------
# 3. Load model
# -----------------------------------

print("Loading model...")

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


print("Model loaded successfully!")


# -----------------------------------
# 4. Text to summarize
# -----------------------------------

text = """
Artificial intelligence has rapidly transformed the way people interact with technology.
Modern AI systems can understand natural language, recognize objects in images,
generate text and software code, translate languages, analyze large datasets,
and assist people in making decisions.

These capabilities are being used across healthcare, education, finance,
entertainment, transportation, and software development.

However, building useful AI applications involves more than simply selecting
a powerful model. Developers must also consider data quality, accuracy,
computational resources, privacy, security, cost, and the experience of
the people using the application.

As AI continues to evolve, engineers who understand both machine learning
models and the software systems surrounding them will be increasingly important.
"""


# -----------------------------------
# 5. Tokenize the text
# -----------------------------------

inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True
)


# -----------------------------------
# 6. Generate summary
# -----------------------------------

print("Generating summary...")

outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    min_new_tokens=30,
    num_beams=4,
    no_repeat_ngram_size=3
)


# -----------------------------------
# 7. Convert tokens back to text
# -----------------------------------

summary = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)


# -----------------------------------
# 8. Display result
# -----------------------------------

print("\n==============================")
print("SUMMARY")
print("==============================")
print(summary)