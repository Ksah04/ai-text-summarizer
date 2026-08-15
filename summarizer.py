from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ============================================================
# 1. MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "sshleifer/distilbart-cnn-12-6"

print("Loading AI model...")
print("This may take a little while the first time.")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

print("Model loaded successfully!")


# ============================================================
# 2. SUMMARIZATION SETTINGS
# ============================================================

MAX_INPUT_TOKENS = 900
CHUNK_SIZE = 700

MAX_SUMMARY_TOKENS = 120
MIN_SUMMARY_TOKENS = 30


# ============================================================
# 3. SUMMARIZE ONE CHUNK
# ============================================================

def summarize_chunk(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_SUMMARY_TOKENS,
        min_new_tokens=MIN_SUMMARY_TOKENS,
        num_beams=4,
        no_repeat_ngram_size=3,
        early_stopping=True
    )

    summary = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return summary


# ============================================================
# 4. SPLIT LONG TEXT INTO CHUNKS
# ============================================================

def create_chunks(text):

    tokens = tokenizer.encode(
        text,
        add_special_tokens=False
    )

    chunks = []

    for i in range(0, len(tokens), CHUNK_SIZE):

        chunk_tokens = tokens[i:i + CHUNK_SIZE]

        chunk_text = tokenizer.decode(
            chunk_tokens,
            skip_special_tokens=True
        )

        chunks.append(chunk_text)

    return chunks


# ============================================================
# 5. MAIN SUMMARIZATION FUNCTION
# ============================================================

def summarize_text(text):

    if not text or not text.strip():
        return "Please enter some text to summarize."

    tokens = tokenizer.encode(
        text,
        add_special_tokens=False
    )

    token_count = len(tokens)

    # Short text
    if token_count <= MAX_INPUT_TOKENS:

        return summarize_chunk(text)

    # Long text
    chunks = create_chunks(text)

    summaries = []

    for chunk in chunks:

        chunk_summary = summarize_chunk(chunk)

        summaries.append(chunk_summary)

    # Combine summaries
    combined_summary = " ".join(summaries)

    # Final summarization
    final_summary = summarize_chunk(combined_summary)

    return final_summary


# ============================================================
# 6. GENERATE STATISTICS
# ============================================================

def summarize_for_ui(text):

    if not text or not text.strip():

        return (
            "",
            "Please enter some text before clicking Summarize."
        )

    input_words = len(text.split())

    input_tokens = len(
        tokenizer.encode(
            text,
            add_special_tokens=False
        )
    )

    summary = summarize_text(text)

    summary_words = len(summary.split())

    if input_words > 0:

        reduction = (
            (input_words - summary_words)
            / input_words
        ) * 100

    else:

        reduction = 0

    statistics = (
        f"📄 Input: {input_words:,} words • "
        f"{input_tokens:,} tokens\n"
        f"✨ Summary: {summary_words:,} words • "
        f"📉 {reduction:.1f}% shorter"
    )

    return summary, statistics


# ============================================================
# 7. TEST THE AI BRAIN
# ============================================================

if __name__ == "__main__":

    test_text = """
    Artificial intelligence is transforming many industries around
    the world. Machine learning allows computers to identify patterns
    in large amounts of data and use those patterns to make predictions.
    Natural language processing allows computers to understand and
    generate human language, while computer vision allows machines to
    interpret images and videos.

    These technologies are being used in healthcare, education,
    finance, transportation, entertainment, and software development.
    However, building reliable AI systems requires careful attention
    to data quality, accuracy, privacy, security, computational
    resources, and cost.
    """

    summary, statistics = summarize_for_ui(test_text)

    print("\n==============================")
    print("SUMMARY")
    print("==============================")
    print(summary)

    print("\n==============================")
    print("STATISTICS")
    print("==============================")
    print(statistics) 