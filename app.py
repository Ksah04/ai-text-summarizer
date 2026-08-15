from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import gradio as gr


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

# Maximum number of tokens we will give the model at once.
MAX_INPUT_TOKENS = 900

# Number of tokens used when splitting long text.
CHUNK_SIZE = 700

# Number of tokens allowed in each generated summary.
MAX_SUMMARY_TOKENS = 120

# Minimum number of tokens in a generated summary.
MIN_SUMMARY_TOKENS = 30


# ============================================================
# 3. SUMMARIZE ONE PIECE OF TEXT
# ============================================================

def summarize_chunk(text):
    """
    Summarizes one piece of text using DistilBART.
    """

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
    """
    Splits long text into smaller pieces based on tokens.
    """

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
    """
    Main function used by the Gradio interface.
    """

    # --------------------------------------------------------
    # Check whether the user entered anything
    # --------------------------------------------------------

    if not text or not text.strip():
        return "Please enter some text to summarize."


    # --------------------------------------------------------
    # Count tokens
    # --------------------------------------------------------

    tokens = tokenizer.encode(
        text,
        add_special_tokens=False
    )

    token_count = len(tokens)


    # --------------------------------------------------------
    # Short text
    # --------------------------------------------------------

    if token_count <= MAX_INPUT_TOKENS:

        summary = summarize_chunk(text)

        return summary


    # --------------------------------------------------------
    # Long text
    # --------------------------------------------------------

    chunks = create_chunks(text)

    summaries = []

    for chunk in chunks:

        chunk_summary = summarize_chunk(chunk)

        summaries.append(chunk_summary)


    # --------------------------------------------------------
    # Combine chunk summaries
    # --------------------------------------------------------

    combined_summary = " ".join(summaries)


    # --------------------------------------------------------
    # Final summarization pass
    # --------------------------------------------------------

    final_summary = summarize_chunk(combined_summary)

    return final_summary


# ============================================================
# 6. GRADIO INTERFACE
# ============================================================

def summarize_for_ui(text):
    """
    Wrapper around the summarization function.

    Returns:
        summary
        statistics
    """

    if not text or not text.strip():
        return (
            "",
            "Please enter some text before clicking Summarize."
        )

    # Count input words
    input_words = len(text.split())

    # Count input tokens
    input_tokens = len(
        tokenizer.encode(
            text,
            add_special_tokens=False
        )
    )

    # Generate summary
    summary = summarize_text(text)

    # Count summary words
    summary_words = len(summary.split())

    # Calculate compression
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
# 7. CUSTOM GRADIO APPLICATION
# ============================================================

custom_css = """
#title {
    text-align: center;
    margin-bottom: 0.2rem;
}

#subtitle {
    text-align: center;
    opacity: 0.75;
    margin-bottom: 1.5rem;
}

#input-box textarea {
    font-size: 16px !important;
}

#output-box textarea {
    font-size: 16px !important;
}

#stats {
    text-align: center;
    font-size: 14px;
    opacity: 0.8;
}

#summarize-button {
    font-size: 17px;
    font-weight: 600;
}
"""


with gr.Blocks(
    theme=gr.themes.Soft(),
    css=custom_css,
    title="AI Text Summarizer"
) as demo:

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    gr.Markdown(
        """
        # ✨ AI Text Summarizer
        """,
        elem_id="title"
    )

    gr.Markdown(
        """
        Transform long text into clear, concise insights
        using an open-source AI model.
        """,
        elem_id="subtitle"
    )


    # --------------------------------------------------------
    # MAIN WORKSPACE
    # --------------------------------------------------------

    with gr.Row():

        # INPUT
        with gr.Column():

            gr.Markdown("### 📝 Your Text")

            input_text = gr.Textbox(
                placeholder=(
                    "Paste an article, notes, document, "
                    "or any text you want to summarize..."
                ),
                lines=15,
                label="",
                elem_id="input-box"
            )


        # OUTPUT
        with gr.Column():

            gr.Markdown("### ✨ AI Summary")

            output_text = gr.Textbox(
                placeholder=(
                    "Your summary will appear here..."
                ),
                lines=15,
                label="",
                elem_id="output-box"
            )


    # --------------------------------------------------------
    # BUTTONS
    # --------------------------------------------------------

    with gr.Row():

        summarize_button = gr.Button(
            "✨ Summarize",
            variant="primary",
            elem_id="summarize-button"
        )

        clear_button = gr.ClearButton(
            [
                input_text,
                output_text
            ],
            value="🗑️ Clear"
        )


    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    statistics = gr.Markdown(
        "📊 Statistics will appear after summarization.",
        elem_id="stats"
    )


    # --------------------------------------------------------
    # EXAMPLE
    # --------------------------------------------------------

    gr.Markdown("### 💡 Try an example")

    gr.Examples(
        examples=[
            [
                """
                Artificial intelligence is transforming many
                industries around the world. Machine learning
                allows computers to identify patterns in data
                and make predictions. Natural language processing
                allows computers to understand and generate human
                language, while computer vision enables machines
                to interpret images and videos.

                These technologies are being used in healthcare,
                education, finance, transportation, entertainment,
                and software development. However, building
                reliable AI systems requires careful consideration
                of data quality, accuracy, privacy, security,
                computational resources, and cost.

                As AI continues to evolve, engineers will need
                both machine learning knowledge and strong software
                engineering skills to build useful and reliable
                applications.
                """
            ]
        ],
        inputs=input_text
    )


    # --------------------------------------------------------
    # BUTTON ACTION
    # --------------------------------------------------------

    summarize_button.click(
        fn=summarize_for_ui,
        inputs=input_text,
        outputs=[
            output_text,
            statistics
        ]
    )


# ============================================================
# 8. START APPLICATION
# ============================================================

if __name__ == "__main__":

    demo.launch()

# ============================================================
# 7. START THE APPLICATION
# ============================================================

if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Soft(),
        css=custom_css
    )