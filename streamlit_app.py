import streamlit as st

from summarizer import summarize_for_ui


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Text Summarizer",
    page_icon="✨",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title("✨ AI Text Summarizer")

st.write(
    "Transform long text into clear, concise insights "
    "using an open-source AI model."
)

st.divider()


# ============================================================
# INPUT / OUTPUT
# ============================================================

col1, col2 = st.columns(2, gap="large")


# ============================================================
# INPUT
# ============================================================

with col1:

    st.subheader("📝 Your Text")

    uploaded_file = st.file_uploader(
        "Upload a .txt file (optional)",
        type=["txt"]
    )

    input_text = st.text_area(
        "Paste your text here:",
        height=350,
        placeholder=(
            "Paste an article, document, notes, "
            "or any text you want to summarize..."
        )
    )

    # If a file was uploaded, use its contents
    if uploaded_file is not None:

        input_text = uploaded_file.read().decode("utf-8")

        st.info(
            f"📄 Using uploaded file: {uploaded_file.name}"
        )

    input_words = len(input_text.split())

    st.caption(
        f"📄 {input_words:,} words"
    )


# ============================================================
# OUTPUT
# ============================================================

with col2:

    st.subheader("✨ AI Summary")

    summary_placeholder = st.empty()

    summary_placeholder.info(
        "Your summary will appear here."
    )


# ============================================================
# SUMMARIZE BUTTON
# ============================================================

st.write("")

summarize_button = st.button(
    "✨ Summarize",
    type="primary",
    use_container_width=True
)


if summarize_button:

    if not input_text.strip():

        st.warning(
            "Please enter some text before clicking Summarize."
        )

    else:

        with st.spinner(
            "🤖 AI is reading and summarizing your text..."
        ):

            summary, statistics = summarize_for_ui(
                input_text
            )

        # Display summary
        with col2:

            st.subheader("✨ AI Summary")

            st.success(summary)

            st.divider()

            # Statistics
            summary_words = len(summary.split())

            if input_words > 0:

                reduction = (
                    (input_words - summary_words)
                    / input_words
                ) * 100

            else:

                reduction = 0

            st.markdown("### 📊 Statistics")

            stat1, stat2, stat3 = st.columns(3)

            with stat1:

                st.metric(
                    "Input",
                    f"{input_words:,} words"
                )

            with stat2:

                st.metric(
                    "Summary",
                    f"{summary_words:,} words"
                )

            with stat3:

                st.metric(
                    "Reduction",
                    f"{reduction:.1f}%"
                )

            # Download summary
            st.download_button(
                "⬇️ Download Summary",
                data=summary,
                file_name="summary.txt",
                mime="text/plain",
                use_container_width=True
            )


# ============================================================
# EXAMPLE
# ============================================================

st.divider()

st.subheader("💡 Try this example")

example_text = """
Artificial intelligence is transforming many industries around
the world. Machine learning allows computers to identify patterns
in large amounts of data and make predictions.

Natural language processing allows computers to understand and
generate human language, while computer vision enables machines
to interpret images and videos.

These technologies are being used in healthcare, education,
finance, transportation, entertainment, and software development.

However, building reliable AI systems requires careful attention
to data quality, accuracy, privacy, security, computational
resources, and cost.

As artificial intelligence continues to evolve, engineers will
need both machine learning knowledge and strong software
engineering skills to build useful and reliable applications.
"""

st.code(
    example_text,
    language="text"
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Built with Python • Hugging Face Transformers • "
    "DistilBART • Streamlit"
)