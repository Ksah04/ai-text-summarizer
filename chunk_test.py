from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -----------------------------------
# 1. Load the model and tokenizer
# -----------------------------------

model_name = "google-t5/t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


# -----------------------------------
# 2. Example long text
# -----------------------------------

text = """
Artificial intelligence is transforming many industries around the world.
Companies are using artificial intelligence to analyze large amounts of data,
automate repetitive tasks, understand human language, recognize images,
generate content, improve customer service, detect fraud, and assist
software developers.

Machine learning is one of the major technologies behind modern artificial
intelligence. Instead of explicitly programming every rule, machine learning
systems can learn patterns from data. These systems can then use those
patterns to make predictions or decisions when they encounter new data.

Generative artificial intelligence has expanded the possibilities even
further. Modern generative models can create text, images, audio, video,
and computer code. Large language models can understand and generate human
language and are now being used in chatbots, programming assistants,
research tools, education platforms, and business applications.

However, developing useful artificial intelligence systems involves more
than simply choosing a model. Developers must think about data quality,
model performance, computational resources, privacy, security, reliability,
cost, and how users will interact with the system.

As artificial intelligence continues to develop, engineers will increasingly
need to understand both the underlying machine learning concepts and the
software engineering required to turn those models into reliable applications.

Artificial intelligence is also changing the way people interact with
software. Traditional software generally follows instructions written by
developers, while machine learning systems can learn patterns from examples.
This makes it possible to build applications that can recognize speech,
classify images, translate languages, recommend products, and generate
creative content.

Natural language processing is another important area of artificial
intelligence. NLP systems allow computers to process human language and
perform tasks such as translation, sentiment analysis, summarization,
question answering, and information extraction.

Computer vision focuses on helping computers understand images and video.
Modern vision models can identify objects, classify images, segment parts
of an image, detect faces, read text from photographs, and analyze medical
images.

AI systems are also increasingly being integrated into software development.
Programming assistants can generate code, explain existing programs,
identify potential errors, write tests, and help developers understand
unfamiliar technologies. However, generated code still needs to be reviewed
and tested by humans.

Another important consideration is responsible AI. Models can sometimes
produce incorrect information, reflect biases present in their training
data, or behave unpredictably in unusual situations. Engineers therefore
need to evaluate models carefully before deploying them to real users.

The future of artificial intelligence will depend not only on larger models
but also on better data, efficient algorithms, improved hardware, reliable
software engineering, and thoughtful approaches to safety and responsible
deployment.
"""


# -----------------------------------
# 3. Convert text into tokens
# -----------------------------------

tokens = tokenizer.encode(
    text,
    add_special_tokens=False
)

print("Total tokens:", len(tokens))


# -----------------------------------
# 4. Split into chunks
# -----------------------------------

chunk_size = 400

chunks = []

for i in range(0, len(tokens), chunk_size):
    chunk_tokens = tokens[i:i + chunk_size]

    chunk_text = tokenizer.decode(
        chunk_tokens,
        skip_special_tokens=True
    )

    chunks.append(chunk_text)


print("Number of chunks:", len(chunks))


# -----------------------------------
# 5. Summarize each chunk
# -----------------------------------

summaries = []

for i, chunk in enumerate(chunks):

    print(f"\nSummarizing chunk {i + 1}...")

    prompt = "summarize: " + chunk

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=100
    )

    summary = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    summaries.append(summary)

    print("Summary:")
    print(summary)


# -----------------------------------
# 6. Combine all summaries
# -----------------------------------

final_summary = " ".join(summaries)

print("\n==============================")
print("FINAL SUMMARY")
print("==============================")
print(final_summary)