import stanza 

# Initialize the Arabic NLP pipeline with the NER processor
nlp = stanza.Pipeline(lang='ar', processors='tokenize,ner')

# Insert factual non-fictional text in Arabic
text = """
أعلنت الحكومة الأردنية اليوم عن خطط لتعزيز الاقتصاد الوطني من خلال سلسلة من المشاريع الكبرى في عمان والعقبة. وفي تصريح صحفي، قال وزير الاقتصاد عمر الرزاز إن هذه المشاريع ستخلق آلاف الوظائف وتعزز النمو الاقتصادي. كما أشار إلى الدعم الكبير من الدول المانحة، بما في ذلك الإمارات العربية المتحدة والمملكة العربية السعودية، اللتين تعهدتا بتقديم مساعدات مالية لدعم هذه المبادرات.
"""

# Process the text
doc = nlp(text)

# Print named entities
print("Named Entities Found:")
for sentence in doc.sentences:
    for entity in sentence.ents:
        print(f'Entity: {entity.text}, Type: {entity.type}')

