# Email-Analyzer-Web-App
This application analyzes email datasets (or any text dataset) to find patterns in the language, especially patterns that may indicate phishing or spam.

1. Loading the Dataset
   
The application begins by loading a CSV file. 
It tries two different reading methods to make sure it can handle messy data, such as text fields that contain line breaks or quotation marks.

2. Detecting Email-Specific Features (When Available)
   
If the dataset looks like an email dataset (for example, if it contains columns like subject, body, sender, or date), the app automatically creates helpful additional features. These include:
• Combining subject and body into one full text field.

• Detecting whether a URL (web link) appears in the email.

• Extracting the sender’s email domain.

• Converting the date into readable time information (hour and weekday).

These extra features make it easier to analyze suspicious behavior, such as whether phishing emails contain more links than legitimate emails.

4. Choosing the Text for Analysis

The app automatically selects the best text column for analysis. For email datasets, it prefers the combined subject and body. This ensures the analysis uses enough information and does not rely only on short subject lines.

5. Text Preprocessing (Cleaning the Text)

Before analyzing the text, the app cleans it. This process is called preprocessing. Preprocessing helps remove noise and makes the text easier to analyze.
The cleaning process includes:

• Converting all words to lowercase.

• Removing punctuation marks.

• Removing common words like 'the', 'and', 'is' (called stopwords).

• Breaking sentences into individual words (called tokenization).

After this step, the text is transformed into a list of meaningful words that can be measured and compared.

6. Basic Data Exploration

The app then performs basic exploration of the dataset. It shows:

• How many rows and columns exist.

• Whether any values are missing.

• How long the messages are.

• How the labels (for example, phishing vs legitimate) are distributed.

This step helps us understand the structure of the dataset before diving deeper into language analysis.

7. Word Clouds

The app creates word clouds. A word cloud is a visual display where frequently used words appear larger. This provides a quick visual summary of the most common words in the dataset.
Word clouds are helpful for spotting patterns quickly, but they are only a starting point. The app also provides more detailed analysis for stronger evidence.

8. Token and Phrase Analysis

The app counts how often each word appears (called token frequency). It also looks at common two-word phrases (called bigrams), such as 'verify account' or 'click link'.
These phrase patterns are especially important in phishing detection because they reveal intent.

9. Part-of-Speech Analysis

The app identifies the grammatical role of words (such as nouns and verbs). This helps describe writing style. For example, phishing messages may use more action verbs to encourage immediate action.

10. Sentiment Analysis

The app measures emotional tone using a method called VADER sentiment analysis. It produces a score that shows whether a message sounds positive, negative, or neutral.
Phishing messages may use urgency or emotional pressure, which can sometimes be detected through sentiment patterns.

11. Phishing vs Legitimate Comparison

If a label column is available, the app compares phishing and legitimate messages side by side.
It shows:

• Which words are most common in each class.

• Which phrases are most common.

• Which terms are most distinctive between classes using TF-IDF.

This comparison helps explain exactly how phishing language differs from normal communication.

12. Exporting the Processed Data

Finally, the app allows the user to download the processed dataset. This makes it easy to use the cleaned and enriched data for further analysis or modeling.

In summary, this pipeline is designed not only to analyze text but to explain what the analysis means. It combines data exploration, text cleaning, language pattern detection, and comparison tools to provide a clear and understandable view of phishing behavior. The focus is on interpretation and insight, not just numerical accuracy.
