import re
import unicodedata
import nltk
from typing import List, Dict, Any
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from bs4 import BeautifulSoup
import html
from spellchecker import SpellChecker


class TextNormalizer:
    """
    A comprehensive text normalizer for RAG pipelines that handles:
    - HTML/XML tag removal
    - Unicode normalization
    - Case normalization
    - Punctuation removal
    - Whitespace normalization
    - Number handling
    - Stopword removal (optional)
    - Stemming/Lemmatization (optional)
    - Special character handling
    """

    def __init__(
        self,
        lowercase: bool = True,
        remove_html: bool = True,
        remove_punctuation: bool = True,
        normalize_whitespace: bool = True,
        correct_spanish_spelling: bool = True,
        normalize_unicode: bool = True,
        remove_stopwords: bool = False,
        stem_words: bool = False,
        lemmatize_words: bool = False,
        language: str = "spanish",
    ) -> None:
        """
        Initialize the text normalizer with configurable options.

        Args:
            lowercase (bool): Convert text to lowercase.
            remove_html (bool): Remove HTML/XML tags.
            remove_punctuation (bool): Remove punctuation marks.
            normalize_whitespace (bool): Replace multiple whitespaces with a single space.
            normalize_unicode (bool): Normalize Unicode characters to their canonical form.
            remove_stopwords (bool): Remove common stopwords.
            stem_words (bool): Apply stemming to words.
            lemmatize_words (bool): Apply lemmatization to words.
            language (str): Language for stopwords (if remove_stopwords is True).
        """
        self.lowercase: bool = lowercase
        self.remove_html: bool = remove_html
        self.remove_punctuation: bool = remove_punctuation
        self.normalize_whitespace: bool = normalize_whitespace
        self.correct_spanish_spelling: bool = correct_spanish_spelling
        self.normalize_unicode: bool = normalize_unicode
        self.remove_stopwords: bool = remove_stopwords
        self.stem_words: bool = stem_words
        self.lemmatize_words: bool = lemmatize_words
        self.language: str = language

        # onitialize required components based on options
        if self.correct_spanish_spelling:
            self.spell = SpellChecker(language="es", distance=1)  # spanish dictionary.

        if self.remove_stopwords:
            self.stop_words: set[str] = set(stopwords.words(self.language))

        if self.stem_words:
            self.stemmer: PorterStemmer = PorterStemmer()

        if self.lemmatize_words:
            self.lemmatizer: WordNetLemmatizer = WordNetLemmatizer()

    def normalize(self, text: str) -> str:
        """
        Apply the configured normalization steps to the input text.

        Args:
            text (str): Input text to normalize.

        Returns:
            str: Normalized text.
        """
        if text is None or text.strip() == "":
            return ""

        # Handle HTML/XML content
        if self.remove_html:
            text = self._remove_html_tags(text)

        # Unicode normalization
        if self.normalize_unicode:
            text = self._normalize_unicode(text)

        # Case normalization
        if self.lowercase:
            text = text.lower()

        # Punctuation removal
        if self.remove_punctuation:
            text = self._remove_punctuation(text)

        # Whitespace normalization
        if self.normalize_whitespace:
            text = self._normalize_whitespace(text)

        # Tokenize the text for word-level operations
        tokens: List[str] = nltk.word_tokenize(text)

        if self.correct_spanish_spelling:
            tokens = self._correct_spanish_spelling(tokens)

        # Stopword removal
        if self.remove_stopwords:
            tokens = self._remove_stopwords(tokens)

        # Stemming
        if self.stem_words:
            tokens = self._stem_words(tokens)

        # Lemmatization
        if self.lemmatize_words:
            tokens = self._lemmatize_words(tokens)

        # Rejoin tokens into text
        return " ".join(tokens)

    def _remove_html_tags(self, text: str) -> str:
        """Remove HTML/XML tags and decode HTML entities."""
        # First use BeautifulSoup to handle complex HTML
        soup: BeautifulSoup = BeautifulSoup(text, "html.parser")
        text = soup.get_text()

        # Also decode HTML entities
        text = html.unescape(text)

        return text

    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters to their canonical form."""
        return unicodedata.normalize("NFKC", text)

    def _remove_punctuation(self, text: str) -> str:
        """Remove punctuation marks and special characters."""
        # Keep spaces, alphanumeric characters, and remove everything else
        return re.sub(r"[^\w\s]", "", text)

    def _normalize_whitespace(self, text: str) -> str:
        """Replace multiple whitespaces with a single space."""
        # Replace newlines, tabs, and multiple spaces with a single space
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _correct_spanish_spelling(self, tokens: List[str]) -> List[str]:
        """fix the spelling mistakes in spanish."""
        # print(tokens)
        # print(len(tokens), type(tokens))
        words = []
        for token in tokens:
            unknown = self.spell.unknown(
                token
            )  # this is a list of the misspelled words.
            if token == unknown:
                words.append(unknown)
            else:
                words.append(token)

        return words

    def _remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove common stopwords."""
        return [token for token in tokens if token not in self.stop_words]

    def _stem_words(self, tokens: List[str]) -> List[str]:
        """Apply stemming to words."""
        return [self.stemmer.stem(token) for token in tokens]

    def _lemmatize_words(self, tokens: List[str]) -> List[str]:
        """Apply lemmatization to words."""
        return [self.lemmatizer.lemmatize(token) for token in tokens]


def normalize_documents(
    documents: List[str], **normalizer_options: Dict[str, Any]
) -> List[str]:
    """
    Normalize a collection of documents using the TextNormalizer.

    Args:
        documents (List[str]): List of document strings.
        **normalizer_options (Dict[str, Any]): Options to pass to TextNormalizer.

    Returns:
        List[str]: List of normalized document strings.
    """
    normalizer: TextNormalizer = TextNormalizer(**normalizer_options)
    final_normalized_text = [normalizer.normalize(doc) for doc in documents]
    # print("final normalized text: ", final_normalized_text)
    return final_normalized_text
