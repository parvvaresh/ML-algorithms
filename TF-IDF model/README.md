**TF-IDF**

TF-IDF or ( Term Frequency(TF) — Inverse Dense Frequency(IDF) )is a technique which is used to find meaning of sentences consisting of words and cancels out the incapabilities of Bag of Words technique which is good for text classification or for helping a machine read words in numbers. However, it just blows up in your face when you ask it to understand the meaning of the sentence or the document.




I highly suggest you read about BoW before you go through this article to get a context -



---



**So what is it, do you want to understand using an example ?**



Let’s say a machine is trying to understand meaning of this —


```
Today is a beautiful day

```


What do you focus on here but tell me as a human not a machine?

This sentence talks about today, it also tells us that today is a beautiful day. The mood is happy/positive, anything else cowboy?

Beauty is clearly the adjective word used here. From a BoW approach all words are broken into count and frequency with no preference to a word in particular, all words have same frequency here (1 in this case)and obviously there is no emphasis on beauty or positive mood by the machine.

The words are just broken down and if we were talking about importance, ‘a’ is as important as ‘day’ or ‘beauty’.

But is it really that ‘a’ tells you more about context of a sentence compared to ‘beauty’ ?

No, that’s why Bag of words needed an upgrade.

Also, another major drawback is say a document has 200 words, out of which ‘a’ comes 20 times, ‘the’ comes 15 times etc.

Many words which are repeated again and again are given more importance in final feature building and we miss out on context of less repeated but important words like Rain, beauty, subway , names.

So it’s easy to miss on what was meant by the writer if read by a machine and it presents a problem that TF-IDF solves, so now we know why do we use TF-IDF.

---

**Let’s now see how does it work, okay?**




TF-IDF is useful in solving the major drawbacks of Bag of words by introducing an important concept called inverse document frequency.

It’s a score which the machine keeps where it is evaluates the words used in a sentence and measures it’s usage compared to words used in the entire document. In other words, it’s a score to highlight each word’s relevance in the entire document. It’s calculated as -



```

IDF =Log[(# Number of documents) / (Number of documents containing the word)] and

TF = (Number of repetitions of word in a document) / (# of words in a document)


```

okay, for now let’s just say that TF answers questions like — how many times is beauty used in that entire document, give me a probability and IDF answers questions like how important is the word beauty in the entire list of documents, is it a common theme in all the documents.

So using TF and IDF machine makes sense of important words in a document and important words throughout all documents.
