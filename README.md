# 🩺💬 Doctor Scribbles - NorthAM Hackathon Submission

![doctor-patient-stock-image.jpeg](https://storage.googleapis.com/aadev-2541-public-assets/doctor-scribbles-img-1.png)

---

**Speech to Text 🗣️🔠 + Translation 🌎💬  + PaLM 🌴🦬**

---

| | |
|----------|-------------|
| Author(s)   | Ayo Adedeji (ayoad@) |
| Contributor(s) | Ayo Adedeji (ayoad@), Sarita Joshi (saritajoshi@)
| Last updated | 08/11/2023 |

## Overview

We seamlessly integrate Google Cloud's Speech-to-Text 🗣️🔠 and Translations APIs 🌎💬 to enable real-time transcription and translation of medical conversations between doctors and patients. Leveraging Google Cloud's PaLM API  🌴, our solution enhances transcriptions with __contextual speaker diarization__, ensuring accurate and nuanced transcripts that capture the essence of the conversation. Our solution also performs __advanced transcript correction__. It discerns subtle nuances, such as distinguishing between "cough" and "cup," based on prior context, ensuring the utmost accuracy and clarity in transcriptions.

Enter chat-bison 🦬, our intelligent, stateful chatbot. As conversations unfold, this AI-powered assistant dynamically tracks context and sentiment, generating __real-time recommendations__. From potential diagnoses and personalized treatment plans to task recommendations and actionable items, chat-bison empowers medical professionals and patients alike with insightful guidance.

This comprehensive system promises to revolutionize medical communication by offering informed insights and actionable suggestions, ultimately improving patient care and enhancing the decision-making process for healthcare providers

## Costs

This tutorial uses billable components of Google Cloud Platform (GCP):

- [Speech-to-Text API](https://cloud.google.com/speech-to-text)
- [Translation API](https://cloud.google.com/translate)
- [Vertex AI LLM APIs](https://cloud.google.com/vertex-ai/pricing#generative_ai_models)
